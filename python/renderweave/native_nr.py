"""Thin ctypes entry for the experimental native C++/CUDA stateless NR SDK.

No framework, tensor operator, normalization, resampling or JIT lives here.
Input is contiguous float32 HWC, sRGB code values in [0,1]. CPU submit copies
the source; submit_cuda borrows device planes and their producer-ready event.
"""
from __future__ import annotations

from array import array
import ctypes as c
import json
import os
import sys
import threading
from itertools import count
from functools import wraps

from ._core import CoreError, _sdk_library_path

ABI = 1
NO_PREDECESSOR = (1 << 64) - 1


class _RuntimeDesc(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32),
                ("flags", c.c_uint32), ("device_ordinal", c.c_int32)] + [
        (name, c.c_uint32) for name in ("max_sessions", "max_inflight", "trace_capacity", "reserved")
    ] + [("device_budget_bytes", c.c_uint64), ("host_budget_bytes", c.c_uint64)] + [
        (name, c.c_char_p) for name in ("weights_path", "weights_sha256", "parameter_manifest_path",
                                      "parameter_manifest_sha256", "plan_path", "plan_sha256")]


class _SessionDesc(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_size", "abi_version", "flags", "seed")] + [
        ("epoch", c.c_uint64), ("predecessor", c.c_uint64)]


class _SessionState(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_size", "abi_version", "accepting", "pending")] + [
        (name, c.c_uint64) for name in ("session_id", "epoch", "predecessor")]


class _Frame(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_size", "abi_version", "flags", "reserved")] + [
        (name, c.c_uint64) for name in ("frame_id", "epoch", "expected_predecessor")
    ] + [("width", c.c_uint32), ("height", c.c_uint32),
         ("color_hwc", c.POINTER(c.c_float)), ("color_elements", c.c_uint64)]


_CudaRelease = c.CFUNCTYPE(None, c.c_void_p)


class _CudaInput(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32),
                ("motion_hwc2", c.POINTER(c.c_float)), ("motion_elements", c.c_uint64),
                ("motion_validity", c.POINTER(c.c_float)), ("motion_validity_elements", c.c_uint64),
                ("ready_event", c.c_void_p), ("release", _CudaRelease), ("release_context", c.c_void_p)]


_cuda_owners, _cuda_owner_ids, _cuda_owners_lock = {}, count(1), threading.Lock()


@_CudaRelease
def _release_cuda_owners(cookie):
    with _cuda_owners_lock:
        owners = _cuda_owners.pop(cookie, None)
    # Destruction, including a producer's device deallocation, occurs outside
    # the registry lock. C++ invokes this on a CPU thread after physical drain.


def _cuda_plane(value, height, width, channels):
    """Read device metadata only; no conversion, host copy or CUDA operation."""
    if value is None:
        return c.POINTER(c.c_float)(), 0
    shape = (height, width, channels)
    interface = getattr(value, "__cuda_array_interface__", None)
    if interface is not None:
        if tuple(interface["shape"]) != shape or interface["typestr"] not in ("<f4", "=f4"):
            raise ValueError("CUDA input must have exact HWC float32 shape")
        strides = interface.get("strides")
        if strides is not None and tuple(strides) != (width*channels*4, channels*4, 4):
            raise ValueError("CUDA input must be contiguous")
        pointer = int(interface["data"][0])
    elif getattr(value, "is_cuda", False) and hasattr(value, "data_ptr"):
        if tuple(value.shape) != shape or str(value.dtype) != "torch.float32" or not value.is_contiguous():
            raise ValueError("CUDA tensor must have exact contiguous HWC float32 shape")
        pointer = int(value.data_ptr())
    else:
        raise ValueError("input must expose CUDA array metadata or a CUDA float32 tensor")
    if not pointer or pointer % 4:
        raise ValueError("CUDA float32 pointer is null or unaligned")
    return c.cast(c.c_void_p(pointer), c.POINTER(c.c_float)), height*width*channels


class _JobInfo(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_size", "abi_version", "state", "physical_completion_confirmed")] + [
        ("status", c.c_int32), ("gpu_submitted", c.c_uint32)
    ] + [(name, c.c_uint64) for name in ("session_id", "frame_id", "epoch", "expected_predecessor",
        "submit_host_ns", "admitted_host_ns", "device_execution_host_ns", "complete_host_ns",
        "output_available_host_ns", "first_wait_timeout_host_ns")
    ] + [("device_stream_span_ms", c.c_double), ("error", c.c_char * 512)]


class _ResultView(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_size", "abi_version", "width", "height")] + [
        ("color_hwc", c.POINTER(c.c_float))
    ] + [(name, c.c_uint64) for name in ("color_elements", "session_id", "frame_id", "epoch")]


def _u64(value):
    if type(value) is not int or not 0 <= value < (1 << 64):
        raise ValueError("expected unsigned 64-bit integer")
    return value


def _u32(value):
    if type(value) is not int or not 0 <= value < (1 << 32):
        raise ValueError("expected unsigned 32-bit integer")
    return value


def _bounded_u32(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _timeout(value):
    if type(value) is not int or not -1 <= value <= 86400000:
        raise ValueError("timeout must be -1 or 0..86400000 milliseconds")
    return value


def _copy_ctypes(view):
    """Expose a contiguous byte view for a synchronous SDK input copy.

    ``memoryview.tobytes()`` followed by ``create_string_buffer`` allocated
    two host temporaries for every submitted plane.  The SDK still receives
    an owned buffer with the same lifetime, but ``from_buffer_copy`` performs
    the transfer in one allocation. The opt-in writable borrow avoids that
    temporary; C++ still owns the queued frame before submit returns.
    """
    buffer_type = c.c_char * view.nbytes
    if os.environ.get("RW_NR_BORROW_PYTHON_INPUT") == "1" and not view.readonly:
        # C++ copies the frame into its Job before submit returns. This owner
        # keeps the exporter alive throughout that synchronous FFI call.
        return buffer_type.from_buffer(view)
    return buffer_type.from_buffer_copy(view)


def _guarded(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class _NativeLease:
    """An independently retained C handle, allocated before C adopts it."""
    def __init__(self, library, release):
        self.library, self.release = library, release
        self.handle = c.c_void_p()

    def close(self):
        if getattr(self, "handle", None) and self.handle.value:
            handle, self.handle = self.handle, c.c_void_p()
            self.release(handle)

    def __del__(self):
        self.close()


class _Library:
    def __init__(self, path=None):
        self.path = _sdk_library_path("nr", path)
        self.dll = c.CDLL(str(self.path))  # ctypes releases the GIL for C calls/waits.
        self.temporal_available = True
        self.motion_available = True
        self.history_available = True
        self.history_validity_available = True
        self.cuda_available = True
        lib = self.dll
        signatures = {
            "rw_nr_runtime_create": ([c.POINTER(_RuntimeDesc), c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_runtime_query": ([c.c_void_p, c.c_void_p, c.POINTER(c.c_size_t)], c.c_int32),
            "rw_nr_runtime_close": ([c.c_void_p, c.c_int64], c.c_int32),
            "rw_nr_runtime_release": ([c.c_void_p], c.c_int32),
            "rw_nr_session_create": ([c.c_void_p, c.POINTER(_SessionDesc), c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_session_snapshot": ([c.c_void_p, c.POINTER(_SessionState)], c.c_int32),
            "rw_nr_session_reset": ([c.c_void_p, c.c_uint64, c.c_uint64], c.c_int32),
            "rw_nr_session_release": ([c.c_void_p], None),
            "rw_nr_submit": ([c.c_void_p, c.POINTER(_Frame), c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_submit_cuda_history": ([c.c_void_p, c.POINTER(_Frame), c.POINTER(_CudaInput), c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_submit_temporal_rgb": ([c.c_void_p, c.POINTER(_Frame), c.POINTER(c.c_float), c.c_uint64, c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_submit_temporal_rgb_motion": ([c.c_void_p, c.POINTER(_Frame), c.POINTER(c.c_float), c.c_uint64,
                                                     c.POINTER(c.c_float), c.c_uint64, c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_submit_temporal_history_motion": ([c.c_void_p, c.POINTER(_Frame),
                                                        c.POINTER(c.c_float), c.c_uint64,
                                                        c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_submit_temporal_history_motion_validity": ([c.c_void_p, c.POINTER(_Frame),
                                                                 c.POINTER(c.c_float), c.c_uint64,
                                                                 c.POINTER(c.c_float), c.c_uint64,
                                                                 c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_job_wait": ([c.c_void_p, c.c_int64], c.c_int32),
            "rw_nr_job_cancel": ([c.c_void_p], c.c_int32),
            "rw_nr_job_query": ([c.c_void_p, c.POINTER(_JobInfo)], c.c_int32),
            "rw_nr_job_get_result": ([c.c_void_p, c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_job_retain": ([c.c_void_p, c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_job_release": ([c.c_void_p], None),
            "rw_nr_result_view_get": ([c.c_void_p, c.POINTER(_ResultView)], c.c_int32),
            "rw_nr_result_retain": ([c.c_void_p, c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_result_release": ([c.c_void_p], None),
            "rw_nr_last_error": ([], c.c_char_p),
        }
        for name, (arguments, result) in signatures.items():
            try:
                function = getattr(lib, name)
            except AttributeError:
                if name == "rw_nr_submit_cuda_history":
                    self.cuda_available = False
                    continue
                if name == "rw_nr_submit_temporal_rgb":
                    self.temporal_available = False
                    continue
                if name == "rw_nr_submit_temporal_rgb_motion":
                    self.motion_available = False
                    continue
                if name == "rw_nr_submit_temporal_history_motion":
                    self.history_available = False
                    continue
                if name == "rw_nr_submit_temporal_history_motion_validity":
                    self.history_validity_available = False
                    continue
                raise
            function.argtypes, function.restype = arguments, result

    def check(self, status):
        if status:
            message = self.dll.rw_nr_last_error()
            raise CoreError(int(status), message.decode("utf-8", "replace") if message else f"NR status {status}")


class NativeRuntime:
    """Experimental H200 runtime; one slot by default, two/eight by explicit opt-in.

    All three file SHA256 values must come from the frozen registration/plan
    receipts. Python does not reinterpret missing hashes or model identities.
    metrics_node_detail opts into node observations without changing execution
    geometry or the mandatory request and physical-completion records.
    """
    def __init__(self, *, weights, weights_sha256, parameter_manifest,
                 parameter_manifest_sha256, plan, plan_sha256, experimental=False,
                 library=None, device_ordinal=0, device_budget_bytes=16 << 30,
                 host_budget_bytes=2 << 30, max_sessions=32, max_inflight=8, trace_capacity=128,
                 execution_slots=1, metrics_node_detail=False):
        self._lock = threading.RLock()
        self._handle = c.c_void_p()
        if experimental is not True:
            raise ValueError("experimental=True is required; this is not a qualified temporal/production backend")
        if type(device_ordinal) is not int or not 0 <= device_ordinal < (1 << 31):
            raise ValueError("invalid CUDA device ordinal")
        if type(execution_slots) is not int or execution_slots not in (1,2,8):
            raise ValueError("experimental execution_slots must be 1, 2 or 8")
        if type(metrics_node_detail) is not bool:
            raise ValueError("metrics_node_detail must be a bool")
        max_sessions = _bounded_u32("max_sessions", max_sessions, 1, 1024)
        max_inflight = _bounded_u32("max_inflight", max_inflight, 1, 64)
        trace_capacity = _bounded_u32("trace_capacity", trace_capacity, 1, 4096)
        if max_inflight < execution_slots:
            raise ValueError("max_inflight must be at least execution_slots")
        device_budget_bytes = _u64(device_budget_bytes)
        host_budget_bytes = _u64(host_budget_bytes)
        if not device_budget_bytes:
            raise ValueError("device_budget_bytes must be positive")
        if not 0 < host_budget_bytes <= 137438953472:
            raise ValueError("host_budget_bytes must be in [1, 137438953472]")
        self._library = _Library(library)
        encoded = [os.fsencode(value) for value in (weights, weights_sha256, parameter_manifest,
                                                   parameter_manifest_sha256, plan, plan_sha256)]
        flags = 1 | (2 if execution_slots==2 else 8 if execution_slots==8 else 0) | (4 if metrics_node_detail else 0)
        desc = _RuntimeDesc(c.sizeof(_RuntimeDesc), ABI, flags, device_ordinal,
                            max_sessions, max_inflight, trace_capacity, 0,
                            device_budget_bytes, host_budget_bytes, *encoded)
        self._library.check(self._library.dll.rw_nr_runtime_create(c.byref(desc), c.byref(self._handle)))

    @_guarded
    def _open(self):
        if not self._handle.value:
            raise RuntimeError("NativeRuntime is closed")

    @_guarded
    def query(self):
        self._open()
        count = c.c_size_t()
        self._library.check(self._library.dll.rw_nr_runtime_query(self._handle, None, c.byref(count)))
        # A worker can complete between the size and data calls. Retry only a
        # buffer-size race; no execution, state or precision is retried here.
        for _ in range(8):
            output = c.create_string_buffer(count.value)
            status = self._library.dll.rw_nr_runtime_query(self._handle, output, c.byref(count))
            if status != 12:
                self._library.check(status)
                return json.loads(output.value)
        raise RuntimeError("runtime trace changed repeatedly during bounded size query")

    @_guarded
    def session(self, *, epoch=1, predecessor=NO_PREDECESSOR, temporal=False):
        self._open()
        if temporal and not self._library.temporal_available:
            raise CoreError(3, "temporal RGB ABI2 is unavailable in this library")
        runtime_info = self.query()
        desc = _SessionDesc(c.sizeof(_SessionDesc), ABI, 1, 0, _u64(epoch), _u64(predecessor))
        result = NativeSession(self._library, c.c_void_p(), runtime_info, temporal=temporal)
        self._library.check(self._library.dll.rw_nr_session_create(self._handle, c.byref(desc), c.byref(result._handle)))
        if temporal:
            from .temporal_packet import NativeHistoryLedger
            state = result.snapshot()
            result._history_ledger = NativeHistoryLedger(state["session_id"], epoch=state["epoch"])
        return result

    @_guarded
    def prewarm(self, color_hwc, *, frame_id=0, epoch=1):
        """Prepare this runtime once with a physically completed stateless frame.

        Keep the runtime alive and create the production session after this call;
        immutable weights, CUDA modules and cuBLAS preparation then stay resident.
        The warmup result is discarded after completion and is never a quality or
        vendor-equivalence claim.
        """
        self._open()
        session = self.session(epoch=epoch, temporal=False)
        job = result = None
        try:
            job = session.submit(color_hwc, frame_id=frame_id, epoch=epoch)
            result = job.wait(timeout_ms=-1)
            info = job.info()
            if info["status"] != 0 or not info["physical_completion_confirmed"]:
                raise RuntimeError("prewarm frame did not physically complete")
            return {"frame_id": frame_id,
                    "epoch": epoch,
                    "device_stream_span_ms": info["device_stream_span_ms"],
                    "physical_completion_confirmed": True}
        finally:
            if result is not None:
                result.close()
            if job is not None:
                job.close()
            session.close()

    @_guarded
    def close(self, timeout_ms=-1):
        if not self._handle.value:
            return
        status = self._library.dll.rw_nr_runtime_close(self._handle, _timeout(timeout_ms))
        if status == 7:
            raise TimeoutError("native NR close is still draining; resources remain owned")
        # release consumes this user handle even if a failed GPU completion has
        # quarantined its internal resource owners until process teardown.
        handle, self._handle = self._handle, c.c_void_p()
        release_status = self._library.dll.rw_nr_runtime_release(handle)
        self._library.check(status or release_status)

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_lock", None):
            try:
                self.close()
            except BaseException:
                pass


class NativeSession:
    def __init__(self, library, handle, runtime_info, *, temporal=False):
        self._lock = threading.RLock()
        self._library, self._handle = library, handle
        self.width, self.height = runtime_info["width"], runtime_info["height"]
        self.temporal = bool(temporal)
        self._history_ledger = None

    @_guarded
    def snapshot(self):
        if not self._handle.value:
            raise RuntimeError("NativeSession is closed")
        result = _SessionState()
        result.struct_size, result.abi_version = c.sizeof(result), ABI
        self._library.check(self._library.dll.rw_nr_session_snapshot(self._handle, c.byref(result)))
        return {name: getattr(result, name) for name, _ in result._fields_[2:]}

    @_guarded
    def reset(self, epoch, predecessor=NO_PREDECESSOR):
        self.snapshot()
        self._library.check(self._library.dll.rw_nr_session_reset(self._handle, _u64(epoch), _u64(predecessor)))
        if self._history_ledger is not None:
            self._history_ledger.reset(epoch)

    @_guarded
    def submit_cuda(self, color_hwc, *, frame_id, epoch=None, expected_predecessor=None,
                    motion_hwc2=None, motion_validity=None, ready_event=None):
        """Borrow GPU producer planes and keep their owners through completion.

        ``ready_event`` is a recorded CUDA event (integer, .ptr or .cuda_event).
        With None, the producer must have completed all writes. Do not mutate
        borrowed planes until this job physically completes. Output is the
        existing CPU-readable NativeResult; previous RGB stays in the SDK.
        """
        state = self.snapshot()
        if not self.temporal or not self._library.cuda_available:
            raise ValueError("CUDA input needs a temporal session and the CUDA input SDK")
        color_pointer, colors = _cuda_plane(color_hwc, self.height, self.width, 3)
        motion_pointer, motions = _cuda_plane(motion_hwc2, self.height, self.width, 2)
        validity_pointer, validities = _cuda_plane(motion_validity, self.height, self.width, 1)
        event = 0 if ready_event is None else int(getattr(ready_event, "ptr", getattr(ready_event, "cuda_event", ready_event)))
        frame = _Frame(c.sizeof(_Frame), ABI, 16, 0, _u64(frame_id),
                       _u64(state["epoch"] if epoch is None else epoch),
                       _u64(state["predecessor"] if expected_predecessor is None else expected_predecessor),
                       self.width, self.height, color_pointer, colors)
        result = NativeJob(self._library, c.c_void_p())
        with _cuda_owners_lock:
            cookie = next(_cuda_owner_ids)
            _cuda_owners[cookie] = (color_hwc, motion_hwc2, motion_validity, ready_event)
        producer = _CudaInput(c.sizeof(_CudaInput), ABI, motion_pointer, motions,
                              validity_pointer, validities, event, _release_cuda_owners, cookie)
        try:
            self._library.check(self._library.dll.rw_nr_submit_cuda_history(
                self._handle, c.byref(frame), c.byref(producer), c.byref(result._handle)))
        except BaseException:
            _release_cuda_owners(cookie)
            result.close()
            raise
        return result

    @_guarded
    def submit(self, color_hwc, *, frame_id, epoch=None, expected_predecessor=None,
               previous_color_hwc=None, motion_hwc2=None, native_history=False):
        state = self.snapshot()
        if native_history:
            if not self.temporal or previous_color_hwc is not None:
                raise ValueError("native history needs a temporal session and SDK-owned previous colour")
            return self._submit_history(color_hwc, frame_id=frame_id,
                epoch=state["epoch"] if epoch is None else epoch,
                expected_predecessor=state["predecessor"] if expected_predecessor is None else expected_predecessor,
                motion_hwc2=motion_hwc2)
        view = memoryview(color_hwc)
        if not view.c_contiguous or view.itemsize != 4 or view.format not in ("f", "@f", "=f", "<f"):
            raise ValueError("input must be a contiguous native/little-endian float32 CPU buffer")
        if sys.byteorder != "little":
            raise ValueError("initial F32 image profile requires a little-endian host")
        if view.ndim not in (1, 3) or (view.ndim == 3 and tuple(view.shape) != (self.height, self.width, 3)):
            raise ValueError("input must have exact HWC shape or a flat H*W*3 float buffer")
        if view.nbytes != self.width * self.height * 3 * 4:
            raise ValueError("input byte length differs from the fixed semantic plan")
        raw = view.cast("B")
        # This copies storage only; normalization/padding/layout operators are C++.
        temporary = _copy_ctypes(raw)
        flags = 16 if self.temporal else 1
        frame = _Frame(c.sizeof(_Frame), ABI, flags, 0, _u64(frame_id),
                       _u64(state["epoch"] if epoch is None else epoch),
                       _u64(state["predecessor"] if expected_predecessor is None else expected_predecessor),
                       self.width, self.height, c.cast(temporary, c.POINTER(c.c_float)), raw.nbytes // 4)
        result = NativeJob(self._library, c.c_void_p())
        if self.temporal:
            if previous_color_hwc is None:
                previous_buffer = None
                previous_pointer = c.POINTER(c.c_float)()
                previous_elements = 0
            else:
                previous_view = memoryview(previous_color_hwc)
                if (not previous_view.c_contiguous or previous_view.itemsize != 4 or
                        previous_view.format not in ("f", "@f", "=f", "<f") or
                        previous_view.ndim not in (1, 3) or
                        (previous_view.ndim == 3 and tuple(previous_view.shape) != (self.height, self.width, 3)) or
                        previous_view.nbytes != self.width * self.height * 3 * 4):
                    raise ValueError("previous input must have exact contiguous HWC float32 shape")
                previous_raw = previous_view.cast("B")
                previous_buffer = _copy_ctypes(previous_raw)
                previous_pointer = c.cast(previous_buffer, c.POINTER(c.c_float))
                previous_elements = previous_raw.nbytes // 4
            if motion_hwc2 is not None:
                if not self._library.motion_available:
                    raise CoreError(3, "motion-compensated temporal ABI2 is unavailable in this library")
                motion_view = memoryview(motion_hwc2)
                if (not motion_view.c_contiguous or motion_view.itemsize != 4 or
                        motion_view.format not in ("f", "@f", "=f", "<f") or
                        motion_view.ndim not in (1, 3) or
                        (motion_view.ndim == 3 and tuple(motion_view.shape) != (self.height, self.width, 2)) or
                        motion_view.nbytes != self.width * self.height * 2 * 4):
                    raise ValueError("motion input must have exact contiguous HWC2 float32 shape")
                motion_raw = motion_view.cast("B")
                motion_buffer = _copy_ctypes(motion_raw)
                motion_pointer = c.cast(motion_buffer, c.POINTER(c.c_float))
                motion_elements = motion_raw.nbytes // 4
                self._library.check(self._library.dll.rw_nr_submit_temporal_rgb_motion(
                    self._handle, c.byref(frame), previous_pointer, previous_elements,
                    motion_pointer, motion_elements, c.byref(result._handle)))
            else:
                self._library.check(self._library.dll.rw_nr_submit_temporal_rgb(
                    self._handle, c.byref(frame), previous_pointer, previous_elements, c.byref(result._handle)))
        else:
            if previous_color_hwc is not None or motion_hwc2 is not None:
                raise ValueError("temporal inputs require NativeRuntime.session(temporal=True)")
            self._library.check(self._library.dll.rw_nr_submit(self._handle, c.byref(frame), c.byref(result._handle)))
        return result

    @_guarded
    def submit_packet(self, packet, color_hwc, *, previous_color_hwc=None,
                      motion_hwc2=None, motion_validity=None,
                      native_history=False):
        """Submit an admitted :class:`TemporalPacket` through the ABI2 lane.

        Packet metadata remains the source of timeline and guide lineage. The
        native ABI currently consumes RGB plus pixel motion only; depth and
        history ownership are validated by the packet and are not silently
        treated as native graph inputs.
        """
        from .temporal_packet import TemporalPacket
        if not isinstance(packet, TemporalPacket):
            raise TypeError("packet must be TemporalPacket")
        packet.validate(width=self.width, height=self.height)
        if not self.temporal:
            raise ValueError("TemporalPacket requires NativeRuntime.session(temporal=True)")
        state = self.snapshot()
        if packet.session_id != state["session_id"]:
            raise ValueError("packet session differs from native session")
        if self._history_ledger is None:
            raise RuntimeError("temporal history ledger is unavailable")
        self._history_ledger.admit(packet, width=self.width, height=self.height)
        if type(native_history) is not bool:
            raise TypeError("native_history must be bool")
        if native_history:
            if not self._library.history_available:
                raise CoreError(3, "session-owned native history ABI is unavailable in this library")
            if previous_color_hwc is not None:
                raise ValueError("native history path owns previous colour; caller payload is forbidden")
            if not packet.reset and motion_hwc2 is None:
                raise ValueError("native history packet needs motion payload")
            if packet.reset and motion_hwc2 is not None:
                raise ValueError("reset packet cannot carry motion")
            if packet.reset and motion_validity is not None:
                raise ValueError("reset packet cannot carry motion validity")
            return self._submit_history(color_hwc, frame_id=packet.frame_id,
                                        epoch=packet.epoch,
                                        expected_predecessor=NO_PREDECESSOR if packet.reset
                                        else packet.expected_predecessor,
                                        motion_hwc2=motion_hwc2,
                                        motion_validity=motion_validity)
        if packet.reset:
            if previous_color_hwc is not None or motion_hwc2 is not None:
                raise ValueError("reset packet cannot submit temporal payloads")
            predecessor = NO_PREDECESSOR
        else:
            if packet.motion.units != "pixels":
                raise ValueError("native motion ABI requires pixel units; convert uv first")
            if previous_color_hwc is None:
                previous_color_hwc = packet.previous_color
            if previous_color_hwc is None or motion_hwc2 is None:
                raise ValueError("non-reset packet needs previous colour and motion payloads")
            predecessor = packet.expected_predecessor
        return self.submit(color_hwc, frame_id=packet.frame_id, epoch=packet.epoch,
                           expected_predecessor=predecessor,
                           previous_color_hwc=previous_color_hwc,
                           motion_hwc2=motion_hwc2)

    def _submit_history(self, color_hwc, *, frame_id, epoch,
                        expected_predecessor, motion_hwc2=None,
                        motion_validity=None):
        if not self._library.history_available:
            raise CoreError(3, "session-owned native history ABI is unavailable in this library")
        view = memoryview(color_hwc)
        if (not view.c_contiguous or view.itemsize != 4 or
                view.format not in ("f", "@f", "=f", "<f") or
                view.ndim not in (1, 3) or
                (view.ndim == 3 and tuple(view.shape) != (self.height, self.width, 3)) or
                view.nbytes != self.width * self.height * 3 * 4):
            raise ValueError("input must have exact contiguous HWC float32 shape")
        temporary = _copy_ctypes(view.cast("B"))
        frame = _Frame(c.sizeof(_Frame), ABI, 16, 0, _u64(frame_id), _u64(epoch),
                       _u64(expected_predecessor), self.width, self.height,
                       c.cast(temporary, c.POINTER(c.c_float)), view.nbytes // 4)
        motion_buffer = None
        motion_pointer = c.POINTER(c.c_float)()
        motion_elements = 0
        if motion_hwc2 is not None:
            motion_view = memoryview(motion_hwc2)
            if (not motion_view.c_contiguous or motion_view.itemsize != 4 or
                    motion_view.format not in ("f", "@f", "=f", "<f") or
                    motion_view.ndim not in (1, 3) or
                    (motion_view.ndim == 3 and tuple(motion_view.shape) != (self.height, self.width, 2)) or
                    motion_view.nbytes != self.width * self.height * 2 * 4):
                raise ValueError("motion input must have exact contiguous HWC2 float32 shape")
            motion_buffer = _copy_ctypes(motion_view.cast("B"))
            motion_pointer = c.cast(motion_buffer, c.POINTER(c.c_float))
            motion_elements = motion_view.nbytes // 4
        validity_buffer = None
        validity_pointer = c.POINTER(c.c_float)()
        validity_elements = 0
        if motion_validity is not None:
            if not self._library.history_validity_available:
                raise CoreError(3, "native history validity ABI is unavailable in this library")
            validity_view = memoryview(motion_validity)
            if (not validity_view.c_contiguous or validity_view.itemsize != 4 or
                    validity_view.format not in ("f", "@f", "=f", "<f") or
                    validity_view.ndim not in (1, 2) or
                    (validity_view.ndim == 2 and tuple(validity_view.shape) != (self.height, self.width)) or
                    validity_view.nbytes != self.width * self.height * 4):
                raise ValueError("motion validity must have exact contiguous HW float32 shape")
            validity_buffer = _copy_ctypes(validity_view.cast("B"))
            validity_pointer = c.cast(validity_buffer, c.POINTER(c.c_float))
            validity_elements = validity_view.nbytes // 4
        result = NativeJob(self._library, c.c_void_p())
        if motion_validity is None:
            self._library.check(self._library.dll.rw_nr_submit_temporal_history_motion(
                self._handle, c.byref(frame), motion_pointer, motion_elements,
                c.byref(result._handle)))
        else:
            self._library.check(self._library.dll.rw_nr_submit_temporal_history_motion_validity(
                self._handle, c.byref(frame), motion_pointer, motion_elements,
                validity_pointer, validity_elements, c.byref(result._handle)))
        return result

    @_guarded
    def submit_history(self, color_hwc, *, frame_id, epoch=None,
                       expected_predecessor=None, motion_hwc2=None,
                       motion_validity=None):
        """Submit using the SDK-retained output history and optional motion."""
        if not self.temporal:
            raise ValueError("session-owned history requires temporal=True")
        state = self.snapshot()
        return self._submit_history(
            color_hwc, frame_id=frame_id,
            epoch=state["epoch"] if epoch is None else epoch,
            expected_predecessor=state["predecessor"] if expected_predecessor is None
            else expected_predecessor,
            motion_hwc2=motion_hwc2, motion_validity=motion_validity)

    @_guarded
    def commit_packet(self, packet, job, result, *, next_history_owner_id):
        """Commit packet history after a physically completed valid output."""
        from .temporal_packet import TemporalPacket
        if not isinstance(packet, TemporalPacket):
            raise TypeError("packet must be TemporalPacket")
        if self._history_ledger is None:
            raise RuntimeError("temporal history ledger is unavailable")
        if not isinstance(job, NativeJob) or not isinstance(result, NativeResult):
            raise TypeError("job and result must be native handles")
        info = job.info()
        metadata = result.metadata()
        if (info["status"] != 0 or not info["physical_completion_confirmed"] or
                metadata["session_id"] != packet.session_id or
                metadata["frame_id"] != packet.frame_id or metadata["epoch"] != packet.epoch):
            raise RuntimeError("native output is not a physically completed packet result")
        return self._history_ledger.commit(
            packet, width=self.width, height=self.height,
            physical_completion=True, output_valid=True,
            next_history_owner_id=next_history_owner_id)

    @_guarded
    def close(self):
        if getattr(self, "_handle", None) and self._handle.value:
            handle, self._handle = self._handle, c.c_void_p()
            self._library.dll.rw_nr_session_release(handle)

    def __enter__(self):
        self.snapshot()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_lock", None):
            self.close()


class NativeJob:
    def __init__(self, library, handle):
        self._lock = threading.RLock()
        self._library, self._handle = library, handle

    @_guarded
    def info(self):
        if not self._handle.value:
            raise RuntimeError("NativeJob user reference is released")
        value = _JobInfo()
        value.struct_size, value.abi_version = c.sizeof(value), ABI
        self._library.check(self._library.dll.rw_nr_job_query(self._handle, c.byref(value)))
        result = {name: getattr(value, name) for name, _ in value._fields_[2:]}
        result["error"] = bytes(value.error).decode("utf-8", "replace")
        return result

    def wait(self, timeout_ms=-1):
        # Do not hold the wrapper lock while waiting: cancel/info/close remain
        # usable. An independent C handle protects this call against close().
        with self._lock:
            self.info()
            lease = _NativeLease(self._library, self._library.dll.rw_nr_job_release)
            self._library.check(self._library.dll.rw_nr_job_retain(self._handle, c.byref(lease.handle)))
        try:
            result = NativeResult(self._library, c.c_void_p())
            status = self._library.dll.rw_nr_job_wait(lease.handle, _timeout(timeout_ms))
            if status == 7:
                raise TimeoutError("wait elapsed; native NR device work remains owned and may still run")
            self._library.check(status)
            self._library.check(self._library.dll.rw_nr_job_get_result(lease.handle, c.byref(result._handle)))
            return result
        finally:
            lease.close()

    @_guarded
    def cancel(self):
        self.info()
        status = self._library.dll.rw_nr_job_cancel(self._handle)
        if status == 7:
            return False  # already terminal; published output is not withdrawn
        self._library.check(status)
        return True

    @_guarded
    def close(self):
        if getattr(self, "_handle", None) and self._handle.value:
            handle, self._handle = self._handle, c.c_void_p()
            self._library.dll.rw_nr_job_release(handle)

    def __del__(self):
        if getattr(self, "_lock", None):
            self.close()


class NativeResult:
    def __init__(self, library, handle):
        self._lock = threading.RLock()
        self._library, self._handle = library, handle

    @_guarded
    def _view(self):
        if not self._handle.value:
            raise RuntimeError("NativeResult is closed")
        value = _ResultView()
        value.struct_size, value.abi_version = c.sizeof(value), ABI
        self._library.check(self._library.dll.rw_nr_result_view_get(self._handle, c.byref(value)))
        return value

    @_guarded
    def metadata(self):
        value = self._view()
        return {"width": value.width, "height": value.height, "channels": 3, "dtype": "float32",
                "layout": "HWC", "color": "srgb_code_full_sdr", "session_id": value.session_id,
                "frame_id": value.frame_id, "epoch": value.epoch, "implementation_kind": "native_self",
                "support_status": "experimental_unqualified", "native_temporal_qualified": False,
                "native_vendor_equivalence_verified": False, "production_qualified": False}

    @_guarded
    def tensor_view(self):
        value = self._view()
        lease = _NativeLease(self._library, self._library.dll.rw_nr_result_release)
        self._library.check(self._library.dll.rw_nr_result_retain(self._handle, c.byref(lease.handle)))
        buffer = (c.c_float * value.color_elements).from_address(c.addressof(value.color_hwc.contents))
        buffer._rw_result_lease = lease
        # The memoryview owns buffer -> lease -> C++ Result, so explicit close()
        # on this NativeResult cannot invalidate a separately retained view.
        return memoryview(buffer).cast("B").cast("f", shape=(value.height, value.width, 3)).toreadonly()

    @_guarded
    def copy_f32(self):
        value = self._view()
        result = array("f")
        result.frombytes(c.string_at(value.color_hwc, value.color_elements * 4))
        return result

    @_guarded
    def close(self):
        if getattr(self, "_handle", None) and self._handle.value:
            handle, self._handle = self._handle, c.c_void_p()
            self._library.dll.rw_nr_result_release(handle)

    def __enter__(self):
        self._view()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_lock", None):
            self.close()
