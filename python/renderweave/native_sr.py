"""C ABI binding for the native K/LDR/2x SR profile."""
from __future__ import annotations

from array import array
import ctypes as c
import os
import sys

from ._core import ABI_VERSION, CoreError, _sdk_library_path, _unsigned64

_NOT_READY = 7
_FLOAT = c.POINTER(c.c_float)
_HANDLE = c.c_void_p
_HEADER = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32)]


class _Desc(c.Structure):
    _fields_ = _HEADER + [("device_ordinal", c.c_int32)] + [
        (name, c.c_uint32) for name in ("width", "height", "execution_slots", "max_sessions")
    ] + [("device_budget_bytes", c.c_uint64), ("weights_path", c.c_char_p), ("filter_path", c.c_char_p),
         ("options", c.c_uint32)]


class _Frame(c.Structure):
    _fields_ = _HEADER + [
        (name, c.c_uint64) for name in ("frame_id", "epoch", "expected_predecessor")
    ] + [(name, kind) for plane in ("color", "motion", "depth")
         for name, kind in ((plane, _FLOAT), (plane + "_elements", c.c_uint64))]


class _Info(c.Structure):
    _fields_ = _HEADER + [("status", c.c_int32), ("physical_complete", c.c_uint32)] + [
        (name, c.c_uint64) for name in
        ("frame_id", "epoch", "submit_ns", "device_enqueued_ns", "observed_complete_ns")
    ] + [("device_stream_ms", c.c_double)]


class _CudaInput(c.Structure):
    _fields_ = _HEADER + [("ready_event", c.c_void_p)]


class _CudaView(c.Structure):
    _fields_ = _HEADER + [("rgba", c.c_void_p), ("rgba_elements", c.c_uint64)]


class _MemoryInfo(c.Structure):
    _fields_ = _HEADER + [(name, c.c_uint64) for name in
        ("shared_model_device_bytes", "slots_device_bytes", "live_sessions_device_bytes",
         "total_device_bytes", "pinned_host_bytes")]


class _StageTimings(c.Structure):
    _fields_ = _HEADER + [(name, c.c_double) for name in
        ("upload_ms", "prepare_ms", "backbone_ms", "reconstruct_ms", "validate_and_readback_ms")]


class _DisplayInfo(c.Structure):
    _fields_ = _HEADER + [("min_rgba", c.c_float), ("max_rgba", c.c_float), ("clamped_rgb_values", c.c_uint64)]


def _library(path):
    lib = c.CDLL(str(_sdk_library_path("sr", path)))
    functions = [
        ("runtime_create", [c.POINTER(_Desc), c.POINTER(_HANDLE)], c.c_int32),
        ("last_error", [], c.c_char_p),
        ("runtime_release", [_HANDLE], None),
        ("session_create", [_HANDLE, c.c_uint64, c.POINTER(_HANDLE)], c.c_int32),
        ("session_reset", [_HANDLE, c.c_uint64], c.c_int32),
        ("session_release", [_HANDLE], None),
        ("submit", [_HANDLE, c.POINTER(_Frame), c.POINTER(_HANDLE)], c.c_int32),
        ("job_wait", [_HANDLE, c.c_int64, _FLOAT, c.c_uint64], c.c_int32),
        ("job_get_info", [_HANDLE, c.POINTER(_Info)], c.c_int32),
        ("job_cancel", [_HANDLE], c.c_int32),
        ("job_release", [_HANDLE], None),
    ]
    for name, args, result in functions:
        function = getattr(lib, "rw_sr_" + name)
        function.argtypes, function.restype = args, result
    create_memory = getattr(lib, "rw_sr_runtime_create_from_memory", None)
    if create_memory is not None:
        create_memory.argtypes = [c.POINTER(_Desc), c.c_char_p, c.c_uint64, c.c_char_p, c.c_uint64,
                                  c.POINTER(_HANDLE)]
        create_memory.restype = c.c_int32
    memory_info = getattr(lib, "rw_sr_runtime_get_memory_info", None)
    if memory_info is not None:
        memory_info.argtypes = [_HANDLE, c.POINTER(_MemoryInfo)]
        memory_info.restype = c.c_int32
    stage_timings = getattr(lib, "rw_sr_job_get_stage_timings", None)
    if stage_timings is not None:
        stage_timings.argtypes = [_HANDLE, c.POINTER(_StageTimings)]
        stage_timings.restype = c.c_int32
    if hasattr(lib,"rw_sr_submit_cuda"):
        lib.rw_sr_submit_cuda.argtypes = [_HANDLE,c.POINTER(_Frame),c.POINTER(_CudaInput),c.POINTER(_HANDLE)]
        lib.rw_sr_submit_cuda.restype = c.c_int32
        lib.rw_sr_job_wait_cuda.argtypes = [_HANDLE,c.c_int64,c.POINTER(_CudaView)]
        lib.rw_sr_job_wait_cuda.restype = c.c_int32
    display = getattr(lib, "rw_sr_display_rgb8", None)
    if display is not None:
        display.argtypes = [_FLOAT, c.c_uint64, c.POINTER(c.c_uint8), c.c_uint64, c.POINTER(_DisplayInfo)]
        display.restype = c.c_int32
    return lib


def _plane(data):
    if data is None:
        return None, 0
    view = memoryview(data)
    formats = ("f", "@f", "=f", "<f") if sys.byteorder == "little" else ("f", "@f", "=f", ">f")
    if view.format not in formats or view.itemsize != 4 or not view.c_contiguous:
        raise ValueError("SR input must be contiguous native-endian float32")
    size = view.nbytes // 4
    plane_type = c.c_float * size
    buffer = view.cast("B")
    owner = plane_type.from_buffer_copy(buffer) if view.readonly else plane_type.from_buffer(buffer)
    return owner, size


def _bounded(value, bits):
    if _unsigned64(value) >= 1 << bits:
        raise ValueError(f"SR argument exceeds the {bits}-bit range")
    return value


class _Owner:
    def __init__(self, library, handle, release, output_elements):
        self._library = library
        self._handle = handle
        self._release = release
        self._output_elements = output_elements

    def _check(self, status):
        if status:
            error = self._library.rw_sr_last_error()
            raise CoreError(status, error.decode() if error else f"RenderWeave SR status {status}")

    def close(self):
        if getattr(self, "_handle", None):
            self._release(self._handle)
            self._handle = _HANDLE()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class NativeSRRuntime(_Owner):
    def __init__(self, width, height, weights_path, filter_path, *, device_budget_bytes,
                 device=0, execution_slots=1, max_sessions=1, library=None, model_bytes=None,
                 cuda_graph=False, tiled_backbone=False, profile_stages=False, parallel_reduce=False,
                 reuse_matrix_rows=False, spin_wait=False, cuda_io_only=False, original_fp16_backbone=False):
        lib = _library(library)
        super().__init__(lib, _HANDLE(), lib.rw_sr_runtime_release, width * height * 16)
        desc = _Desc(c.sizeof(_Desc), ABI_VERSION, _bounded(device, 31), _bounded(width, 32),
                     _bounded(height, 32), _bounded(execution_slots, 32), _bounded(max_sessions, 32),
                     _unsigned64(device_budget_bytes))
        desc.options = (int(bool(cuda_graph)) | (2 * int(bool(tiled_backbone))) |
                        (4 * int(bool(profile_stages))) | (8 * int(bool(parallel_reduce))) |
                        (16 * int(bool(reuse_matrix_rows))) | (32 * int(bool(spin_wait))) |
                        (64 * int(bool(cuda_io_only))) | (128 * int(bool(original_fp16_backbone))))
        if model_bytes is None:
            desc.weights_path, desc.filter_path = os.fsencode(weights_path), os.fsencode(filter_path)
            status = lib.rw_sr_runtime_create(c.byref(desc), c.byref(self._handle))
        else:
            weights, filter_bytes = model_bytes
            status = lib.rw_sr_runtime_create_from_memory(c.byref(desc), weights, len(weights),
                                                          filter_bytes, len(filter_bytes), c.byref(self._handle))
        self._check(status)

    def memory_info(self):
        info = _MemoryInfo(c.sizeof(_MemoryInfo), ABI_VERSION)
        self._check(self._library.rw_sr_runtime_get_memory_info(self._handle,c.byref(info)))
        return {name: getattr(info,name) for name,_ in _MemoryInfo._fields_[2:]}

    @property
    def has_native_display_rgb8(self):
        return hasattr(self._library, "rw_sr_display_rgb8")

    def display_rgb8(self, rgba, output):
        """Fill a writable packed RGB8 buffer and return the raw RGBA summary."""
        source, elements = _plane(rgba)
        rgb = memoryview(output)
        if rgb.format != "B" or not rgb.c_contiguous or rgb.readonly:
            raise ValueError("RGB8 output must be writable contiguous uint8")
        destination = (c.c_uint8 * rgb.nbytes).from_buffer(rgb.cast("B"))
        info = _DisplayInfo(c.sizeof(_DisplayInfo), ABI_VERSION)
        status = self._library.rw_sr_display_rgb8(source, elements, destination, rgb.nbytes, c.byref(info))
        if status:
            raise CoreError(status, f"SR display conversion status {status}")
        return info.min_rgba, info.max_rgba, info.clamped_rgb_values

    def session(self, epoch=1):
        handle = _HANDLE()
        lib = self._library
        self._check(lib.rw_sr_session_create(self._handle, _unsigned64(epoch), c.byref(handle)))
        return NativeSRSession(lib, handle, lib.rw_sr_session_release, self._output_elements)


class NativeSRSession(_Owner):
    def reset(self, epoch):
        self._check(self._library.rw_sr_session_reset(self._handle, _unsigned64(epoch)))

    def submit(self, color, *, frame_id, epoch, predecessor, motion=None, depth=None):
        frame = _Frame(c.sizeof(_Frame), ABI_VERSION, _unsigned64(frame_id), _unsigned64(epoch),
                       _unsigned64(predecessor), *_plane(color), *_plane(motion), *_plane(depth))
        handle = _HANDLE()
        lib = self._library
        self._check(lib.rw_sr_submit(self._handle, c.byref(frame), c.byref(handle)))
        return NativeSRJob(lib, handle, lib.rw_sr_job_release, self._output_elements)

    def submit_cuda(self, color, *, frame_id, epoch, predecessor, motion=0, depth=0, ready_event=0):
        """Submit contiguous device float32 addresses. Retain planes until physically complete."""
        pixels = self._output_elements//16
        frame = _Frame(c.sizeof(_Frame),ABI_VERSION,_unsigned64(frame_id),_unsigned64(epoch),
                       _unsigned64(predecessor),
                       c.cast(c.c_void_p(_unsigned64(color)),_FLOAT),pixels*3,
                       c.cast(c.c_void_p(_unsigned64(motion)),_FLOAT),pixels*2 if motion else 0,
                       c.cast(c.c_void_p(_unsigned64(depth)),_FLOAT),pixels if depth else 0)
        cuda = _CudaInput(c.sizeof(_CudaInput),ABI_VERSION,_unsigned64(ready_event))
        handle = _HANDLE()
        lib = self._library
        self._check(lib.rw_sr_submit_cuda(self._handle,c.byref(frame),c.byref(cuda),c.byref(handle)))
        return NativeSRJob(lib,handle,lib.rw_sr_job_release,self._output_elements)


class NativeSRJob(_Owner):
    def wait(self, timeout_ms=-1, *, output=None):
        """Return flat 2x RGBA float32, or None on timeout. A supplied buffer is reused."""
        if type(timeout_ms) is not int or not -1 <= timeout_ms < 1 << 63:
            raise ValueError("invalid SR wait timeout")
        if output is None:
            output = array("f", [0]) * self._output_elements
        elif memoryview(output).readonly:
            raise ValueError("SR output buffer must be writable")
        rgba, size = _plane(output)
        status = self._library.rw_sr_job_wait(self._handle, timeout_ms, rgba, size)
        if status == _NOT_READY:
            return None
        self._check(status)
        return output

    def info(self):
        info = _Info(struct_size=c.sizeof(_Info), abi_version=ABI_VERSION)
        self._check(self._library.rw_sr_job_get_info(self._handle, c.byref(info)))
        return {name: getattr(info, name) for name, _ in _Info._fields_[2:]}

    def wait_cuda(self, timeout_ms=-1):
        """Return (CUDA address, RGBA elements), or None. Hold the job through consumer completion."""
        if type(timeout_ms) is not int or not -1 <= timeout_ms < 1 << 63:
            raise ValueError("invalid SR wait timeout")
        view = _CudaView(c.sizeof(_CudaView),ABI_VERSION)
        status = self._library.rw_sr_job_wait_cuda(self._handle,timeout_ms,c.byref(view))
        if status == _NOT_READY:
            return None
        self._check(status)
        return view.rgba,view.rgba_elements

    def stage_timings(self):
        info = _StageTimings(c.sizeof(_StageTimings), ABI_VERSION)
        self._check(self._library.rw_sr_job_get_stage_timings(self._handle,c.byref(info)))
        return {name: getattr(info,name) for name,_ in _StageTimings._fields_[2:]}

    def cancel(self):
        self._check(self._library.rw_sr_job_cancel(self._handle))
