"""Thin ctypes binding for the ABI2 temporal contract.

The current library exposes CPU preflight and memory arithmetic only. This
module intentionally reports native temporal as unavailable and never creates
or fabricates history on behalf of a caller.
"""
from __future__ import annotations

import ctypes as c
import threading
from typing import Any

from ._core import CoreError, _resolve_library

ABI = 2
NO_PREDECESSOR = (1 << 64) - 1
NATIVE_EXECUTION_AVAILABLE = False


class TemporalError(CoreError):
    """ABI2 temporal contract failure."""


class _Tensor(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("dtype", c.c_uint32),
                ("device_kind", c.c_uint32), ("device_ordinal", c.c_int32),
                ("ndim", c.c_uint32), ("access_mode", c.c_uint32),
                ("shape", c.c_int64 * 8), ("strides_bytes", c.c_int64 * 8),
                ("base", c.c_void_p), ("byte_offset", c.c_uint64),
                ("allocation_bytes", c.c_uint64), ("owner_token", c.c_void_p),
                ("ready_event", c.c_void_p)]


class _Metadata(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("color_space", c.c_uint32),
                ("dynamic_range", c.c_uint32), ("depth_definition", c.c_uint32),
                ("motion_direction", c.c_uint32), ("motion_units", c.c_uint32),
                ("jitter_included", c.c_uint32), ("reserved", c.c_uint32)]


class _Resource(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("resource_semantic", c.c_uint32),
                ("validity", c.c_uint32), ("provenance", c.c_uint32),
                ("tensor", _Tensor), ("metadata_json", c.c_char_p),
                ("metadata", _Metadata)]


class TemporalResource(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32),
                ("resource", _Resource), ("owner_id", c.c_uint64),
                ("owner_session_id", c.c_uint64), ("owner_epoch", c.c_uint64),
                ("owner_frame_id", c.c_uint64)]


class TemporalFrame(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32),
                ("flags", c.c_uint32), ("reset", c.c_uint32),
                ("session_id", c.c_uint64), ("sequence_id", c.c_uint64),
                ("frame_id", c.c_uint64), ("epoch", c.c_uint64),
                ("expected_predecessor", c.c_uint64), ("pts", c.c_int64),
                ("time_base_num", c.c_int32), ("time_base_den", c.c_int32),
                ("scene_epoch", c.c_uint32), ("device_kind", c.c_uint32),
                ("device_ordinal", c.c_int32), ("required_semantics", c.c_uint64),
                ("resources", c.POINTER(TemporalResource)), ("resource_count", c.c_uint32),
                ("reserved", c.c_uint32), ("motion_direction", c.c_uint32),
                ("motion_units", c.c_uint32), ("jitter_included", c.c_uint32),
                ("jitter_explicitly_disabled", c.c_uint32),
                ("jitter_x_bits", c.c_uint32), ("jitter_y_bits", c.c_uint32),
                ("exposure_scale_bits", c.c_uint32),
                ("exposure_explicit_no_transform", c.c_uint32),
                ("sequence_identity", c.c_uint8 * 32),
                ("model_fingerprint", c.c_uint8 * 32),
                ("rng_algorithm", c.c_uint8 * 32),
                ("history_codec_sha256", c.c_uint8 * 32),
                ("sampler_sha256", c.c_uint8 * 32)]


class TemporalState(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32),
                ("valid", c.c_uint32), ("reserved", c.c_uint32),
                ("session_id", c.c_uint64), ("sequence_id", c.c_uint64),
                ("epoch", c.c_uint64), ("predecessor", c.c_uint64),
                ("pts", c.c_int64), ("time_base_num", c.c_int32),
                ("time_base_den", c.c_int32), ("scene_epoch", c.c_uint32),
                ("reserved2", c.c_uint32), ("history_owner_id", c.c_uint64),
                ("history_epoch", c.c_uint64), ("history_frame", c.c_uint64),
                ("sequence_identity", c.c_uint8 * 32),
                ("model_fingerprint", c.c_uint8 * 32),
                ("rng_algorithm", c.c_uint8 * 32),
                ("history_codec_sha256", c.c_uint8 * 32),
                ("sampler_sha256", c.c_uint8 * 32)]


class TemporalPreflightResult(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32),
                ("code", c.c_uint32), ("native_execution_available", c.c_uint32),
                ("required_semantics", c.c_uint64), ("present_semantics", c.c_uint64),
                ("history_commit_allowed", c.c_uint32), ("reserved", c.c_uint32)]


class TemporalMemoryPlan(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32),
                ("budget_bytes", c.c_uint64), ("shared_weights_bytes", c.c_uint64),
                ("bytes_per_slot", c.c_uint64), ("slot_count", c.c_uint64),
                ("committed_history_bytes", c.c_uint64),
                ("provisional_history_bytes", c.c_uint64),
                ("session_count", c.c_uint64), ("import_output_bytes", c.c_uint64),
                ("graph_handle_bytes", c.c_uint64), ("reserve_quarantine_bytes", c.c_uint64)]


def _u64(value: int) -> int:
    if type(value) is not int or not 0 <= value < (1 << 64):
        raise ValueError("expected unsigned 64-bit integer")
    return value


def _bool(name: str, value: bool) -> int:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return int(value)


class TemporalLibrary:
    """ABI2 CPU contract entry; model execution remains unavailable."""

    def __init__(self, library: str | None = None):
        self.path = _resolve_library(library)
        self.dll = c.CDLL(str(self.path))
        signatures = {
            "rw_nr_temporal_preflight": ([c.POINTER(TemporalFrame), c.POINTER(TemporalState),
                                           c.POINTER(TemporalPreflightResult)], c.c_int32),
            "rw_nr_temporal_commit": ([c.POINTER(TemporalState), c.POINTER(TemporalFrame),
                                        c.c_uint32, c.c_uint32, c.c_uint64,
                                        c.POINTER(TemporalState)], c.c_int32),
            "rw_nr_temporal_memory_required": ([c.POINTER(TemporalMemoryPlan),
                                                 c.POINTER(c.c_uint64)], c.c_int32),
            "rw_nr_temporal_session_create": ([c.c_uint64, c.c_uint64,
                                                 c.POINTER(c.c_void_p)], c.c_int32),
            "rw_nr_temporal_session_snapshot": ([c.c_void_p, c.POINTER(TemporalState)], c.c_int32),
            "rw_nr_temporal_session_reset": ([c.c_void_p, c.c_uint64], c.c_int32),
            "rw_nr_temporal_session_preflight": ([c.c_void_p, c.POINTER(TemporalFrame),
                                                    c.POINTER(TemporalPreflightResult)], c.c_int32),
            "rw_nr_temporal_session_commit": ([c.c_void_p, c.POINTER(TemporalFrame),
                                                 c.c_uint32, c.c_uint32, c.c_uint64,
                                                 c.POINTER(TemporalState)], c.c_int32),
            "rw_nr_temporal_session_release": ([c.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = args, result

    @staticmethod
    def _result(value: TemporalPreflightResult) -> dict[str, Any]:
        return {"code": int(value.code),
                "native_execution_available": bool(value.native_execution_available),
                "required_semantics": int(value.required_semantics),
                "present_semantics": int(value.present_semantics),
                "history_commit_allowed": bool(value.history_commit_allowed),
                "native_temporal_qualified": False}

    def preflight(self, frame: TemporalFrame, state: TemporalState | None = None) -> dict[str, Any]:
        if not isinstance(frame, TemporalFrame):
            raise TypeError("frame must be TemporalFrame")
        result = TemporalPreflightResult()
        result.struct_size, result.abi_version = c.sizeof(result), ABI
        state_ptr = c.byref(state) if state is not None else c.POINTER(TemporalState)()
        status = self.dll.rw_nr_temporal_preflight(c.byref(frame), state_ptr, c.byref(result))
        if status:
            raise TemporalError(int(status), f"ABI2 temporal preflight failed with status {status}")
        return self._result(result)

    def commit(self, before: TemporalState, frame: TemporalFrame, *,
               physical_completion: bool, output_valid: bool,
               next_history_owner_id: int) -> TemporalState:
        if not isinstance(before, TemporalState) or not isinstance(frame, TemporalFrame):
            raise TypeError("before must be TemporalState and frame must be TemporalFrame")
        after = TemporalState()
        after.struct_size, after.abi_version = c.sizeof(after), ABI
        status = self.dll.rw_nr_temporal_commit(c.byref(before), c.byref(frame),
                                                 _bool("physical_completion", physical_completion),
                                                 _bool("output_valid", output_valid),
                                                 _u64(next_history_owner_id), c.byref(after))
        if status:
            raise TemporalError(int(status), f"ABI2 temporal commit failed with status {status}")
        return after

    def memory_required(self, plan: TemporalMemoryPlan) -> int:
        if not isinstance(plan, TemporalMemoryPlan):
            raise TypeError("plan must be TemporalMemoryPlan")
        required = c.c_uint64()
        status = self.dll.rw_nr_temporal_memory_required(c.byref(plan), c.byref(required))
        if status:
            raise TemporalError(int(status), f"ABI2 temporal memory planning failed with status {status}")
        return int(required.value)

    def capabilities(self) -> dict[str, Any]:
        return {"abi_version": ABI, "native_execution_available": False,
                "native_temporal_qualified": False, "history_commit_allowed": False,
                "execution_kind": "cpu_contract_only"}

    def session(self, *, session_id: int, epoch: int = 1) -> "TemporalSession":
        handle = c.c_void_p()
        status = self.dll.rw_nr_temporal_session_create(_u64(session_id), _u64(epoch), c.byref(handle))
        if status:
            raise TemporalError(int(status), f"ABI2 temporal session create failed with status {status}")
        return TemporalSession(self, handle)


class TemporalSession:
    """CPU history/epoch control plane; no model or CUDA execution."""

    def __init__(self, library: TemporalLibrary, handle: c.c_void_p):
        self._lock = threading.RLock()
        self.library, self.handle = library, handle

    def _open(self) -> None:
        if not self.handle or not self.handle.value:
            raise RuntimeError("TemporalSession is closed")

    def snapshot(self) -> TemporalState:
        with self._lock:
            self._open()
            state = TemporalState()
            state.struct_size, state.abi_version = c.sizeof(state), ABI
            status = self.library.dll.rw_nr_temporal_session_snapshot(self.handle, c.byref(state))
            if status:
                raise TemporalError(int(status), f"ABI2 temporal session snapshot failed with status {status}")
            return state

    def reset(self, epoch: int) -> None:
        with self._lock:
            self._open()
            status = self.library.dll.rw_nr_temporal_session_reset(self.handle, _u64(epoch))
            if status:
                raise TemporalError(int(status), f"ABI2 temporal session reset failed with status {status}")

    def preflight(self, frame: TemporalFrame) -> dict[str, Any]:
        with self._lock:
            self._open()
            if not isinstance(frame, TemporalFrame):
                raise TypeError("frame must be TemporalFrame")
            result = TemporalPreflightResult()
            result.struct_size, result.abi_version = c.sizeof(result), ABI
            status = self.library.dll.rw_nr_temporal_session_preflight(
                self.handle, c.byref(frame), c.byref(result))
            if status:
                raise TemporalError(int(status), f"ABI2 temporal session preflight failed with status {status}")
            return self.library._result(result)

    def commit(self, frame: TemporalFrame, *, physical_completion: bool,
               output_valid: bool, next_history_owner_id: int) -> TemporalState:
        with self._lock:
            self._open()
            if not isinstance(frame, TemporalFrame):
                raise TypeError("frame must be TemporalFrame")
            state = TemporalState()
            state.struct_size, state.abi_version = c.sizeof(state), ABI
            status = self.library.dll.rw_nr_temporal_session_commit(
                self.handle, c.byref(frame), _bool("physical_completion", physical_completion),
                _bool("output_valid", output_valid),
                _u64(next_history_owner_id), c.byref(state))
            if status:
                raise TemporalError(int(status), f"ABI2 temporal session commit failed with status {status}")
            return state

    def close(self) -> None:
        with self._lock:
            if getattr(self, "handle", None) and self.handle.value:
                handle, self.handle = self.handle, c.c_void_p()
                self.library.dll.rw_nr_temporal_session_release(handle)

    def __enter__(self):
        with self._lock:
            self._open()
            return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try: self.close()
        except BaseException: pass


__all__ = ["ABI", "NO_PREDECESSOR", "NATIVE_EXECUTION_AVAILABLE", "TemporalError",
           "TemporalResource", "TemporalFrame", "TemporalState", "TemporalPreflightResult",
           "TemporalMemoryPlan", "TemporalLibrary", "TemporalSession"]

