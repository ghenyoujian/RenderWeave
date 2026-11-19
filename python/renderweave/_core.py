"""ctypes owns no operator implementation and does not depend on PyTorch."""
from __future__ import annotations

import ctypes as c
import json
import os
from pathlib import Path
import sys
from typing import Iterable

ABI_VERSION = 1


class CoreError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class _ContextDesc(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32), ("abi_version", c.c_uint32),
        ("device_kind", c.c_uint32), ("device_ordinal", c.c_int32),
        ("memory_budget_bytes", c.c_uint64), ("max_sessions", c.c_uint32),
        ("max_inflight", c.c_uint32), ("execution_slots", c.c_uint32),
        ("flags", c.c_uint32), ("config_json", c.c_char_p),
    ]


class _MemoryPlan(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("abi_version", c.c_uint32)] + [
        (name, c.c_uint64) for name in (
            "budget_bytes", "shared_weights_bytes", "bytes_per_slot", "slot_count",
            "bytes_per_session", "session_count", "import_output_bytes",
            "graph_handle_bytes", "reserve_quarantine_bytes",
        )
    ]


def _unsigned64(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << 64):
        raise ValueError("expected an unsigned 64-bit integer")
    return value


def _resolve_library(explicit: str | os.PathLike[str] | None) -> Path:
    chosen = explicit if explicit is not None else os.environ.get("RW_LIBRARY_PATH")
    names = ("rw_core.dll",) if sys.platform == "win32" else (
        ("librw_core.dylib",) if sys.platform == "darwin" else ("librw_core.so", "librw_core.so.0")
    )
    candidates: list[Path] = []
    if chosen:
        selected = Path(chosen).expanduser()
        candidates = [selected / name for name in names] if selected.is_dir() else [selected]
    else:
        for directory in (Path(__file__).parent / "lib", Path(sys.prefix) / "lib",
                          Path(sys.prefix) / "bin", Path(sys.prefix) / "Library/bin"):
            candidates.extend(directory / name for name in names)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise FileNotFoundError("librw_core was not found; set RW_LIBRARY_PATH to the installed library")


def _sdk_library_path(component, explicit):
    name = f"rw_{component}_sdk.dll" if sys.platform == "win32" else f"librw_{component}_sdk.so"
    variable = f"RW_{component.upper()}_LIBRARY_PATH"
    selected = explicit or os.environ.get(variable)
    if selected:
        path = Path(selected).expanduser()
        return (path / name if path.is_dir() else path).resolve(strict=True)
    directories = ("Library/bin", "bin") if sys.platform == "win32" else ("lib",)
    for directory in directories:
        path = Path(sys.prefix) / directory / name
        if path.is_file():
            return path.resolve(strict=True)
    raise FileNotFoundError(f"{name} was not found; set {variable} to the installed library")


class Runtime:
    """Owner of the current CPU contract/arithmetic SDK.

    capabilities()['native_model_execution_available'] is false in this initial
    build. An unavailable CUDA backend raises UNSUPPORTED; it never falls back.
    ctypes.CDLL releases the GIL during C ABI calls.
    """

    def __init__(self, device: str = "cpu-reference", *, library=None, memory_budget_bytes: int = 0):
        self.library_path = _resolve_library(library)
        self._library = c.CDLL(str(self.library_path))
        self._context = c.c_void_p()
        self._configure()
        if self._library.rw_abi_version() != ABI_VERSION:
            raise CoreError(2, "ABI_MISMATCH")
        if device == "cpu-reference":
            kind, ordinal = 1, 0
        elif device.startswith("cuda:"):
            kind, ordinal = 2, int(device.split(":", 1)[1])
        else:
            raise ValueError("device must be cpu-reference or an explicit cuda:N")
        if not 0 <= ordinal < (1 << 31):
            raise ValueError("invalid device ordinal")
        self._device_kind = kind
        descriptor = _ContextDesc(c.sizeof(_ContextDesc), ABI_VERSION, kind, ordinal,
                                  _unsigned64(memory_budget_bytes), 1, 1, 1, 0, None)
        self._check(self._library.rw_context_create(c.byref(descriptor), c.byref(self._context)))

    def _configure(self):
        lib = self._library
        lib.rw_abi_version.argtypes, lib.rw_abi_version.restype = [], c.c_uint32
        lib.rw_status_string.argtypes, lib.rw_status_string.restype = [c.c_int32], c.c_char_p
        lib.rw_context_create.argtypes = [c.POINTER(_ContextDesc), c.POINTER(c.c_void_p)]
        lib.rw_context_create.restype = c.c_int32
        lib.rw_context_query.argtypes = [c.c_void_p, c.c_char_p, c.c_void_p, c.POINTER(c.c_size_t)]
        lib.rw_context_query.restype = c.c_int32
        lib.rw_context_release.argtypes, lib.rw_context_release.restype = [c.c_void_p], None
        lib.rw_e4m3_decode_array.argtypes = [c.POINTER(c.c_uint8), c.POINTER(c.c_float), c.c_size_t]
        lib.rw_e4m3_decode_array.restype = c.c_int32
        lib.rw_e4m3_encode_array.argtypes = [c.POINTER(c.c_float), c.POINTER(c.c_uint8), c.c_size_t]
        lib.rw_e4m3_encode_array.restype = c.c_int32
        lib.rw_memory_plan_required.argtypes = [c.POINTER(_MemoryPlan), c.POINTER(c.c_uint64)]
        lib.rw_memory_plan_required.restype = c.c_int32

    def _check(self, status):
        if status:
            raise CoreError(int(status), self._library.rw_status_string(status).decode("ascii"))

    def _require_open(self):
        if not self._context.value:
            raise RuntimeError("Runtime is closed")

    def capabilities(self) -> dict:
        self._require_open()
        count = c.c_size_t()
        self._check(self._library.rw_context_query(self._context, b"capabilities", None, c.byref(count)))
        data = c.create_string_buffer(count.value)
        self._check(self._library.rw_context_query(self._context, b"capabilities", data, c.byref(count)))
        return json.loads(data.value)

    def encode_e4m3(self, values: Iterable[float]) -> bytes:
        self._require_open()
        values = tuple(values)
        source = (c.c_float * len(values))(*values)
        output = (c.c_uint8 * len(values))()
        self._check(self._library.rw_e4m3_encode_array(source, output, len(values)))
        return bytes(output)

    def decode_e4m3(self, codes: bytes | bytearray | Iterable[int]) -> list[float]:
        self._require_open()
        codes = bytes(codes)
        source = (c.c_uint8 * len(codes)).from_buffer_copy(codes)
        output = (c.c_float * len(codes))()
        self._check(self._library.rw_e4m3_decode_array(source, output, len(codes)))
        return list(output)

    def memory_required(self, *, budget_bytes: int, shared_weights_bytes: int,
                        bytes_per_slot: int, slot_count: int, bytes_per_session: int,
                        session_count: int, import_output_bytes: int = 0,
                        graph_handle_bytes: int = 0, reserve_quarantine_bytes: int = 0) -> int:
        self._require_open()
        plan = _MemoryPlan(c.sizeof(_MemoryPlan), ABI_VERSION, *[
            _unsigned64(value) for value in (
                budget_bytes, shared_weights_bytes, bytes_per_slot, slot_count,
                bytes_per_session, session_count, import_output_bytes,
                graph_handle_bytes, reserve_quarantine_bytes,
            )
        ])
        needed = c.c_uint64()
        self._check(self._library.rw_memory_plan_required(c.byref(plan), c.byref(needed)))
        return needed.value

    def temporal_session(self, *, session_id: int, epoch: int = 1):
        """Return the ABI2 CPU session control plane for this runtime.

        CUDA/model execution remains explicitly unsupported; this method is
        only the shared host-adapter transaction boundary.
        """
        self._require_open()
        if self._device_kind != 1:
            raise CoreError(3, "ABI2 temporal session is CPU-contract-only")
        from .temporal import TemporalLibrary
        return TemporalLibrary(str(self.library_path)).session(session_id=session_id, epoch=epoch)

    def close(self):
        if getattr(self, "_context", None) and self._context.value:
            self._library.rw_context_release(self._context)
            self._context = c.c_void_p()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
