"""Thin native gsplat guide adapter; tensor allocation stays with the producer."""
import ctypes as c
from dataclasses import dataclass
from .native_nr import _cuda_plane, ABI
from ._core import CoreError


class _Frame(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("struct_size", "abi_version", "width", "height")] + [
        (name, c.POINTER(c.c_float)) for name in ("rgb_depth", "alpha", "previous_depth", "previous_alpha",
            "color", "depth", "retained_alpha", "motion", "validity")
    ] + [("intrinsics", c.c_double*4), ("previous_intrinsics", c.c_double*4),
         ("current_to_previous", c.c_double*12), ("alpha_min", c.c_float),
         ("absolute_depth_tolerance", c.c_float), ("relative_depth_tolerance", c.c_float), ("reserved", c.c_uint32)]


@dataclass
class PreparedFrame:
    color: object
    depth: object
    alpha: object
    motion: object
    validity: object
    intrinsic: object
    view: object
    has_previous: bool

    @classmethod
    def allocate(cls, height, width, intrinsic, view, *, device):
        import torch
        def plane(channels):
            return torch.empty((height, width, channels), device=device, dtype=torch.float32)
        return cls(plane(3), plane(1), plane(1), plane(2), plane(1), intrinsic, view, False)


class CudaGuideAdapter:
    def __init__(self, library):
        self.dll = c.CDLL(str(library))
        self.dll.rw_gsplat_prepare_cuda.argtypes = [c.POINTER(_Frame), c.c_void_p]
        self.dll.rw_gsplat_prepare_cuda.restype = c.c_int32
        self.dll.rw_gsplat_last_error.argtypes = []
        self.dll.rw_gsplat_last_error.restype = c.c_char_p

    def prepare(self, rendered, alpha, output, *, previous=None, stream):
        """Enqueue on the producer stream; no synchronization or CPU pixels."""
        import numpy as np
        height, width = output.color.shape[:2]
        # The combined gsplat input is HWC4; unlike NR planes it is not RGB.
        if (tuple(rendered.shape) != (height, width, 4) or not rendered.is_contiguous()
                or not rendered.is_cuda or str(rendered.dtype) != "torch.float32"):
            raise ValueError("gsplat RGB+ED must be contiguous HWC4")
        fields = [c.cast(c.c_void_p(rendered.data_ptr()), c.POINTER(c.c_float)),
                  _cuda_plane(alpha, height, width, 1)[0]]
        fields += [_cuda_plane(previous.depth, height, width, 1)[0],
                   _cuda_plane(previous.alpha, height, width, 1)[0]] if previous else [None, None]
        fields += [_cuda_plane(x, height, width, channels)[0] for x, channels in
                   [(output.color,3),(output.depth,1),(output.alpha,1),(output.motion,2),(output.validity,1)]]
        intrinsic = np.asarray(output.intrinsic, dtype=np.float64)
        old_k = intrinsic if previous is None else np.asarray(previous.intrinsic, dtype=np.float64)
        transform = np.eye(4) if previous is None else np.asarray(previous.view, dtype=np.float64) @ np.linalg.inv(np.asarray(output.view, dtype=np.float64))
        camera = lambda k: (c.c_double*4)(k[0,0],k[1,1],k[0,2],k[1,2])
        frame = _Frame(c.sizeof(_Frame), ABI, width, height, *fields, camera(intrinsic), camera(old_k),
                       (c.c_double*12)(*transform[:3].reshape(-1)), .99, .001, .01, 0)
        status = self.dll.rw_gsplat_prepare_cuda(c.byref(frame), int(getattr(stream,"cuda_stream",stream)))
        if status:
            error = self.dll.rw_gsplat_last_error()
            raise CoreError(status, error.decode() if error else "gsplat CUDA guide input is invalid")
        output.has_previous = previous is not None
        return output
