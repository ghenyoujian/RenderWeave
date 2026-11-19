"""Explicit RGB-D/pose to current-to-previous pixel motion projection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np

PoseConvention = Literal["camera_to_world", "world_to_camera"]


@dataclass(frozen=True)
class MotionProjection:
    motion_pixels: np.ndarray
    validity: np.ndarray
    provenance: dict


def _pose_matrix(value: np.ndarray | list[float]) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.size != 16 or not np.isfinite(pose).all():
        raise ValueError("pose must contain 16 finite values")
    pose = pose.reshape(4, 4)
    if not np.array_equal(pose[3], np.array([0., 0., 0., 1.])):
        raise ValueError("pose homogeneous row differs")
    if abs(float(np.linalg.det(pose[:3, :3])) - 1.0) >= 0.02:
        raise ValueError("pose rotation determinant differs")
    return pose


def project_current_to_previous(depth_u16: np.ndarray, current_pose, previous_pose, *,
                                fx: float, fy: float, cx: float, cy: float,
                                depth_scale_m_per_unit: float,
                                pose_convention: PoseConvention,
                                invalid_values: tuple[int, ...] = (0, 65535)) -> MotionProjection:
    """Return current-to-previous pixel vectors in y-down pixel units."""
    depth = np.asarray(depth_u16)
    if depth.ndim != 2 or depth.dtype.kind not in "iu":
        raise ValueError("depth must be a 2-D integer image")
    h, w = depth.shape
    if not all(np.isfinite(v) and v > 0 for v in (fx, fy, cx, cy, depth_scale_m_per_unit)):
        raise ValueError("intrinsics and depth scale must be positive finite values")
    if pose_convention not in ("camera_to_world", "world_to_camera"):
        raise ValueError("pose convention must be explicit")
    current, previous = _pose_matrix(current_pose), _pose_matrix(previous_pose)
    yy, xx = np.indices((h, w), dtype=np.float64)
    z = depth.astype(np.float64) * depth_scale_m_per_unit
    valid = np.isfinite(z) & (z > 0)
    for invalid in invalid_values:
        valid &= depth != invalid
    camera = np.stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z,
                       np.ones_like(z)), axis=-1).reshape(-1, 4).T
    if pose_convention == "camera_to_world":
        previous_camera = np.linalg.inv(previous) @ (current @ camera)
    else:
        previous_camera = previous @ (np.linalg.inv(current) @ camera)
    pz = previous_camera[2].reshape(h, w)
    finite = np.isfinite(previous_camera).all(axis=0).reshape(h, w)
    valid &= finite & (pz > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pu = fx * previous_camera[0].reshape(h, w) / pz + cx
        pv = fy * previous_camera[1].reshape(h, w) / pz + cy
    valid &= np.isfinite(pu) & np.isfinite(pv) & (pu >= 0) & (pu <= w - 1) & (pv >= 0) & (pv <= h - 1)
    motion = np.stack((pu - xx, pv - yy), axis=-1).astype(np.float32)
    motion[~valid] = 0
    payload_sha = hashlib.sha256(motion.tobytes()).hexdigest()
    validity_sha = hashlib.sha256(valid.astype(np.uint8).tobytes()).hexdigest()
    return MotionProjection(motion, valid.astype(np.uint8), {
        "source": "renderer_derived_rgbd_pose_projection", "direction": "current_to_previous",
        "units": "pixels", "depth_definition": "camera_z", "pose_convention": pose_convention,
        "depth_scale_m_per_unit": float(depth_scale_m_per_unit),
        "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
        "shape": [h, w, 2], "validity_shape": [h, w], "valid_fraction": float(valid.mean()),
        "payload_sha256": payload_sha, "validity_sha256": validity_sha,
    })
