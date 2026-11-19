"""Camera-only motion from expected depth of a static Gaussian scene.

This is a renderer-derived estimate with an independent validity mask. Alpha
blending does not provide exact surface identity, so these guides must never be
labelled engine ground truth.
"""
from __future__ import annotations


def derive_camera_motion(depth, alpha, intrinsic, view, previous_depth,
                         previous_alpha, previous_intrinsic, previous_view,
                         *, alpha_min=0.99, absolute_depth_tolerance=0.001,
                         relative_depth_tolerance=0.01):
    import numpy as np

    height, width = depth.shape
    previous_height, previous_width = previous_depth.shape
    if alpha.shape != depth.shape or previous_alpha.shape != previous_depth.shape:
        raise ValueError("Depth and alpha dimensions differ")
    if not (0 < alpha_min <= 1) or min(absolute_depth_tolerance, relative_depth_tolerance) < 0:
        raise ValueError("Invalid guide validity thresholds")
    k = np.asarray(intrinsic, dtype=np.float64)
    previous_k = np.asarray(previous_intrinsic, dtype=np.float64)
    transform = np.asarray(previous_view, dtype=np.float64) @ np.linalg.inv(np.asarray(view, dtype=np.float64))
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    pixel_x, pixel_y = xx + 0.5, yy + 0.5
    z = np.asarray(depth, dtype=np.float64)
    points = np.stack(((pixel_x-k[0, 2])/k[0, 0]*z,
                       (pixel_y-k[1, 2])/k[1, 1]*z, z), axis=-1)
    previous_points = points @ transform[:3, :3].T + transform[:3, 3]
    projected_z = previous_points[..., 2]
    valid = np.isfinite(z) & (z > 0) & np.isfinite(alpha) & (alpha >= alpha_min)
    valid &= np.isfinite(previous_points).all(-1) & (projected_z > 0)
    safe_z = np.where(valid, projected_z, 1.0)
    previous_x = previous_k[0, 0]*previous_points[..., 0]/safe_z + previous_k[0, 2]
    previous_y = previous_k[1, 1]*previous_points[..., 1]/safe_z + previous_k[1, 2]
    valid &= (previous_x >= 0) & (previous_x < previous_width)
    valid &= (previous_y >= 0) & (previous_y < previous_height)
    sample_x = np.clip(np.where(np.isfinite(previous_x), previous_x, 0), 0, previous_width-1).astype(np.int64)
    sample_y = np.clip(np.where(np.isfinite(previous_y), previous_y, 0), 0, previous_height-1).astype(np.int64)
    old_depth = previous_depth[sample_y, sample_x]
    old_alpha = previous_alpha[sample_y, sample_x]
    tolerance = absolute_depth_tolerance + relative_depth_tolerance*np.abs(old_depth)
    valid &= np.isfinite(old_depth) & (old_depth > 0) & np.isfinite(old_alpha) & (old_alpha >= alpha_min)
    valid &= np.abs(projected_z-old_depth) <= tolerance
    motion = np.stack((previous_x/previous_width-pixel_x/width,
                       previous_y/previous_height-pixel_y/height), axis=-1)
    motion = np.where(valid[..., None], motion, 0).astype(np.float32)
    return motion, valid
