"""Adapt timed renderer input to the original K/LDR/2x SR SDK."""
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

from .temporal import load_motion, read_guide_tensor


MODE = "sr_k_ldr_2x_renderer_guides"
REGISTRATION = Path("config/models/sr_k_310.7.0_capture.json")


def open_runtime(args, context, shape, sha256, within, reader):
    from renderweave import NativeSRRuntime

    storage = Path(context["storage_root"])
    registration_path = within(args.model or Path(context["source_root"]) / REGISTRATION, storage)
    registration = json.loads(registration_path.read_text())
    if registration.get("schema") != "rw.sr.capture_assets.v1" or registration.get("profile") != "LDR_2X":
        raise ValueError("SR requires a K/LDR/2x capture asset registration")
    root = Path(context["shared_assets_root"]) / "weights"
    directory = root / registration["shared_subdirectory"]
    assets = []
    for name in ("weights", "filter"):
        descriptor = registration[name]
        path = within(directory / descriptor["file"], root)
        expected = descriptor["sha256"]
        raw = reader(path, expected) if reader else path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"SR {name} differs from the registered original capture")
        assets.append(raw)
    library = within(args.sdk_library, storage)
    height, width = shape[:2]
    original_fp16 = getattr(args, "sr_original_fp16_backbone", False)
    runtime = NativeSRRuntime(width, height, None, None, library=library, model_bytes=tuple(assets),
                              device=args.device_ordinal, device_budget_bytes=args.device_budget_mib << 20,
                              cuda_graph=True, tiled_backbone=True, parallel_reduce=True, reuse_matrix_rows=True,
                              original_fp16_backbone=original_fp16)
    return runtime, {"initial": runtime.memory_info(), "library_path": str(library), "library_sha256": sha256(library),
                     "parameter_manifest_sha256": sha256(registration_path), "weights_sha256": registration["weights"]["sha256"],
                     "filter_sha256": registration["filter"]["sha256"], "input_wh": [width, height],
                     "output_wh": [2 * width, 2 * height], "temporal_mode": MODE,
                     "original_fp16_backbone": original_fp16}


def project_guides(guides, storage, shape, depth_range, *, reader=None):
    import numpy as np

    h, w = shape
    depth_guide = guides["depth"]
    if (depth_guide.get("valid") is not True or depth_guide.get("depth_kind") != "camera_z_expected"
            or depth_guide.get("units") != "colmap_scene_units"):
        raise ValueError("SR projection requires registered camera-space Z in scene units")
    camera_z = read_guide_tensor(depth_guide, storage, shape, np.float32, reader=reader)
    depth_valid = read_guide_tensor(depth_guide, storage, shape, np.bool_, mask=True, reader=reader)
    motion, motion_valid = load_motion(guides["motion"], storage, h, w, reader=reader)
    near, far = depth_range
    z = camera_z[depth_valid].astype(np.float64)
    if not (np.isfinite(z).all() and (z >= near).all() and (z <= far).all()):
        raise ValueError("Valid camera depth lies outside the declared fixed near/far projection")
    depth = np.ones(shape, dtype=np.float32)
    depth[depth_valid] = (far / (far - near)) * ((z - near) / z)
    # Both original preparation and reconstruction reject a reprojected
    # coordinate outside the viewport. Preserve legitimate zero vectors.
    valid = depth_valid & motion_valid.astype(bool)
    motion[~valid] = (w + 1, 0)
    return motion, depth, {"projection": "perspective_forward_z", "near": near, "far": far,
                           "invalid_depth": "far_plane", "invalid_motion": "reprojects_outside_viewport",
                           "valid_depth_pixels": int(np.count_nonzero(depth_valid)),
                           "valid_motion_and_depth_pixels": int(np.count_nonzero(valid)),
                           "camera_z_valid_min": float(z.min()) if z.size else None,
                           "camera_z_valid_max": float(z.max()) if z.size else None}


def prepare_input(item, levels, storage, depth_range, reset, reader):
    import numpy as np

    prepared = {"motion": None, "depth": None, "timing": {}}
    try:
        start = perf_counter_ns()
        color = np.ascontiguousarray(levels[item["pixels"]])
        prepared["color"] = color
        prepared["normalized_input_sha256"] = hashlib.sha256(memoryview(color)).hexdigest()
        guide_start = perf_counter_ns()
        prepared["timing"]["normalize_and_input_sha"] = (guide_start - start) / 1e6
        if not reset:
            motion, depth, projection = project_guides(item["guides"], storage, color.shape[:2],
                                                       depth_range, reader=reader)
            prepared.update(motion=motion, depth=depth, sr_projection=projection,
                            sr_guide_sha256={"motion": hashlib.sha256(memoryview(motion)).hexdigest(),
                                             "depth": hashlib.sha256(memoryview(depth)).hexdigest()})
        prepared["timing"]["guide_prepare_and_sha"] = (perf_counter_ns() - guide_start) / 1e6
    except Exception as error:
        # The consumer records the same frame's admission failure and input.
        prepared["error"] = error
    return prepared


def display_rgb8(rgb):
    import numpy as np

    scaled = np.clip(rgb, 0, 1)
    scaled *= np.float32(255)
    # Add the half in FP64 so values just below a midpoint stay below it.
    return np.add(scaled, 0.5, dtype=np.float64).astype(np.uint8)


def prepare_delivery(raw, pixels, display, directory, record, save_raw):
    """Fill one host delivery buffer and preserve its evidence before encoding."""
    import numpy as np

    index = record["frame_id"]
    start = perf_counter_ns()
    try:
        if display is not None:
            minimum, maximum, clipped = display(raw, pixels)
        else:
            if not np.isfinite(raw).all():
                raise ValueError("Native SDK returned non-finite color")
            minimum, maximum = float(raw.min()), float(raw.max())
            rgb = raw[..., :3]
            clipped = int(np.count_nonzero((rgb < 0) | (rgb > 1)))
            pixels = display_rgb8(rgb)
    except Exception:
        record["status"] = "FAILED_OUTPUT_PREPARATION"
        np.save(directory / f"frame_{index:06d}.invalid_output.float32.npy", raw, allow_pickle=False)
        raise
    hash_start = perf_counter_ns()
    raw_hash = hashlib.sha256(memoryview(raw)).hexdigest()
    record["raw_output"] = {"dtype": "float32", "shape": list(raw.shape), "sha256": raw_hash,
                            "min": minimum, "max": maximum}
    record["display_clamped_rgb_values"] = clipped
    record["host_delivery_ms"].update(summary_and_pack=(hash_start - start) / 1e6,
                                      raw_sha=(perf_counter_ns() - hash_start) / 1e6)
    if save_raw:
        np.save(directory / f"frame_{index:06d}.native.float32.npy", raw, allow_pickle=False)
    return pixels
