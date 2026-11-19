#!/usr/bin/env python3
"""Render registered, fixed MIP360 Gaussian assets through gsplat, inference only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def import_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


io = import_local("rw_mip360_io", ROOT / "adapters/mip360/io.py")
motion_contract = import_local("rw_mip360_motion", ROOT / "adapters/mip360/motion.py")
sequence_contract = import_local("rw_gsplat_sequence", ROOT / "tools/data/sequence_contract.py")


def event_indices(value):
    if not value:
        return []
    parts = value.split(",")
    if any(not part.isdecimal() for part in parts) or len(parts) != len(set(parts)):
        raise argparse.ArgumentTypeError("Use distinct comma-separated positive frame indices")
    return [int(part) for part in parts]


def load_graphdeco_ply(path, device, max_gaussians):
    """Read the documented raw Graphdeco SH/log-scale/logit-opacity PLY layout."""
    import numpy as np
    import torch
    from plyfile import PlyData

    ply = PlyData.read(str(path), mmap="r")
    if "vertex" not in ply:
        raise ValueError("PLY has no vertex element")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
    required |= {f"scale_{index}" for index in range(3)}
    required |= {f"rot_{index}" for index in range(4)}
    if not required <= names:
        raise ValueError("PLY is not a trained Graphdeco Gaussian asset; sparse COLMAP points are rejected")
    if not 1 <= len(vertex) <= max_gaussians:
        raise ValueError("Gaussian count exceeds the predeclared experimental capacity")
    rest = sorted((name for name in names if name.startswith("f_rest_")), key=lambda name: int(name[7:]))
    if rest != [f"f_rest_{index}" for index in range(len(rest))] or len(rest) % 3:
        raise ValueError("Incomplete spherical-harmonic coefficient layout")
    coefficient_count = 1 + len(rest)//3
    degree = math.isqrt(coefficient_count)-1
    if (degree+1)**2 != coefficient_count or not 0 <= degree <= 3:
        raise ValueError("Unsupported spherical-harmonic degree")

    def stack(fields):
        values = np.stack([vertex[name] for name in fields], axis=-1).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Gaussian parameters contain NaN/Inf")
        return torch.from_numpy(values).to(device)

    means = stack(("x", "y", "z"))
    quats = stack([f"rot_{index}" for index in range(4)])
    norms = torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Invalid Gaussian quaternion")
    quats = quats / norms
    scales = stack([f"scale_{index}" for index in range(3)]).exp()
    opacities = stack(["opacity"]).squeeze(-1).sigmoid()
    dc = stack([f"f_dc_{index}" for index in range(3)]).reshape(-1, 1, 3)
    if rest:
        higher = stack(rest).reshape(-1, 3, len(rest)//3).transpose(1, 2).contiguous()
        colors = torch.cat((dc, higher), dim=1)
    else:
        colors = dc
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid physical Gaussian scales")
    arrays = (means, quats, scales, opacities, colors)
    if any(value.requires_grad for value in arrays):
        raise AssertionError("Gaussian rendering must not retain autograd state")
    return arrays, {"gaussian_count": len(vertex), "sh_degree": degree,
                    "immutable_parameter_bytes": sum(x.numel()*x.element_size() for x in arrays),
                    "source_layout": "graphdeco_raw_SH_log_scale_logit_opacity_quaternion_wxyz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-gaussians", type=int, default=10000000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-frames", type=int,
                        help="Generate an explicitly timed continuous camera trajectory; default keeps independent capture views")
    parser.add_argument("--sequence-id", help="Required stable ID for a generated continuous sequence")
    parser.add_argument("--sequence-anchor-index", type=int, default=0)
    parser.add_argument("--sequence-fps", default="15/1", help="Exact rational generated rate; not capture timestamps")
    parser.add_argument("--camera-step-scene-units", type=float, default=0.002)
    parser.add_argument("--reset-frames", type=event_indices, default=[])
    parser.add_argument("--cut-frames", type=event_indices, default=[])
    parser.add_argument("--max-sequence-output-gib", type=float, default=1.0)
    args = parser.parse_args()
    if args.sequence_frames is None and (args.sequence_id or args.reset_frames or args.cut_frames):
        parser.error("Sequence ID/events require --sequence-frames")
    if args.sequence_frames is not None and not args.sequence_id:
        parser.error("--sequence-frames requires an explicit --sequence-id")
    _, evidence = io.execution_context(args.output_directory / "inputs.json")
    inputs = json.loads(args.input_manifest.read_text())
    assets = json.loads(args.asset_manifest.read_text())
    selected = [entry for entry in inputs["frames"] if entry["scene_id"] == args.scene]
    matches = [entry for entry in assets["assets"] if entry["scene_id"] == args.scene]
    if not selected or len(matches) != 1:
        raise ValueError("Exactly one registered Gaussian asset and at least one input camera are required")
    asset = matches[0]
    if asset.get("coordinate_space") != "mip360_colmap_world":
        raise ValueError("Asset coordinate space is not registered to MIP360 COLMAP")
    ply_path = io.public_input(Path(asset["path"]))
    if io.sha256(ply_path) != asset["sha256"]:
        raise ValueError("Gaussian asset identity mismatch")
    plan = None
    if args.sequence_frames is not None:
        if not 0 <= args.sequence_anchor_index < len(selected):
            raise ValueError("Sequence anchor index is not in the registered scene cameras")
        base = selected[args.sequence_anchor_index]
        if any(base.get("color", {}).get(key) != value for key, value in
               {"primaries": "bt709", "transfer": "srgb", "range": "full", "dtype": "uint8", "alpha": "none"}.items()):
            raise ValueError("Generated sequence requires the registered SDR sRGB anchor color contract")
        plan = sequence_contract.build_plan(
            base, sequence_id=args.sequence_id, source_manifest_sha256=io.sha256(args.input_manifest),
            asset_sha256=asset["sha256"], frames=args.sequence_frames, fps=args.sequence_fps,
            camera_step=args.camera_step_scene_units, reset_frames=args.reset_frames, cut_frames=args.cut_frames)
        for key in ("split", "split_id", "split_config_sha256", "split_assignments"):
            if key not in inputs:
                raise ValueError("Continuous sequence requires scene-level split provenance")
        if args.scene not in inputs["split_assignments"].get(inputs["split"], []):
            raise ValueError("Sequence scene does not belong to its registered split")
        if not math.isfinite(args.max_sequence_output_gib) or args.max_sequence_output_gib <= 0:
            raise ValueError("Sequence output budget must be finite and positive")
        bound = plan["resource_bound"]["output_upper_bytes"]
        if bound > int(args.max_sequence_output_gib * 1024**3):
            raise ValueError("Sequence output upper bound exceeds the explicit disk admission budget")
        ancestor = args.output_directory.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < bound:
            raise ValueError("Insufficient free space for the admitted sequence output upper bound")
        selected = plan["frames"]
    # A new directory prevents accidental overwrite of previous images or failure evidence.
    args.output_directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    report = {"schema_version": 1, **evidence, "task_id": "RW-20261101-039" if plan else "RW-20261101-012",
              "run_id": args.run_id, "status": "RUNNING", "start_utc": started,
              "input_manifest_sha256": io.sha256(args.input_manifest),
              "asset_manifest_sha256": io.sha256(args.asset_manifest), "asset": asset,
              "input_manifest_path": str(args.input_manifest), "asset_manifest_path": str(args.asset_manifest),
              "renderer": "gsplat", "training": False, "frames": [],
              "native_temporal_qualified": False, "resource_contract_status": "experimental_renderer_workspace_not_production_qualified"}
    if plan:
        report["sequence_contract"] = {key: value for key, value in plan.items() if key != "frames"}
        for key in ("split", "split_id", "split_config_sha256", "split_assignments"):
            report[key] = inputs[key]
        report["feature_input_adaptation"] = {
            "direct_native_feature_consumption_verified": False,
            "NR": "pending frozen current-native color/control/motion/history contract; continuous input is not current stateless SDK support",
            "SR": "requires separately fingerprinted raw-F32 to RGBA16F range policy, UV-to-render-pixel motion scale and camera-z-to-device-depth near/far projection; reset absent-motion legality unproven",
            "FG": "requires independently frozen model-specific guide and presentation contract",
            "RR": "missing genuine radiance/material/normal guides; ordinary MIP360 RGB is insufficient",
        }
    try:
        import numpy as np
        import torch
        from PIL import Image
        from gsplat import rasterization

        if not args.device.startswith("cuda:") or not torch.cuda.is_available():
            raise ValueError("A Slurm-visible CUDA device is required; no CPU fallback")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.set_grad_enabled(False)
        torch.cuda.reset_peak_memory_stats(device)
        report["environment"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                                 "gsplat": importlib.metadata.version("gsplat"), "device": str(device),
                                 "device_name": torch.cuda.get_device_name(device),
                                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable"))}
        arrays, resource = load_graphdeco_ply(ply_path, device, args.max_gaussians)
        report["resources"] = resource
        previous = None
        for index, frame in enumerate(selected):
            if plan and frame["reset"]:
                previous = None
            width, height = frame["width"], frame["height"]
            if width <= 0 or height <= 0 or width*height > 16777216:
                raise ValueError("Input image exceeds the explicit 16-Mpixel experimental render capacity")
            intrinsic = frame["camera"]["camera_matrix_for_input"]
            view = frame["camera"]["world_to_camera"]
            k = torch.tensor([intrinsic], device=device, dtype=torch.float32)
            viewmat = torch.tensor([view], device=device, dtype=torch.float32)
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            host_begin = time.monotonic_ns()
            start.record()
            with torch.inference_mode():
                rendered, alpha, _ = rasterization(
                    *arrays, viewmat, k, width, height, sh_degree=resource["sh_degree"],
                    render_mode="RGB+ED", packed=False, rasterize_mode="classic",
                    backgrounds=torch.zeros((1, 3), dtype=torch.float32, device=device),
                )
            stop.record()
            stop.synchronize()
            completed = time.monotonic_ns()
            data, alpha_cpu = rendered[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()
            readback_completed = time.monotonic_ns()
            if (data.shape != (height, width, 4) or alpha_cpu.shape != (height, width)
                    or not np.isfinite(data).all() or not np.isfinite(alpha_cpu).all()):
                failed_prefix = args.output_directory / f"{index:06d}.failed"
                failures = []
                for name, values in (("rendered", data), ("alpha", alpha_cpu)):
                    path = Path(str(failed_prefix)+f".{name}.npy")
                    np.save(path, values, allow_pickle=False)
                    finite = np.isfinite(values)
                    failures.append({"role": name, "path": str(path), "sha256": io.sha256(path),
                                     "shape": list(values.shape), "dtype": str(values.dtype),
                                     "nonfinite_count": int(values.size-np.count_nonzero(finite)),
                                     "finite_min": float(values[finite].min()) if finite.any() else None,
                                     "finite_max": float(values[finite].max()) if finite.any() else None})
                report["failure_sample"] = {"frame_id": index, "camera": frame["camera"], "outputs": failures}
                raise ValueError("Invalid gsplat rendered output")
            # Detach depth from the RGB+ED backing allocation so previous holds
            # only one F32 depth and one F32 alpha plane, not old color frames.
            rgb, depth = data[..., :3], data[..., 3].copy()
            prefix = args.output_directory / f"{index:06d}"
            rgb_path, depth_path, alpha_path = (Path(str(prefix)+suffix) for suffix in (".color.float32.npy", ".depth.float32.npy", ".alpha.float32.npy"))
            np.save(rgb_path, rgb, allow_pickle=False)
            np.save(depth_path, depth, allow_pickle=False)
            np.save(alpha_path, alpha_cpu, allow_pickle=False)
            depth_valid_path = Path(str(prefix)+".depth.valid.npy")
            depth_guide_begin = time.monotonic_ns()
            depth_valid = (alpha_cpu >= 0.99) & (depth > 0)
            depth_guide_compute_ns = time.monotonic_ns()-depth_guide_begin
            np.save(depth_valid_path, depth_valid, allow_pickle=False)
            display_path = Path(str(prefix)+".color.png")
            Image.fromarray(np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(display_path)
            guides = {
                "depth": {"source": "renderer-derived", "valid": True, "path": str(depth_path), "sha256": io.sha256(depth_path), "shape": [height, width], "dtype": "float32", "depth_kind": "camera_z_expected", "definition": "gsplat expected camera-z depth conditional on accumulated alpha", "units": "colmap_scene_units", "valid_mask_path": str(depth_valid_path), "valid_mask_sha256": io.sha256(depth_valid_path), "valid_pixel_fraction": float(depth_valid.mean()), "near": None, "far": None, "reversed_z": False, "per_pixel_validity": "alpha>=0.99 and finite positive depth; blended-depth surface uncertainty remains"},
                "alpha": {"source": "renderer", "valid": True, "path": str(alpha_path), "sha256": io.sha256(alpha_path), "shape": [height, width], "dtype": "float32"},
                "motion": {"source": "absent", "valid": False},
                "normals": {"source": "absent", "valid": False},
                "history": {"source": "absent", "valid": False},
            }
            guide_begin = time.monotonic_ns()
            guide_compute_ns = 0
            if previous is not None:
                motion, valid = motion_contract.derive_camera_motion(
                    depth, alpha_cpu, intrinsic, view,
                    previous["depth"], previous["alpha"], previous["intrinsic"], previous["view"])
                guide_compute_ns = time.monotonic_ns() - guide_begin
                motion_path, valid_path = Path(str(prefix)+".motion.float32.npy"), Path(str(prefix)+".motion.valid.npy")
                np.save(motion_path, motion, allow_pickle=False)
                np.save(valid_path, valid, allow_pickle=False)
                guides["motion"] = {"source": "renderer-derived", "valid": True,
                                    "path": str(motion_path), "sha256": io.sha256(motion_path),
                                    "shape": [height, width, 2], "dtype": "float32",
                                    "valid_mask_path": str(valid_path), "valid_mask_sha256": io.sha256(valid_path),
                                    "direction": "current_to_previous", "units": "uv", "y_axis": "down",
                                    "resolution": [width, height], "previous_resolution": [previous["depth"].shape[1], previous["depth"].shape[0]],
                                    "jitter_included": False, "scene_assumption": "static_gaussians_camera_motion_only",
                                    "validity_rule": {"alpha_min": 0.99, "absolute_depth_tolerance": 0.001, "relative_depth_tolerance": 0.01, "depth_comparison": "previous_expected_z_nearest_pixel"},
                                    "valid_pixel_fraction": float(valid.mean())}
            previous = {"depth": depth, "alpha": alpha_cpu, "intrinsic": intrinsic, "view": view}
            record = {
                "frame_id": index, "scene_id": args.scene,
                "sequence_id": f"mip360:{args.scene}:gsplat_capture_views",
                "input_path": str(display_path), "sha256": io.sha256(display_path),
                "width": width, "height": height, "channels": 3,
                "input_source": "gsplat_render", "provenance": {"gaussian_asset_sha256": asset["sha256"], "camera_input_sha256": frame["sha256"], "source_manifest_sha256": report["input_manifest_sha256"]},
                "raw_color": {"path": str(rgb_path), "sha256": io.sha256(rgb_path), "dtype": "float32", "shape": [height, width, 3], "transfer": "srgb_source_code_values", "nominal_range": [0.0, 1.0], "clamped": False, "min": float(rgb.min()), "max": float(rgb.max())},
                "color": {"primaries": "bt709", "transfer": "srgb", "range": "full", "dtype": "uint8", "alpha": "none", "profile_source": "Gaussian asset fitted to SDR source code values; no implicit linear-to-sRGB conversion", "input_conversion": "explicit clamp_0_1_round_nearest_uint8_for_NR_reference_image_entry"},
                "guides": guides, "camera": frame["camera"],
                "temporal_mode": "stateless_reset_each_frame", "reset": True,
                "pts": None, "time_base": None, "timeline_provenance": "captured camera views have no registered temporal spacing",
                "scene_epoch": index, "native_reference": {"status": "not_collected", "path": None},
                "timing": {"host_submit_ns": host_begin, "device_complete_host_ns": completed,
                           "gpu_elapsed_ms": start.elapsed_time(stop), "host_through_render_ms": (completed-host_begin)/1e6,
                           "host_render_readback_complete_ns": readback_completed,
                           "depth_validity_compute_ns": depth_guide_compute_ns,
                           "camera_motion_compute_ns": guide_compute_ns,
                           "guide_compute_ns": depth_guide_compute_ns+guide_compute_ns},
            }
            if plan:
                for key in ("frame_id", "sequence_id", "session_id", "pts", "time_base", "temporal_mode", "scene_epoch",
                            "reset", "reset_reasons", "events", "expected_predecessor", "jitter", "exposure", "valid_rect", "timeline_provenance"):
                    record[key] = frame[key]
                record["provenance"]["camera_sha256"] = sequence_contract.fingerprint(frame["camera"])
                record["provenance"]["trajectory_plan_sha256"] = report["sequence_contract"]["plan_sha256"]
                record["source_frame_contract"] = sequence_contract.CONTRACT
                record["backend_consumption"] = {"native_temporal": "not_executed", "stateless_sdk_compatible": False}
            record["timing"]["host_input_export_complete_ns"] = time.monotonic_ns()
            report["frames"].append(record)
        report["status"] = "COMPLETED"
        if plan:
            report["sequence_validation"] = sequence_contract.validate_manifest(report, verify_payloads=True)
        report["resources"].update({"measured_peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "measured_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    except Exception as error:
        report["status"] = "FAILED"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["end_utc"] = datetime.now(timezone.utc).isoformat()
        report["exit_code"] = 0 if report["status"] == "COMPLETED" else 1
        io.write_new_json(args.output_directory / "inputs.json", report)
    print(json.dumps({"manifest": str(args.output_directory / "inputs.json"), "frames": len(report["frames"]), "status": report["status"]}))


if __name__ == "__main__":
    main()
