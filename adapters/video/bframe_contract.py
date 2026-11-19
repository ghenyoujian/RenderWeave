#!/usr/bin/env python3
"""Close the B-frame boundary with codec-only controls before SDK replay.

x264's original lossless branch disables B frames. Test lossless RGB x265
without changing model inputs, dimensions, arithmetic or RGB byte parity.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import sys
import traceback

from renderweave_video.pipeline import digest_array, helpers, main as adapter_main, ratio


SOURCES = [
    {"url": "https://code.videolan.org/videolan/x264/-/raw/master/encoder/encoder.c",
     "observed_sha256": "20a7290a727d7583f3f0fded5f427db6b4c84d934e86ca5df00e75a443a5e2c0",
     "lines": "930-946", "finding": "QP/CRF0 lossless branch sets param.i_bframe=0 after rate-control selection"},
    {"url": "https://raw.githubusercontent.com/FFmpeg/FFmpeg/n8.0/libavcodec/libx265.c",
     "observed_sha256": "2bd7c83dcb74f389fc0269f7fef2ec58e1ca954db9205f4c672cdff44c72ee4a",
     "lines": "923-926", "finding": "libx265 supports 8-bit planar full RGB AV_PIX_FMT_GBRP"},
    {"url": "https://raw.githubusercontent.com/videolan/x265/master/source/encoder/encoder.cpp",
     "observed_sha256": "f8f958f093cb71d812568f278f46fff8961ad901f0f6ce8a2e8b96ed90b7c5ab",
     "lines": "3650-3656", "finding": "lossless branch selects CQP and disables quality metrics; it does not clear bframes"},
]


def codec_arm(name, pixels, count, output, sha256):
    import av
    parameters = {
        "libx264rgb": {"pixel_format": "rgb24", "options": {"crf": "0", "preset": "ultrafast",
                       "x264-params": "bframes=3:b-adapt=0:scenecut=0:keyint=12:min-keyint=12:threads=1"}},
        "libx265": {"pixel_format": "gbrp", "options": {"preset": "ultrafast",
                    "x265-params": "lossless=1:bframes=3:b-adapt=0:b-pyramid=0:scenecut=0:keyint=12:min-keyint=12:rc-lookahead=4:frame-threads=1:pools=none:wpp=0:log-level=error:repeat-headers=1"}},
    }[name]
    path = output / (name + ".nut")
    packets, decoded = [], []
    with av.open(str(path), "w", format="nut", options={"avoid_negative_ts": "disabled"}) as container:
        stream = container.add_stream(name, rate=15)
        stream.width, stream.height = pixels.shape[1], pixels.shape[0]
        stream.pix_fmt = parameters["pixel_format"]
        stream.time_base = Fraction(1, 15)
        context = stream.codec_context
        context.time_base, context.max_b_frames, context.gop_size, context.thread_count = Fraction(1, 15), 3, 12, 1
        context.color_range, context.colorspace, context.color_primaries, context.color_trc = 2, 0, 1, 13
        stream.options = parameters["options"]
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame = frame.reformat(format=parameters["pixel_format"])
            frame.pts, frame.time_base = index + 4, Fraction(1, 15)
            frame.color_range, frame.colorspace = 2, 0
            for packet in stream.encode(frame):
                packets.append({"pts": packet.pts, "dts": packet.dts,
                                "time_base": ratio(Fraction(packet.time_base)) if packet.time_base else None})
                container.mux(packet)
        for packet in stream.encode(None):
            packets.append({"pts": packet.pts, "dts": packet.dts,
                            "time_base": ratio(Fraction(packet.time_base)) if packet.time_base else None,
                            "encoder_flush": True})
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoder = stream.codec_context
        has_b = False
        for index, frame in enumerate(container.decode(stream)):
            if index >= count or frame.pts is None or frame.time_base is None:
                raise RuntimeError("Codec emitted extra/untimed frames")
            actual_time = frame.pts * Fraction(frame.time_base)
            expected_time = Fraction(index + 4, 15)
            rgb = frame.to_ndarray(format="rgb24")
            if actual_time != expected_time:
                raise RuntimeError("Codec changed original rational PTS")
            if digest_array(rgb) != digest_array(pixels):
                raise RuntimeError("Codec changed full RGB bytes; lossless criterion remains strict")
            decoded.append({"pts": frame.pts, "time_base": ratio(Fraction(frame.time_base)),
                            "picture_type": int(frame.pict_type), "rgb_sha256": digest_array(rgb),
                            "format": frame.format.name, "color_range": int(frame.color_range),
                            "colorspace": int(frame.colorspace), "exact_RGB": True, "exact_PTS": True})
        has_b = bool(decoder.has_b_frames)
    if len(decoded) != count:
        raise RuntimeError("Codec flush/EOS lost frames")
    return {"codec": name, "parameters": parameters, "requested_b_frames": 3,
            "path": str(path), "sha256": sha256(path), "decoded": decoded,
            "B_frames_observed": sum(frame["picture_type"] == 3 for frame in decoded),
            "decoder_has_B_frames": has_b, "packets": packets,
            "packet_reordering_observed": any(packet["pts"] is not None and packet["dts"] is not None and
                                               packet["pts"] != packet["dts"] for packet in packets),
            "exact_RGB_and_PTS": True, "decoded_frames": len(decoded)}


def codec_probe(args, context, output, atomic_json, sha256, within, report):
    import av
    import numpy as np
    from PIL import Image
    storage = Path(context["storage_root"])
    manifest_path = within(args.sequence_manifest, storage / "runs")
    source = json.loads(manifest_path.read_text())
    if source.get("renderer") != "gsplat" or source.get("status") != "COMPLETED" or not source.get("frames"):
        raise ValueError("Only real completed gsplat source frames are permitted")
    frame = source["frames"][0]
    path = within(Path(frame["input_path"]), storage / "runs")
    if frame.get("input_source") != "gsplat_render" or sha256(path) != frame["sha256"]:
        raise ValueError("Real source frame identity differs")
    color = frame.get("color", {})
    if any(color.get(key) != value for key, value in {"primaries": "bt709", "transfer": "srgb", "range": "full", "dtype": "uint8"}.items()):
        raise ValueError("Source is not the frozen full-color SDR RGB profile")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("RGB source required")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    report.update(source_manifest={"path": str(manifest_path), "sha256": sha256(manifest_path)},
                  source_frame={"path": str(path), "sha256": sha256(path), "width": pixels.shape[1], "height": pixels.shape[0],
                                "rgb_sha256": digest_array(pixels), "resize": False, "pixel_modification": False},
                  environment={"pyav": av.__version__, "FFmpeg_libraries": av.library_versions},
                  codec_arms=[])
    for name in ("libx264rgb", "libx265"):
        arm = codec_arm(name, pixels, args.frames, output, sha256)
        report["codec_arms"].append(arm)
        atomic_json(output / "bframe_contract.json", report)
    control, candidate = report["codec_arms"]
    if control["B_frames_observed"] != 0:
        raise RuntimeError("x264 lossless control falsified source-derived no-B hypothesis; inspect rather than reinterpret")
    if candidate["B_frames_observed"] == 0 or not candidate["decoder_has_B_frames"] or not candidate["packet_reordering_observed"]:
        raise RuntimeError("Candidate produced no actual reordered B frames")
    report.update(status="CODEC_BOUNDARY_PASSED", derived_from_mip360_gsplat=True,
                  output_video=candidate["path"], output_video_sha256=candidate["sha256"],
                  source_replay_frames=args.frames, root_cause="x264_lossless_forces_i_bframe_zero",
                  source_content_cause="held_constant_between_arms; x265 static-frame B output falsifies content-only explanation")


def sdk_replay(args, context, output, atomic_json, sha256, within, report):
    storage = Path(context["storage_root"])
    probe_path = within(args.probe_evidence, storage / "runs")
    probe = json.loads(probe_path.read_text())
    if probe.get("status") != "CODEC_BOUNDARY_PASSED" or probe.get("exit_code") != 0:
        raise RuntimeError("A passed codec-only boundary is required before GPU replay")
    if probe.get("source_replay_frames") != args.frames:
        raise RuntimeError("Probe frame count differs from preregistered replay resource count")
    video = within(Path(probe["output_video"]), storage / "runs")
    if sha256(video) != probe["output_video_sha256"]:
        raise ValueError("Codec-only video changed")
    report["codec_probe"] = {"path": str(probe_path), "sha256": sha256(probe_path)}
    report["cases"] = []
    def invoke(name, argv, expected):
        directory = output / name
        arguments = [*argv, "--output-dir", str(directory), "--run-id", args.run_id + ":" + name,
                     "--color-profile", "srgb-bt709-full-rgb8", "--max-frames", str(probe["source_replay_frames"])]
        code = adapter_main(arguments)
        case = {"name": name, "argv": arguments, "exit_code": code, "expected_exit": expected,
                "passed": code == expected, "evidence": str(directory / "adapter.json"),
                "evidence_sha256": sha256(directory / "adapter.json")}
        report["cases"].append(case)
        atomic_json(output / "bframe_contract.json", report)
        if code != expected:
            raise RuntimeError(f"{name} exit{code} differs from preregistered {expected}")
        return json.loads((directory / "adapter.json").read_text())
    sdk = ["--model", str(args.model), "--plan", str(args.plan), "--sdk-library", str(args.sdk_library)]
    observed = invoke("bframes_sdk", ["enhance", "--input-video", str(video), "--input-provenance", str(probe_path), *sdk], 0)
    if observed["counts"]["delivered_encoded_frames"] != probe["source_replay_frames"]:
        raise RuntimeError("Native SDK path lost B-decoded frames")
    if not any(frame["decode"]["picture_type"] == 3 for frame in observed["requests"]):
        raise RuntimeError("No actual B frame reached native SDK admission")
    if any(frame["sdk_job"]["physical_completion_confirmed"] != 1 for frame in observed["requests"]):
        raise RuntimeError("SDK output published without physical completion")
    source_manifest = within(Path(probe["source_manifest"]["path"]), storage / "runs")
    if sha256(source_manifest) != probe["source_manifest"]["sha256"]:
        raise RuntimeError("Original sequence manifest changed before negative tests")
    invoke("missing_pts_rejected", ["encode", "--sequence-manifest", str(source_manifest)], 1)
    invalid = copy.deepcopy(json.loads(source_manifest.read_text()))
    invalid["frames"][0]["color"]["transfer"] = "smpte2084"
    invalid_path = output / "invalid_HDR_metadata.json"
    atomic_json(invalid_path, invalid)
    invoke("HDR_rejected", ["encode", "--sequence-manifest", str(invalid_path), "--generated-fps", "15"], 1)
    report.update(status="BFRAME_SDK_AND_REMAINING_GUARDS_PASSED", actual_B_frames_admitted=True,
                  prior_passed_CFR_VFR_seek_timeout_cases_repeated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("codec-only", "sdk-replay"))
    parser.add_argument("--sequence-manifest", type=Path)
    parser.add_argument("--probe-evidence", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--sdk-library", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.mode == "codec-only" and args.sequence_manifest is None:
        parser.error("codec-only requires --sequence-manifest")
    if args.mode == "sdk-replay" and any(value is None for value in (args.probe_evidence, args.model, args.plan, args.sdk_library)):
        parser.error("sdk-replay requires passed codec probe and original native SDK model/plan/library")
    if not 6 <= args.frames <= 16:
        parser.error("Use only6..16 small boundary frames, never an unbounded codec search")
    _, context, atomic_json, sha256, within = helpers()
    output = within(args.output_dir, Path(context["storage_root"]) / "runs")
    output.mkdir(parents=True, exist_ok=False)
    contract = {"mode": args.mode, "codec_controls": ["libx264rgb_qp0_badapt0_b3", "libx265_lossless1_badapt0_b3"],
                "frames": args.frames, "content": "one_real_gsplat_RGB_frame_replayed_unchanged_in_both_arms",
                "generated_start_pts": 4, "generated_time_base": {"numerator": 1, "denominator": 15},
                "timeline_provenance": "explicit_generated_codec_replay_not_camera_capture_timestamps",
                "full_RGB_byte_parity": "exact", "original_rational_PTS": "exact",
                "candidate_actual_B_picture_count": ">0", "candidate_packet_PTS_DTS_reordering": True,
                "SDK_replay_requires_codec_only_pass": True, "model_parameters_or_dimensions_changed": False,
                "independent_quality_sequences": 0, "qualification_passed": False,
                "source_audit": SOURCES, "registered_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "pre_registered_contract.json", contract)
    report = {"schema_version": 1, "project_id": "renderweave", "task_id": "RW-20261101-021",
              "run_id": args.run_id, "mode": args.mode, "execution": context,
              "start_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
              "contract_sha256": sha256(output / "pre_registered_contract.json"),
              "source_audit": SOURCES, "independent_quality_sequences": 0,
              "native_temporal_qualified": False, "production_qualified": False}
    code = 0
    try:
        if args.mode == "codec-only":
            codec_probe(args, context, output, atomic_json, sha256, within, report)
        else:
            sdk_replay(args, context, output, atomic_json, sha256, within, report)
    except Exception as error:
        code = 1
        report.update(status="FAILED", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        report.update(exit_code=code, end_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output / "bframe_contract.json", report)
    print(json.dumps({"status": report["status"], "evidence": str(output / "bframe_contract.json"), "exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
