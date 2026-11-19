#!/usr/bin/env python3
"""Run real FFmpeg+compiled SDK CFR/VFR/B-frame/seek/EOS contracts on MIP gsplat frames."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

from renderweave_video.pipeline import helpers, main as adapter_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sdk-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    if not 6 <= args.frames <= 32:
        parser.error("Use 6..32 real rendered frame replays for the initial codec/lifecycle contract")
    _, context, atomic_json, sha256, within = helpers()
    output = within(args.output_dir, Path(context["storage_root"]) / "runs")
    output.mkdir(parents=True, exist_ok=False)
    source = within(args.sequence_manifest, Path(context["storage_root"]) / "runs")
    manifest = json.loads(source.read_text())
    if not manifest.get("frames") or manifest.get("renderer") != "gsplat" or manifest.get("status") != "COMPLETED":
        raise ValueError("A real completed gsplat input sequence is required")
    frames = manifest["frames"]
    replay = copy.deepcopy(manifest)
    replay["frames"] = [copy.deepcopy(frames[index % len(frames)]) for index in range(args.frames)]
    replay.update(codec_contract_replay=True, independent_quality_sequences=0,
                  replay_note="Same registered rendered views replayed solely for codec/lifecycle boundaries; not independent camera or quality sequences")
    replay_path = output / "replay_manifest.json"
    atomic_json(replay_path, replay)
    report = {"project_id": "renderweave", "task_id": "RW-20261101-021", "run_id": args.run_id,
              "execution": context, "original_render_manifest_sha256": sha256(source), "cases": [],
              "native_temporal_qualified": False, "production_qualified": False,
              "independent_quality_sequences": 0, "audio_stream_rejection_tested": False}

    def run_case(name, argv, expected_exit=0):
        directory = output / name
        arguments = [*argv, "--output-dir", str(directory), "--run-id", args.run_id + ":" + name,
                     "--color-profile", "srgb-bt709-full-rgb8", "--max-frames", str(args.frames)]
        case = {"name": name, "argv": arguments, "expected_exit": expected_exit}
        report["cases"].append(case)
        try:
            code = adapter_main(arguments)
            case.update(exit_code=code, passed=code == expected_exit)
            if (directory / "adapter.json").is_file():
                case["evidence"] = str(directory / "adapter.json")
                case["evidence_sha256"] = sha256(directory / "adapter.json")
            if code != expected_exit:
                raise RuntimeError(f"{name} exit{code} differs from preregistered {expected_exit}")
            if code == 0:
                observation = json.loads((directory / "adapter.json").read_text())
                if "--seek-pts" not in argv and observation["counts"]["delivered_encoded_frames"] != args.frames:
                    raise RuntimeError("Codec or SDK path lost an input frame during decode/flush/EOS")
                events = [json.loads(line) for line in (directory / "audit.jsonl").read_text().splitlines()]
                if not any(event["event"] == "input_eos" for event in events):
                    raise RuntimeError("Natural input EOS was not observed; frame limit is not EOS proof")
        except Exception as error:
            case.update(passed=False, failure=str(error), traceback=traceback.format_exc())
            raise
        finally:
            atomic_json(output / "codec_contracts.json", report)
        return json.loads((directory / "adapter.json").read_text())

    sdk = ["--model", str(args.model), "--plan", str(args.plan), "--sdk-library", str(args.sdk_library)]
    try:
        cfr = run_case("cfr_encode", ["encode", "--sequence-manifest", str(replay_path), "--generated-fps", "15"])
        cfr_video = ["--input-video", cfr["output_video"], "--input-provenance", str(output / "cfr_encode/adapter.json")]
        run_case("cfr_sdk", ["enhance", *cfr_video, *sdk])
        vfr = run_case("vfr_encode", ["encode", "--sequence-manifest", str(replay_path),
                                      "--generated-fps", "15", "--vfr-steps", "1,2,1,3"])
        run_case("vfr_sdk", ["enhance", "--input-video", vfr["output_video"],
                              "--input-provenance", str(output / "vfr_encode/adapter.json"), *sdk])
        target_pts = cfr["output_timeline"][3]["pts"]
        seek = run_case("seek_sdk", ["enhance", *cfr_video, "--seek-pts", str(target_pts), *sdk])
        if not seek["requests"] or "seek" not in seek["requests"][0].get("reset_reasons", []):
            raise RuntimeError("Seek did not create the preregistered SDK epoch reset")
        timed = run_case("wait_timeout_sdk", ["enhance", *cfr_video, "--wait-timeout-ms", "0", *sdk])
        report["wait_timeout_observed"] = timed["counts"]["wait_timeouts"] > 0
        if not report["wait_timeout_observed"]:
            raise RuntimeError("Zero-time wait did not observe a timeout; timeout path remains untested")
        bframes = run_case("bframes_encode", ["encode", "--sequence-manifest", str(replay_path),
                                              "--generated-fps", "15", "--generated-start-pts", "4",
                                              "--codec", "libx264rgb", "--b-frames", "3"])
        report["b_frames_observed"] = any(frame["picture_type"] == 3 for frame in bframes["output_timeline"])
        if not report["b_frames_observed"]:
            raise RuntimeError("Codec produced no actual B frames; requested options alone are not proof")
        run_case("bframes_sdk", ["enhance", "--input-video", bframes["output_video"],
                                  "--input-provenance", str(output / "bframes_encode/adapter.json"), *sdk])
        run_case("missing_pts_rejected", ["encode", "--sequence-manifest", str(replay_path)], expected_exit=1)
        bad = copy.deepcopy(replay)
        bad["frames"][0]["color"]["transfer"] = "smpte2084"
        bad_path = output / "invalid_HDR_metadata.json"
        atomic_json(bad_path, bad)
        run_case("HDR_rejected", ["encode", "--sequence-manifest", str(bad_path), "--generated-fps", "15"], expected_exit=1)
        report.update(status="CONTRACT_CASES_PASSED", exit_code=0)
    except Exception as error:
        report.update(status="FAILED_OR_UNCOVERED", exit_code=1, error=str(error), traceback=traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        atomic_json(output / "codec_contracts.json", report)
    print(json.dumps({"status": report["status"], "evidence": str(output / "codec_contracts.json"), "exit_code": report["exit_code"]}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
