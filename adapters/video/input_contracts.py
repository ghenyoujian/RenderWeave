#!/usr/bin/env python3
"""CPU metadata counterexamples for the real media-to-SDK admission boundary."""
from __future__ import annotations

import argparse
import builtins
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from renderweave_video.capability import STATELESS, media_profile, sequence_profile
from renderweave_video.pipeline import checked_media_profile, execute

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))
from common import atomic_json, execution_context, sha256, within


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    context = execution_context()
    storage = Path(context["storage_root"])
    output = within(args.output_dir, storage / "runs")
    output.mkdir(parents=True, exist_ok=False)
    report = {**context, "task_id": "RW-20261101-043", "run_id": args.run_id,
              "status": "RUNNING", "cases": [], "GPU_queried": False,
              "model_or_dataset_payload_read": False, "native_temporal_qualified": False}
    legacy = {"renderer": "gsplat", "status": "COMPLETED",
              "frames": [{"frame_id": 0, "temporal_mode": STATELESS},
                         {"frame_id": 1, "temporal_mode": STATELESS}]}

    def case(name, function, reject=False):
        row = {"name": name, "expected": "reject" if reject else "accept", "passed": False}
        report["cases"].append(row)
        try:
            value = function()
        except ValueError as error:
            row.update(actual="reject", message=str(error))
            if not reject:
                raise
        else:
            row["actual"] = "accept"
            if reject:
                raise AssertionError(name + " silently admitted incompatible semantics")
            if value is False:
                raise AssertionError(name + " failed its positive condition")
        row["passed"] = True

    try:
        case("legacy_sequence", lambda: sequence_profile(legacy, "enhance")["stateless_sdk_compatible"])
        case("legacy_media_explicit_stateless", lambda: media_profile({"temporal_mode": STATELESS}, "enhance")["stateless_sdk_compatible"])
        mutations = {
            "continuous_only_on_later_frame": lambda x: x["frames"][1].update(temporal_mode="continuous_frame_input"),
            "unknown_future_mode": lambda x: x["frames"][1].update(temporal_mode="unknown_future_profile"),
            "explicit_incompatible": lambda x: x["frames"][1].update(backend_consumption={"stateless_sdk_compatible": False}),
            "state_predecessor": lambda x: x["frames"][1].update(expected_predecessor={"frame_id": 0, "scene_epoch": 1}),
            "versioned_unverified_frame": lambda x: x["frames"][1].update(source_frame_contract="rw.gsplat.continuous_sequence.v1"),
            "sequence_contract_requires_mapper": lambda x: x.update(sequence_contract={"contract": "rw.gsplat.continuous_sequence.v1"}),
        }
        for name, mutation in mutations.items():
            changed = copy.deepcopy(legacy)
            mutation(changed)
            case(name, lambda changed=changed: sequence_profile(changed, "enhance"), reject=True)

        continuous = copy.deepcopy(legacy)
        continuous["frames"][1]["temporal_mode"] = "continuous_frame_input"
        encoded_policy = sequence_profile(continuous, "encode")
        case("codec_preserves_incompatibility", lambda: encoded_policy["stateless_sdk_compatible"] is False)
        case("codec_cannot_launder_continuous_input", lambda: media_profile({"input_capability": encoded_policy, "temporal_mode": STATELESS}, "enhance"), reject=True)
        valid_policy = sequence_profile(legacy, "encode")
        for explicit_mode in ("continuous_frame_input", "unknown_future_profile"):
            case("true_policy_cannot_override_" + explicit_mode, lambda explicit_mode=explicit_mode: media_profile(
                {"input_capability": valid_policy, "temporal_mode": explicit_mode}, "enhance"), reject=True)
        case("unknown_legacy_media_profile", lambda: media_profile({}, "enhance"), reject=True)

        source = output / "synthetic-legacy-sequence.json"
        atomic_json(source, legacy)
        old_codec = {"source_manifest": {"path": str(source), "sha256": sha256(source)}}
        case("legacy_codec_source_reverified", lambda: checked_media_profile(old_codec, "enhance", storage, sha256, within)["stateless_sdk_compatible"])
        unknown_wrapper = {**old_codec, "sequence_contract": {"contract": "unknown_future_profile"}}
        case("legacy_source_cannot_override_unknown_wrapper", lambda: checked_media_profile(
            unknown_wrapper, "enhance", storage, sha256, within), reject=True)
        versioned_wrapper = {**old_codec, "source_frame_contract": "unknown_future_frame"}
        case("legacy_source_cannot_override_unknown_frame_wrapper", lambda: checked_media_profile(
            versioned_wrapper, "enhance", storage, sha256, within), reject=True)
        bad_hash = copy.deepcopy(old_codec)
        bad_hash["source_manifest"]["sha256"] = "0" * 64
        case("legacy_codec_source_hash_mismatch", lambda: checked_media_profile(bad_hash, "enhance", storage, sha256, within), reject=True)

        source = output / "synthetic-continuous-sequence.json"
        atomic_json(source, continuous)
        old_continuous = {"source_manifest": {"path": str(source), "sha256": sha256(source)}}
        case("legacy_codec_reverified_continuous_rejected", lambda: checked_media_profile(old_continuous, "enhance", storage, sha256, within), reject=True)
        media = output / "synthetic-continuous-media.json"
        atomic_json(media, {"input_capability": encoded_policy, "temporal_mode": STATELESS})
        original_import = builtins.__import__

        def no_device_or_codec_import(name, *values, **options):
            if name.split(".")[0] in ("av", "numpy", "renderweave"):
                raise AssertionError("Unsupported input reached codec or SDK import: " + name)
            return original_import(name, *values, **options)

        with patch("builtins.__import__", no_device_or_codec_import):
            case("real_execute_sequence_rejects_before_codec_or_SDK", lambda: execute(
                SimpleNamespace(command="enhance", sequence_manifest=source, input_provenance=None),
                {}, context, sha256, within, None), reject=True)
            case("real_execute_media_rejects_before_codec_or_SDK", lambda: execute(
                SimpleNamespace(command="enhance", sequence_manifest=None, input_provenance=media),
                {}, context, sha256, within, None), reject=True)
        report.update(status="PASS", exit_code=0)
    except Exception as error:
        report.update(status="FAILED", exit_code=1, failure={"type": type(error).__name__, "message": str(error)})
    finally:
        atomic_json(output / "input_contracts.json", report)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "output": str(output)}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
