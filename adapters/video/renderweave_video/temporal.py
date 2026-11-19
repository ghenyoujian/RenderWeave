"""Bind decoded frame times to the renderer's motion sidecars."""
from fractions import Fraction
import hashlib
import importlib.util
import io
from pathlib import Path

from .secure_files import json_snapshot, read_bounded


MODE = "sdr_four_head_temporal_candidate"


def timestamp(frame):
    base = frame["time_base"]
    return frame["pts"] * Fraction(base["numerator"], base["denominator"])


def validate_sequence(source_root, manifest):
    spec = importlib.util.spec_from_file_location(
        "rw_media_sequence_contract", Path(source_root) / "tools/data/sequence_contract.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    return contract.validate_manifest(manifest)


def media_source(provenance, source_root, storage):
    """Open the timed source of a lossless encode receipt and bind its frames."""
    if provenance.get("encoder", {}).get("timestamp_policy") != "exact":
        raise ValueError("Temporal video requires an exact-timestamp lossless encode receipt")
    capability = provenance["input_capability"]
    path = Path(capability["provenance_path"])
    _, manifest, digest = json_snapshot(path, storage / "runs")
    if digest != capability["provenance_sha256"]:
        raise ValueError("Continuous source manifest SHA-256 differs from the encode receipt")
    validated = validate_sequence(source_root, manifest)
    timeline, requests = provenance["output_timeline"], provenance["requests"]
    if not 0 < len(timeline) == len(requests) <= len(manifest["frames"]):
        raise ValueError("Temporal video receipt has an incomplete frame map")
    frames = {}
    for frame, encoded, request in zip(manifest["frames"], timeline, requests):
        stamp = timestamp(frame)
        if (timestamp(encoded) != stamp or timestamp(request) != stamp or stamp in frames
                or request["status"] != "ENCODE_ONLY_NO_MODEL_EXECUTION"
                or request["source"]["source_frame_id"] != frame["frame_id"]
                or request["source"]["sha256"] != frame["sha256"]
                or request["encoded_input_sha256"] != encoded["pixel_sha256"]):
            raise ValueError("Encoded frame cannot be joined to its original timed guide")
        frames[stamp] = (frame, encoded["pixel_sha256"])
    return frames, {"sequence": str(path), "sha256": digest, **validated}


def join_frame(item, frames, previous_source_frame):
    stamp = item["pts"] * item["time_base"]
    frame, expected_rgb = frames.get(stamp, (None, None))
    pixels = memoryview(item["pixels"])
    rgb_digest = hashlib.sha256(pixels if pixels.c_contiguous else pixels.tobytes()).hexdigest()
    if frame is None or rgb_digest != expected_rgb:
        raise ValueError("Decoded frame PTS or RGB differs from the guide's encoding receipt")
    frame_id = frame["frame_id"]
    if previous_source_frame is not None and frame_id != previous_source_frame + 1:
        raise ValueError("Decoded frame skipped its registered motion predecessor")
    item.update(guides=frame["guides"], scene_id=frame["scene_id"], reset=frame["reset"],
                decoded_rgb_sha256=rgb_digest,
                reset_reasons=frame["reset_reasons"], source_frame_id=frame_id,
                source_predecessor=frame["expected_predecessor"], source_scene_epoch=frame["scene_epoch"])
    return frame_id


def read_guide_tensor(guide, storage, shape, dtype, *, mask=False, reader=None):
    import numpy as np
    path_key, hash_key = ("valid_mask_path", "valid_mask_sha256") if mask else ("path", "sha256")
    path, expected = Path(guide[path_key]), guide[hash_key]
    length = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    raw = reader(path, expected) if reader else read_bounded(path, storage / "runs", int(length) + 4096)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Guide sidecar SHA-256 differs from the renderer receipt")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if array.shape != shape or array.dtype != dtype:
        raise ValueError("Guide sidecar dtype or geometry differs from the renderer receipt")
    return array


def load_motion(guide, storage, height, width, *, reader=None):
    import numpy as np

    if (guide.get("valid") is not True or guide.get("direction") != "current_to_previous"
            or guide.get("units") != "uv" or guide.get("resolution") != [width, height]):
        raise ValueError("Continuous frame requires its registered current-to-previous UV motion")

    uv = read_guide_tensor(guide, storage, (height, width, 2), np.float32, reader=reader)
    validity = read_guide_tensor(guide, storage, (height, width), np.bool_, mask=True, reader=reader)
    return np.ascontiguousarray(uv * np.array((width, height), np.float32)), np.asarray(validity, dtype=np.float32)
