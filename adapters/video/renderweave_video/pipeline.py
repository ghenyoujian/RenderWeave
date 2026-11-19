"""Enhance registered gsplat sequences or their lossless video encode through the native SDK."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from functools import partial
import hashlib
import importlib.metadata
import io
from itertools import chain, count
import json
import math
import os
from pathlib import Path
import sys
from threading import Lock
import time
import traceback

from .capability import media_profile, sequence_profile
from .secure_files import json_snapshot, read_bounded
from . import sr, temporal


def helpers(execution=None):
    source = Path(os.environ["PROJECT_SOURCE_ROOT"]).resolve()
    sys.path.insert(0, str(source / "tools/reference"))
    from common import atomic_json, execution_context, sha256, within
    context = execution if execution is not None else execution_context()
    sys.path.insert(0, str(source / "python"))
    return source, context, atomic_json, sha256, within


def rational(value):
    value = Fraction(str(value))
    if value <= 0:
        raise ValueError("Time bases and frame rates must be positive rational numbers")
    return value


def ratio(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def digest_array(value):
    buffer = memoryview(value)
    return hashlib.sha256(buffer if buffer.c_contiguous else value.tobytes()).hexdigest()


def input_operation(args):
    if getattr(args, "sdr_temporal_head", False) or getattr(args, "feature", "nr") == "sr":
        return "encode"
    return args.command


def _lexical_source_path(value):
    """Restore CLI path syntax without following any filesystem links."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def checked_media_profile(provenance, command, storage, sha256, within):
    """Recompute declared compatibility from its hash-bound source chain.

    A codec receipt can carry a capability declaration, but that declaration
    cannot upgrade the continuous/unknown semantics of the referenced source.
    Historical receipts containing only an explicit mode are codec-only.  A
    descendant must not turn that unverified leaf into a model input.
    """
    fields = ("contract", "source_kind", "stateless_sdk_compatible", "model_input_profile",
              "guides_preserved_in_video", "native_temporal_qualified")

    def read_source(descriptor, seen):
        if not isinstance(descriptor, dict):
            raise ValueError("Media source identity is missing")
        path_value, expected = descriptor.get("path"), descriptor.get("sha256")
        if (not isinstance(path_value, str) or not path_value or not isinstance(expected, str)
                or len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected)):
            raise ValueError("Media source path or SHA-256 is invalid")
        path = _lexical_source_path(path_value)
        if path in seen or len(seen) >= 8:
            raise ValueError("Media source chain is cyclic or exceeds eight links")
        raw, value, actual = json_snapshot(path, storage / "runs")
        if actual != expected:
            raise ValueError("Media source manifest changed or does not match its SHA-256")
        return value, path, seen | {path}

    def evaluate(value, operation, seen):
        if not isinstance(value, dict):
            raise ValueError("Media provenance must be an object")
        declaration = value.get("input_capability")
        if declaration is not None:
            if not isinstance(declaration, dict):
                raise ValueError("Invalid media input capability")
            parent_seen = seen
            source, path, seen = read_source({"path": declaration.get("provenance_path"),
                                              "sha256": declaration.get("provenance_sha256")}, seen)
            if source.get("renderer") == "gsplat":
                actual = sequence_profile(source, "encode")
            elif source.get("derived_from_mip360_gsplat") is True:
                actual = evaluate(source, "encode", seen)
            else:
                raise ValueError("Unknown media source kind; capability cannot be inherited")
            if any(type(declaration.get(key)) is not type(actual.get(key))
                   or declaration.get(key) != actual.get(key) for key in fields):
                raise ValueError("Media capability contradicts its hash-bound source")
            extra_ref = value.get("source_manifest")
            if extra_ref is not None:
                extra, _, _ = read_source(extra_ref, parent_seen)
                if extra.get("renderer") != "gsplat":
                    raise ValueError("Unknown additional media source kind")
                extra_capability = sequence_profile(extra, "encode")
                for key in ("stateless_sdk_compatible", "model_input_profile", "guides_preserved_in_video",
                            "native_temporal_qualified"):
                    if type(extra_capability[key]) is not type(actual[key]) or extra_capability[key] != actual[key]:
                        raise ValueError("Additional media source contradicts the input capability")
            result = media_profile(value, operation)
            result["source_capability_reverified"] = {"path": str(path), "sha256": declaration["provenance_sha256"]}
            return result
        # Existing codec receipts sometimes contain a source reference without
        # a declaration; never let an explicit mode override that source.
        source_ref = value.get("source_manifest")
        if source_ref is not None:
            source, path, seen = read_source(source_ref, seen)
            if source.get("renderer") != "gsplat":
                raise ValueError("Unknown legacy media source kind")
            verified = sequence_profile(source, "encode")
            result = media_profile({**value, "input_capability": verified}, operation)
            result["legacy_source_manifest_reverified"] = {"path": str(path), "sha256": source_ref["sha256"]}
            return result
        return media_profile(value, operation)

    return evaluate(provenance, command, set())


def read_media_snapshot(args, context, sha256, within, snapshot=None):
    storage = Path(context["storage_root"])
    provenance_path, _, provenance, provenance_sha = snapshot or _source_snapshot(args.input_provenance, storage)
    operation = input_operation(args)
    capability = checked_media_profile(provenance, operation, storage, sha256, within)
    if provenance.get("derived_from_mip360_gsplat") is not True:
        raise ValueError("Input video requires derived_from_mip360_gsplat=true")
    path = _lexical_source_path(args.input_video)
    if type(args.max_output_mib) is not int or not 0 < args.max_output_mib <= 1024:
        raise ValueError("Media input snapshot budget must be within 1 GiB")
    maximum = args.max_output_mib << 20
    raw = read_bounded(path, storage / "runs", maximum)
    actual = hashlib.sha256(raw).hexdigest()
    if provenance.get("output_video_sha256") != actual:
        raise ValueError("Input video SHA-256 differs from provenance")
    return {"path": str(path), "sha256": actual, "bytes": len(raw),
            "snapshot_maximum_bytes": maximum, "provenance_path": str(provenance_path),
            "provenance_sha256": provenance_sha, "input_capability": capability,
            "performed_before_model_creation": True}, raw


def validate_media_input(args, context, sha256, within, snapshot=None):
    """Verify the media identity without starting the decoder or model."""
    return read_media_snapshot(args, context, sha256, within, snapshot)[0]


class Audit:
    def __init__(self, path):
        self.stream = path.open("x", encoding="utf-8")
        self.lock = Lock()

    def emit(self, event, **fields):
        with self.lock:
            value = {"event": event, "host_ns": time.perf_counter_ns(), **fields}
            self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
        return value

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def _source_snapshot(path, storage):
    path = _lexical_source_path(path)
    raw, value, digest = json_snapshot(path, storage / "runs")
    return path, raw, value, digest


def sequence_frames(args, context, sha256, within, audit, snapshot=None, *, reader=None):
    import numpy as np
    from PIL import Image
    storage = Path(context["storage_root"])
    manifest_path, _, manifest, manifest_sha = snapshot or _source_snapshot(args.sequence_manifest, storage)
    sequence_profile(manifest, input_operation(args))
    if manifest.get("renderer") != "gsplat" or manifest.get("status") != "COMPLETED":
        raise ValueError("Only a completed registered gsplat render manifest is accepted")
    if args.generated_fps is None and any(frame.get("pts") is None for frame in manifest["frames"]):
        raise ValueError("Untimed camera views require an explicitly generated frame rate")
    rate = rational(args.generated_fps) if args.generated_fps else None
    registered_rate = manifest.get("sequence_contract", {}).get("trajectory", {}).get("fps")
    steps = [int(value) for value in args.vfr_steps.split(",")]
    if not steps or any(value < 1 for value in steps):
        raise ValueError("Generated VFR steps must all be positive integers")
    pts = args.generated_start_pts
    for index, frame in enumerate(manifest["frames"]):
        if frame.get("input_source") != "gsplat_render" or not frame.get("provenance"):
            raise ValueError("Sequence frames require real Gaussian/camera render provenance")
        profile = frame.get("color", {})
        if (profile.get("transfer") != "srgb" or profile.get("primaries") != "bt709" or
                profile.get("range") != "full" or profile.get("dtype") != "uint8"):
            raise ValueError("Sequence does not bind the declared uint8 sRGB/BT709/full profile")
        path = Path(frame["input_path"])
        if reader:
            payload = reader(path, frame["sha256"])
            if hashlib.sha256(payload).hexdigest() != frame["sha256"]:
                raise ValueError("Registered gsplat frame changed")
            image_source = io.BytesIO(payload)
        else:
            image_source = within(path, storage / "runs")
            if sha256(image_source) != frame["sha256"]:
                raise ValueError("Registered gsplat frame changed")
        with Image.open(image_source) as image:
            if image.mode != "RGB":
                raise ValueError("Initial gsplat media profile requires explicit RGB, no alpha conversion")
            pixels = np.asarray(image, dtype=np.uint8).copy()
        if rate:
            frame_pts, base = pts, Fraction(rate.denominator, rate.numerator)
            pts += steps[index % len(steps)]
            timeline = "explicit_generated_camera_view_timeline_not_capture_PTS"
        else:
            frame_pts = int(frame["pts"])
            value = frame["time_base"]
            base = Fraction(value["numerator"], value["denominator"]) if isinstance(value, dict) else rational(value)
            timeline = frame["timeline_provenance"]
        yield {"pixels": pixels, "pts": frame_pts, "time_base": base,
               "source_rate": rate or (rational(registered_rate) if registered_rate else None),
               "scene_id": frame.get("scene_id"), "reset": bool(frame.get("reset", False)),
               "reset_reasons": frame.get("reset_reasons", []), "source_scene_epoch": frame.get("scene_epoch"),
               "source_predecessor": frame.get("expected_predecessor"), "source_frame_id": frame.get("frame_id"),
               "source": {"path": str(path), "sha256": frame["sha256"],
                          "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
                          "provenance": frame["provenance"], "source_frame_id": frame.get("frame_id")},
               "guides": frame.get("guides", {}), "timeline_provenance": timeline,
               "decode": {"source": "registered_gsplat_RGB_PNG", "codec_conversion": "none"}}
    audit.emit("input_eos", source="sequence", natural_eos=True)


def media_frames(args, audit, report, snapshot, temporal_frames=None):
    import av
    import numpy as np
    identity, video_bytes = snapshot
    path = Path(identity["path"])
    provenance_path, provenance_sha = identity["provenance_path"], identity["provenance_sha256"]
    previous_source_frame = None
    with io.BytesIO(video_bytes) as source, av.open(source, mode="r") as container:
        audio = [{"index": stream.index, "codec": stream.codec_context.name} for stream in container.streams.audio]
        if audio and args.audio_policy != "drop":
            raise ValueError("Input contains audio; explicit --audio-policy drop required (audio preservation unsupported)")
        report["audio"] = {"policy": args.audio_policy, "input_streams": audio,
                           "result": "explicitly_dropped" if audio else "no_audio_stream"}
        if len(container.streams.video) != 1:
            raise ValueError("Exactly one video stream is supported by the initial adapter")
        stream = container.streams.video[0]
        codec = stream.codec_context
        if int(codec.color_trc) in (16, 18):
            raise ValueError("PQ/HLG video cannot enter the SDR profile")
        report["input_codec"] = {"name": codec.name, "format": codec.format.name if codec.format else None,
                                  "color_range": int(codec.color_range), "colorspace": int(codec.colorspace),
                                  "color_primaries": int(codec.color_primaries), "color_trc": int(codec.color_trc),
                                  "stream_time_base": ratio(stream.time_base), "caller_profile": args.color_profile,
                                  "unknown_metadata_policy": "explicit_caller_profile_required; HDR and YUV rejected",
                                  "container_sha256": identity["sha256"], "provenance_sha256": identity["provenance_sha256"],
                                  "decode_source": "same_hashed_NOFOLLOW_bounded_bytes_snapshot"}
        if args.seek_pts is not None:
            container.seek(args.seek_pts, stream=stream, backward=True, any_frame=False)
            audit.emit("seek", target_pts=args.seek_pts, time_base=ratio(stream.time_base),
                       state_policy="new_epoch_before_next_admission", decoder_policy="keyframe_seek_then_discard_preroll")
        for packet in container.demux(stream):
            packet_identity = {"pts": packet.pts, "dts": packet.dts,
                               "time_base": ratio(packet.time_base) if packet.time_base else None,
                               "decoder_flush_packet": packet.pts is None and packet.dts is None}
            for frame in packet.decode():
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Missing decoded PTS/time_base is not silently reconstructed")
                if int(frame.color_range) not in (0, 2) or int(frame.colorspace) not in (0, 2):
                    raise ValueError("Decoded video range/matrix conflicts with explicit RGB full-range profile")
                if not frame.format.is_rgb or any(component.bits > 8 for component in frame.format.components):
                    raise ValueError("Only 8-bit RGB media is admitted; YUV/HDR conversion requires a separate verified profile")
                if args.seek_pts is not None and Fraction(frame.pts) * frame.time_base < Fraction(args.seek_pts) * stream.time_base:
                    audit.emit("seek_preroll_discard", pts=frame.pts, time_base=ratio(frame.time_base))
                    continue
                if frame.format.name in ("rgba", "bgra", "argb", "abgr", "gbrap"):
                    rgba = frame.to_ndarray(format="rgba")
                    if not np.all(rgba[..., 3] == 255):
                        raise ValueError("Non-opaque media alpha has no verified compositing profile")
                pixels = frame.to_ndarray(format="rgb24")
                item = {"pixels": pixels, "pts": int(frame.pts), "time_base": Fraction(frame.time_base),
                       "source_rate": Fraction(stream.average_rate) if stream.average_rate else None,
                       "scene_id": "derived_mip360_media", "reset": False,
                       "source": {"path": str(path), "sha256": report["input_codec"]["container_sha256"],
                                  "provenance": str(provenance_path), "provenance_sha256": provenance_sha},
                       "guides": {name: {"source": "absent", "valid": False} for name in ("motion", "depth", "history")},
                       "timeline_provenance": "decoded_original_PTS_and_time_base_preserved",
                       "decode": {"packet": packet_identity, "picture_type": int(frame.pict_type),
                                  "key_frame": bool(frame.key_frame), "format": frame.format.name,
                                  "codec_conversion": "RGB_channel_layout_to_rgb24_only"}}
                if temporal_frames is not None:
                    previous_source_frame = temporal.join_frame(item, temporal_frames, previous_source_frame)
                yield item
        audit.emit("input_eos", source="video", decoder_flush_completed=True, natural_eos=True)


class Writer:
    def __init__(self, path, args, first, *, output_wh=None):
        import av
        self.path, self.args, self.expected = path, args, []
        self.base = first["time_base"]
        self.container = av.open(str(path), mode="w", format=args.container)
        self.nominal_rate = first.get("source_rate") or Fraction(25)
        self.stream = self.container.add_stream(args.codec, rate=self.nominal_rate)
        self.stream.width, self.stream.height = output_wh or (first["pixels"].shape[1], first["pixels"].shape[0])
        self.stream.pix_fmt = "bgr0" if args.codec == "ffv1" else "rgb24"
        self.pixel_format = self.stream.pix_fmt
        self.stream.time_base = self.base
        self.stream.codec_context.time_base = self.base
        self.stream.codec_context.color_range = 2
        self.stream.codec_context.colorspace = 0
        self.stream.codec_context.color_primaries = 1
        self.stream.codec_context.color_trc = 13
        self.reformatter = None
        if args.codec == "ffv1":
            self.stream.options = {"level": "3", "coder": "1", "context": "1", "slicecrc": "1"}
            self.reformatter = av.video.reformatter.VideoReformatter()
        elif args.codec == "libx264rgb":
            self.stream.options = {"crf": "0", "preset": "ultrafast", "bf": str(args.b_frames), "g": "12"}
        self.closed = False

    def write(self, pixels, item, timings=None):
        import av
        timestamp = item["pts"] * item["time_base"]
        scaled = timestamp / self.base
        if scaled.denominator != 1 and self.args.timestamp_policy == "exact":
            raise ValueError("Output encoder time base cannot exactly represent source PTS")
        start = time.perf_counter_ns()
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        if self.reformatter is not None:
            frame = self.reformatter.reformat(frame, format=self.pixel_format)
        frame.pts, frame.time_base = round(scaled), self.base
        frame.colorspace, frame.color_range = 0, 2
        encode_start = time.perf_counter_ns()
        packets = self.stream.encode(frame)
        mux_start = time.perf_counter_ns()
        for packet in packets:
            self.container.mux(packet)
        pixel_sha256 = digest_array(pixels)
        if timings is not None:
            timings.update(encoder_frame_prepare=(encode_start - start) / 1e6,
                           codec_encode=(mux_start - encode_start) / 1e6,
                           mux_and_rgb_sha=(time.perf_counter_ns() - mux_start) / 1e6)
        self.expected.append({"pts": item["pts"], "time_base": ratio(item["time_base"]),
                              "time": timestamp, "pixel_sha256": pixel_sha256})
        return pixel_sha256

    def close(self):
        if not self.closed:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
            self.container.close()
            self.closed = True

    def verify(self):
        import av
        observations = []
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index >= len(self.expected) or frame.pts is None or frame.time_base is None:
                    raise ValueError("Output decoder produced an extra/untimed frame")
                expected = self.expected[index]
                actual = Fraction(frame.pts) * frame.time_base
                difference = abs(actual - expected["time"])
                allowance = Fraction(0) if self.args.timestamp_policy == "exact" else Fraction(frame.time_base)
                if difference > allowance:
                    raise ValueError("Output PTS exceeds its preregistered exact/one-tick rounding contract")
                pixel_sha256 = digest_array(frame.to_ndarray(format="rgb24"))
                if pixel_sha256 != expected["pixel_sha256"]:
                    raise ValueError("Lossless codec round trip changed submitted RGB bytes")
                observations.append({"pts": frame.pts, "time_base": ratio(Fraction(frame.time_base)),
                                     "timestamp_error": ratio(difference), "picture_type": int(frame.pict_type),
                                     "pixel_sha256": pixel_sha256, "matches_encoded_input": True})
        if len(observations) != len(self.expected):
            raise ValueError("Encoder flush/EOS lost output frames")
        return observations


class OutputDelivery:
    """Keep at most one frame delivery outstanding and publish receipts in order."""

    def __init__(self, writer, audit, directory, budget_bytes, *, overlap=False):
        self.writer, self.audit = writer, audit
        self.directory, self.budget_bytes = directory, budget_bytes
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rw-video-encode") if overlap else None
        self.pending = None

    def _encode(self, pixels, timestamp, record, prepare):
        if prepare is not None:
            pixels = prepare()
        started = time.perf_counter_ns()
        digest = self.writer.write(pixels, timestamp, record.get("host_delivery_ms"))
        return digest, started, time.perf_counter_ns()

    def _publish(self, record, write):
        try:
            digest, started, finished = write()
        except Exception:
            if record["status"] == "FAILED_OUTPUT_PREPARATION":
                self.audit.emit("output_preparation_failed", **record)
            else:
                record["status"] = "FAILED_ENCODER"
                self.audit.emit("encoder_failed", **record)
            raise
        record.update(encoded_input_sha256=digest, output_written_host_ns=finished)
        timing = record.get("host_delivery_ms")
        if timing is not None:
            timing["encode_write_and_rgb_sha"] = (finished - started) / 1e6
        self.audit.emit("output_written", **record)
        written_bytes = sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())
        if written_bytes > self.budget_bytes:
            raise RuntimeError("Output quota reached; failure evidence retained")

    def drain(self):
        if self.pending is not None:
            record, future = self.pending
            self.pending = None
            self._publish(record, future.result)

    def write(self, pixels, item, record, prepare=None):
        timestamp = {"pts": item["pts"], "time_base": item["time_base"]}
        if self.executor is None:
            self._publish(record, lambda: self._encode(pixels, timestamp, record, prepare))
            return
        start = time.perf_counter_ns()
        try:
            self.drain()
        except Exception:
            record["status"] = "FAILED_ENCODER_PREDECESSOR"
            raise
        record["host_delivery_ms"]["wait_previous_encode"] = (time.perf_counter_ns() - start) / 1e6
        self.pending = (record, self.executor.submit(self._encode, pixels, timestamp, record, prepare))

    def close(self):
        try:
            self.drain()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
            self.writer.close()


def encode_pixels(raw, np):
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError("Native output is not finite SDR in [0,1]")
    scaled = raw * np.float32(255)
    whole = np.floor(scaled)
    return (whole + (scaled - whole >= np.float32(.5))).astype(np.uint8)


def timed_frames(iterator):
    while True:
        start = time.perf_counter_ns()
        try:
            frame = next(iterator)
        except StopIteration:
            return
        yield frame, (time.perf_counter_ns() - start) / 1e6


def frame_reset_reasons(item, index, previous_scene, use_temporal, seek_pts):
    reasons = [] if use_temporal else ["stateless_profile_reset_each_frame"]
    if use_temporal and index == 0:
        reasons.append("first_decoded_frame")
    if item["reset"]:
        reasons.extend(item.get("reset_reasons") or ["upstream_declared_reset"])
    if previous_scene is not None and item["scene_id"] != previous_scene:
        reasons.append("scene_change")
    if index == 0 and seek_pts is not None:
        reasons.append("seek")
    return reasons


def execute(args, report, context, sha256, within, audit, *, reader=None):
    input_start, input_cpu_start = time.perf_counter_ns(), time.process_time_ns()
    # Reject unsupported input semantics before importing codecs or creating a
    # CUDA runtime. Recheck when reading the actual source as well.
    storage = Path(context["storage_root"])
    use_sr = getattr(args, "feature", "nr") == "sr"
    use_temporal = use_sr or getattr(args, "sdr_temporal_head", False)
    operation = input_operation(args)
    temporal_frames = None
    snapshot = _source_snapshot(args.sequence_manifest if args.sequence_manifest else args.input_provenance, storage)
    source_path, source_raw, source_manifest, source_sha = snapshot
    if args.sequence_manifest:
        report["input_capability"] = sequence_profile(source_manifest, operation)
    else:
        start = time.perf_counter_ns()
        media_snapshot = read_media_snapshot(args, context, sha256, within, snapshot)
        report["media_identity_before_model"] = media_snapshot[0]
        report["media_snapshot_ms"] = (time.perf_counter_ns() - start) / 1e6
        report["input_capability"] = dict(media_snapshot[0]["input_capability"])
    report["input_capability"].update(provenance_path=str(source_path), provenance_sha256=source_sha)
    if read_bounded(source_path, storage / "runs", 16 * 1024 * 1024) != source_raw:
        raise ValueError("Input source changed after capability validation")
    if use_temporal:
        if args.sequence_manifest:
            report["temporal_source"] = temporal.validate_sequence(context["source_root"], source_manifest)
        else:
            temporal_frames, report["temporal_source"] = temporal.media_source(
                source_manifest, context["source_root"], storage)
        report["temporal_mode"] = sr.MODE if use_sr else temporal.MODE
    else:
        report["temporal_mode"] = ("stateless_reset_each_frame" if report["input_capability"]["stateless_sdk_compatible"]
                                   else "codec_only_unverified_input_semantics")
    import av
    import numpy as np
    storage = Path(context["storage_root"])
    report["environment"] = {"python": sys.version, "python_executable": sys.executable,
                              "pyav": av.__version__, "ffmpeg_libraries": av.library_versions,
                              "numpy": np.__version__, "pillow": importlib.metadata.version("Pillow")}
    source_iterator = (sequence_frames(args, context, sha256, within, audit, snapshot, reader=reader) if args.sequence_manifest
                       else media_frames(args, audit, report, media_snapshot, temporal_frames))
    runtime, session, writer, delivery = None, None, None, None
    overlap_write = use_sr and getattr(args, "sr_overlap_write", False)
    input_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rw-video-input") if overlap_write else None
    previous_time, previous_scene, epoch = None, None, 1
    levels = (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)
    decode_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
    try:
        try:
            first_item = input_executor.submit(next, source_iterator).result() if input_executor else next(source_iterator)
        except StopIteration:
            raise ValueError("No frames were supplied") from None
        first_decode_ms = (time.perf_counter_ns() - decode_start) / 1e6
        first_pixels = first_item["pixels"]
        if first_pixels.dtype != np.uint8 or first_pixels.ndim != 3 or first_pixels.shape[2] != 3:
            raise ValueError("Decoder did not produce registered RGB8 HWC")
        report["first_decode_before_model"] = {"pts": first_item["pts"], "time_base": ratio(first_item["time_base"]),
                                               "shape": list(first_pixels.shape), "dtype": "uint8",
                                               "source": first_item["decode"]["source"] if args.sequence_manifest else "video",
                                               "model_created": False}
        if args.command == "enhance":
            setup_start = time.perf_counter_ns()
            if read_bounded(source_path, storage / "runs", 16 * 1024 * 1024) != source_raw:
                raise ValueError("Input source changed before model creation")
            if use_sr:
                input_h, input_w = first_pixels.shape[:2]
                runtime, report["sdk"] = sr.open_runtime(args, context, first_pixels.shape, sha256, within, reader)
                session = runtime.session(epoch=1)
                report["sdk"]["active"] = runtime.memory_info()
                buffers = 2 if overlap_write else 1
                sr_outputs = tuple(np.empty((input_h * 2, input_w * 2, 4), np.float32) for _ in range(buffers))
                display = getattr(args, "sr_display", "auto")
                native_display = runtime.has_native_display_rgb8 and display != "numpy"
                if display == "native" and not native_display:
                    raise ValueError("The SR SDK library lacks native RGB8 display conversion")
                report["sdk"]["host_display_path"] = "native" if native_display else "numpy"
                if native_display:
                    packed_rgb = tuple(np.empty(sr_outputs[0].shape[:2] + (3,), np.uint8) for _ in range(buffers))
                report["sdk"]["host_rgba_delivery_buffer_bytes"] = sum(raw.nbytes for raw in sr_outputs)
            else:
                from renderweave.native_nr import NativeRuntime
                registration_path, plan_path, library_path = (within(path, storage) for path in (args.model, args.plan, args.sdk_library))
                registration = json.loads(registration_path.read_text())
                weights = within(Path(registration["parameters_path"]), Path(context["shared_assets_root"]) / "weights")
                if sha256(weights) != registration["parameters_sha256"]:
                    raise ValueError("Native parameter manifest or original weight file changed")
                previous_mode = os.environ.get("RW_NR_SDR_TEMPORAL_HEAD")
                if use_temporal:
                    from renderweave.h200_profile import configure_h200_fast
                    configure_h200_fast()
                    os.environ["RW_NR_SDR_TEMPORAL_HEAD"] = "1"
                try:
                    runtime = NativeRuntime(weights=str(weights), weights_sha256=registration["parameters_sha256"],
                                        parameter_manifest=str(registration_path), parameter_manifest_sha256=sha256(registration_path),
                                        plan=str(plan_path), plan_sha256=sha256(plan_path), library=str(library_path),
                                        experimental=True, device_ordinal=args.device_ordinal,
                                        device_budget_bytes=args.device_budget_mib << 20, host_budget_bytes=args.host_budget_mib << 20,
                                        max_sessions=1, max_inflight=1)
                finally:
                    if use_temporal:
                        if previous_mode is None:
                            os.environ.pop("RW_NR_SDR_TEMPORAL_HEAD", None)
                        else:
                            os.environ["RW_NR_SDR_TEMPORAL_HEAD"] = previous_mode
                report["sdk"] = {"initial": runtime.query(), "library_path": str(library_path),
                                  "library_sha256": sha256(library_path), "parameter_manifest_sha256": sha256(registration_path),
                                  "plan_sha256": sha256(plan_path), "temporal_mode": report["temporal_mode"]}
                if use_temporal and report["sdk"]["initial"].get("sdr_temporal_head_candidate") is not True:
                    raise ValueError("The SDK library does not support the SDR temporal head")
                session = runtime.session(epoch=1, temporal=use_temporal)
                input_h, input_w = session.height, session.width
            report["sdk_setup_ms"] = (time.perf_counter_ns() - setup_start) / 1e6
        source_frames = chain(((first_item, first_decode_ms),), timed_frames(source_iterator))
        input_future = None
        if use_sr:
            report["input_preparation"] = {"max_prefetched_frames": int(overlap_write),
                                           "logical_bytes_per_prepared_frame": input_h * input_w * (3 + 6 * 4)}

        def fetch_prepared(index, scene):
            start = time.perf_counter_ns()
            entry = next(source_frames, None)
            if entry is None or index >= args.max_frames:
                return entry, None, None
            item, _ = entry
            pixels = item["pixels"]
            if pixels.dtype != np.uint8 or pixels.shape != (input_h, input_w, 3):
                return entry, None, None
            reset = bool(frame_reset_reasons(item, index, scene, use_temporal, args.seek_pts))
            prepared = sr.prepare_input(item, levels, storage, args.sr_depth_range, reset, reader)
            return entry, prepared, [start, time.perf_counter_ns()]

        for index in count():
            prep_wait_ms = 0.0
            if input_future is None:
                entry, prepared, input_window = next(source_frames, None), None, None
            else:
                start = time.perf_counter_ns()
                entry, prepared, input_window = input_future.result()
                prep_wait_ms = (time.perf_counter_ns() - start) / 1e6
                input_future = None
            if entry is None:
                break
            item, decoded_ms = entry
            if index >= args.max_frames:
                audit.emit("input_limit", max_frames=args.max_frames, input_eos_not_assumed=True,
                           decoded_not_admitted_pts=item["pts"], time_base=ratio(item["time_base"]))
                break
            record = {"frame_id": index, "pts": item["pts"], "time_base": ratio(item["time_base"]),
                      "source": item["source"], "timeline_provenance": item["timeline_provenance"],
                      "decode": item["decode"], "guides": item["guides"], "status": "ARRIVED",
                      "arrived_host_ns": time.perf_counter_ns(), "color_profile": args.color_profile}
            report["requests"].append(record)
            audit.emit("arrival", **record)
            prepare = None
            if use_sr:
                # Deliveries are drained before the next frame is queued, so
                # an alternating buffer stays owned until its worker finishes.
                delivery_index = index % len(sr_outputs)
                sr_output = sr_outputs[delivery_index]
                record["host_delivery_ms"] = {"input_read_and_decode": decoded_ms}
                if input_executor:
                    record["host_delivery_ms"]["input_prefetch_wait"] = prep_wait_ms
                if input_window is not None:
                    record["input_prefetch_window_ns"] = input_window
            pixels = item["pixels"]
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError("Decoder did not produce registered RGB8 HWC")
            timestamp = item["pts"] * item["time_base"]
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError("Decoded/display PTS must strictly increase; duplicate/backward timestamps rejected")
            previous_time = timestamp
            record["decoded_input"] = {"shape": list(pixels.shape), "dtype": "uint8",
                                       "sha256": item.get("decoded_rgb_sha256") or digest_array(pixels)}
            if writer is None:
                storage_allowance = pixels.shape[0] * pixels.shape[1] * 3 * args.max_frames * (6 if args.save_raw else 2) + (1 << 20)
                if use_sr:
                    storage_allowance = sr_output.shape[0] * sr_output.shape[1] * args.max_frames * (22 if args.save_raw else 6) + (1 << 20)
                if storage_allowance > args.max_output_mib << 20:
                    record["status"] = "REJECTED_OUTPUT_BUDGET"
                    raise ValueError("Raw/codec storage allowance exceeds predeclared output quota")
                report["storage_admission"] = {"raw_and_codec_allowance_bytes": storage_allowance,
                                                "quota_bytes": args.max_output_mib << 20,
                                                "production_compression_bound_certified": False}
                stem = "enhanced" if args.command == "enhance" else "input"
                extension = "nut" if args.container == "nut" else "mkv"
                writer_options = {"output_wh": (2 * input_w, 2 * input_h)} if use_sr else {}
                writer = Writer(Path(args.output_dir) / f"{stem}.{extension}", args, item, **writer_options)
                delivery = OutputDelivery(writer, audit, Path(args.output_dir), args.max_output_mib << 20,
                                          overlap=overlap_write)
            if args.command == "enhance":
                if pixels.shape[:2] != (input_h, input_w):
                    record["status"] = "REJECTED_SHAPE"
                    raise ValueError("Decoded dimensions differ from the exact registered native plan; no adapter resize")
                reasons = frame_reset_reasons(item, index, previous_scene, use_temporal, args.seek_pts)
                if reasons:
                    epoch += 1
                    session.reset(epoch)
                record.update(epoch=epoch, reset_reasons=reasons,
                              input_normalization="exact_uint8_code_lookup_round_f64_div255_to_f32")
                if use_sr:
                    if prepared is None:
                        prepared = sr.prepare_input(item, levels, storage, args.sr_depth_range, bool(reasons), reader)
                    color = prepared.get("color")
                else:
                    color = np.ascontiguousarray(levels[pixels])
                    record["normalized_input_sha256"] = digest_array(color)
                job, result = None, None
                try:
                    if use_sr:
                        record["host_delivery_ms"].update(prepared["timing"])
                        for key in ("normalized_input_sha256", "sr_projection", "sr_guide_sha256"):
                            if key in prepared:
                                record[key] = prepared[key]
                        if "error" in prepared:
                            raise prepared["error"]
                        predecessor = 0 if reasons else index
                        record.update(source_frame_id=item["source_frame_id"], source_predecessor=item["source_predecessor"],
                                      source_scene_epoch=item["source_scene_epoch"], sdk_predecessor=predecessor,
                                      motion_submission="absent_at_reset" if reasons else "current_to_previous_pixels_y_down")
                        if input_executor:
                            input_future = input_executor.submit(fetch_prepared, index + 1, item["scene_id"])
                        sdk_start = time.perf_counter_ns()
                        job = session.submit(color, frame_id=index + 1, epoch=epoch, predecessor=predecessor,
                                             motion=prepared["motion"], depth=prepared["depth"])
                    elif use_temporal:
                        motion, validity = (None, None) if reasons else temporal.load_motion(
                            item["guides"]["motion"], storage, session.height, session.width)
                        record.update(source_frame_id=item["source_frame_id"], source_predecessor=item["source_predecessor"],
                                      source_scene_epoch=item["source_scene_epoch"], sdk_predecessor=None if reasons else index-1,
                                      motion_submission="absent_at_reset" if reasons else "current_to_previous_pixels_y_down")
                        job = session.submit_history(color, frame_id=index, motion_hwc2=motion, motion_validity=validity)
                    else:
                        job = session.submit(color, frame_id=index)
                    record["status"] = "ADMITTED"
                    record["admitted_host_ns"] = time.perf_counter_ns()
                    try:
                        if use_sr:
                            if job.wait(timeout_ms=args.wait_timeout_ms, output=sr_output) is None:
                                raise TimeoutError("SR job still running")
                        else:
                            result = job.wait(timeout_ms=args.wait_timeout_ms)
                    except TimeoutError:
                        record["wait_timed_out"] = True
                        audit.emit("wait_timeout", frame_id=index, job=job.info(), resources_reusable=False)
                        if use_sr:
                            job.wait(timeout_ms=-1, output=sr_output)
                        else:
                            result = job.wait(timeout_ms=-1)
                    info = job.info()
                    if not info["physical_complete" if use_sr else "physical_completion_confirmed"]:
                        raise RuntimeError("SDK result lacks physical completion confirmation")
                    record["sdk_job"] = info
                    if use_sr:
                        sdk_end = time.perf_counter_ns()
                        record["host_delivery_ms"]["sdk_submit_and_wait"] = (sdk_end - sdk_start) / 1e6
                        record["sdk_submit_and_wait_window_ns"] = [sdk_start, sdk_end]
                        prepare = partial(sr.prepare_delivery, sr_output, packed_rgb[delivery_index] if native_display else None,
                                          runtime.display_rgb8 if native_display else None,
                                          Path(args.output_dir), record, args.save_raw)
                    else:
                        record["sdk_result"] = result.metadata()
                        raw = np.asarray(result.tensor_view()).copy()
                        if not np.isfinite(raw).all():
                            np.save(Path(args.output_dir) / f"frame_{index:06d}.invalid_output.float32.npy", raw, allow_pickle=False)
                            raise ValueError("Native SDK returned non-finite color")
                        record["raw_output"] = {"dtype": "float32", "shape": list(raw.shape), "sha256": digest_array(raw),
                                                "min": float(raw.min()), "max": float(raw.max())}
                        if args.save_raw:
                            np.save(Path(args.output_dir) / f"frame_{index:06d}.native.float32.npy", raw, allow_pickle=False)
                        pixels = encode_pixels(raw, np)
                    record["status"] = "VALID_LATE_OUTPUT" if record.get("wait_timed_out") else "VALID_OUTPUT"
                except Exception:
                    if color is not None:
                        np.save(Path(args.output_dir) / f"frame_{index:06d}.failed_input.float32.npy", color, allow_pickle=False)
                    record["status"] = "FAILED" if job else "REJECTED_ADMISSION"
                    if job:
                        record["sdk_job"] = job.info()
                    raise
                finally:
                    if result:
                        result.close()
                    if job:
                        job.close()
            else:
                record["status"] = "ENCODE_ONLY_NO_MODEL_EXECUTION"
            previous_scene = item["scene_id"]
            delivery.write(None if prepare is not None else pixels, item, record, prepare)
        if writer is None:
            raise ValueError("No frames were supplied")
        delivery.close()
        cpu_elapsed = (time.process_time_ns() - cpu_start) / 1e9
        flushed = time.perf_counter_ns()
        elapsed = (flushed - decode_start) / 1e9
        report["decode_to_flush"] = {"elapsed_seconds": elapsed, "encoded_frames": len(writer.expected),
                                     "frames_per_second": len(writer.expected) / elapsed,
                                     "process_cpu_seconds": cpu_elapsed, "mean_process_cpu_cores": cpu_elapsed / elapsed}
        full_elapsed = (flushed - input_start) / 1e9
        full_cpu = (time.process_time_ns() - input_cpu_start) / 1e9
        report["source_to_flush"] = {"elapsed_seconds": full_elapsed, "encoded_frames": len(writer.expected),
                                     "frames_per_second": len(writer.expected) / full_elapsed,
                                     "process_cpu_seconds": full_cpu, "mean_process_cpu_cores": full_cpu / full_elapsed}
        audit.emit("encoder_eos", flushed=True, frames=len(writer.expected))
        report["output_timeline"] = writer.verify()
        report["output_video"] = str(writer.path)
        report["output_video_sha256"] = sha256(writer.path)
        report["derived_from_mip360_gsplat"] = True
        report["encoder"] = {"container": args.container, "codec": args.codec, "pixel_format": writer.pixel_format,
                              "format_conversion": "reused_explicit_bgr0" if writer.reformatter is not None else "codec_default",
                              "nominal_encoder_rate_hint": ratio(writer.nominal_rate),
                              "timestamp_policy": args.timestamp_policy, "original_PTS_timebase_in_audit_unchanged": True,
                              "rounding_bound": "zero" if args.timestamp_policy == "exact" else "one_actual_output_time_base_tick",
                              "lossless_RGB_roundtrip_verified": True, "encoder_flush_verified": True,
                              "max_async_frames": int(overlap_write)}
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        # Codec/filesystem failure must never prevent draining accepted native work.
        for name, close in (("input_prefetch", partial(input_executor.shutdown, wait=True, cancel_futures=True)
                             if input_executor else None),
                            ("input_decoder", source_iterator.close),
                            ("output_encoder", delivery.close if delivery else writer.close if writer else None),
                            ("session", session.close if session else None)):
            if close:
                try:
                    close()
                except Exception as error:
                    cleanup_errors.append({"resource": name, "error": str(error)})
        if runtime:
            try:
                report["sdk"]["before_close"] = runtime.memory_info() if use_sr else runtime.query()
            except Exception as error:
                cleanup_errors.append({"resource": "runtime_query", "error": str(error)})
            try:
                if use_sr:
                    runtime.close()
                else:
                    runtime.close(timeout_ms=-1)
                report["sdk"]["close_drained"] = True
            except Exception as error:
                cleanup_errors.append({"resource": "runtime_drain", "error": str(error)})
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            if not active_error:
                raise RuntimeError("Adapter cleanup/drain failed; see retained cleanup evidence")


def main(argv=None, *, execution=None, reader=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("encode", "enhance"))
    parser.add_argument("--feature", choices=("nr", "sr"), default="nr")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence-manifest", type=Path)
    source.add_argument("--input-video", type=Path)
    parser.add_argument("--input-provenance", type=Path)
    parser.add_argument("--generated-fps")
    parser.add_argument("--generated-start-pts", type=int, default=0)
    parser.add_argument("--vfr-steps", default="1")
    parser.add_argument("--color-profile", choices=("srgb-bt709-full-rgb8",), required=True)
    parser.add_argument("--audio-policy", choices=("reject", "drop"), default="reject")
    parser.add_argument("--seek-pts", type=int)
    parser.add_argument("--container", choices=("nut", "matroska"), default="nut")
    parser.add_argument("--timestamp-policy", choices=("exact", "round-within-one-tick"), default="exact")
    parser.add_argument("--codec", choices=("ffv1", "libx264rgb"), default="ffv1")
    parser.add_argument("--b-frames", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-output-mib", type=int, default=1024)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--sdk-library", type=Path)
    parser.add_argument("--sdr-temporal-head", action="store_true",
                        help="Use registered renderer motion and SDR HT session history")
    parser.add_argument("--sr-depth-range", nargs=2, type=float, metavar=("NEAR", "FAR"),
                        help="Fixed camera-space clip planes in renderer scene units for SR")
    parser.add_argument("--sr-display", choices=("auto", "native", "numpy"), default="auto",
                        help="SR host RGB8 conversion. Auto uses the native SDK when available.")
    parser.add_argument("--sr-overlap-write", action="store_true",
                        help="Overlap one queued SR input and one ordered output with SDK execution")
    parser.add_argument("--sr-original-fp16-backbone", action="store_true",
                        help="Use the original FP16 SR backbone arithmetic")
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--device-budget-mib", type=int, default=16384)
    parser.add_argument("--host-budget-mib", type=int, default=2048)
    parser.add_argument("--wait-timeout-ms", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.feature == "sr":
        if args.command != "enhance" or not args.sdk_library or args.generated_fps or args.sdr_temporal_head:
            parser.error("SR requires enhance, a native SR SDK library and the original renderer timeline")
        if not args.sr_depth_range or not all(map(math.isfinite, args.sr_depth_range)) or not 0 < args.sr_depth_range[0] < args.sr_depth_range[1]:
            parser.error("SR requires --sr-depth-range with finite 0 < NEAR < FAR")
    elif args.command == "enhance" and any(value is None for value in (args.model, args.plan, args.sdk_library)):
        parser.error("enhance requires --model --plan --sdk-library; encode never claims model execution")
    if args.input_video and (not args.input_provenance or args.generated_fps):
        parser.error("video input requires provenance and uses original decoded timestamps")
    if args.sdr_temporal_head and (args.command != "enhance" or args.generated_fps):
        parser.error("--sdr-temporal-head enhances an existing renderer timeline")
    if not 1 <= args.max_frames <= 12000 or not 0 <= args.b_frames <= 4 or args.max_output_mib <= 0 or args.generated_start_pts < 0:
        parser.error("Invalid explicit frame/output/B-frame budget")
    source_root, context, atomic_json, sha256, within = helpers(execution) if execution is not None else helpers()
    output = within(args.output_dir, Path(context["storage_root"]) / "runs")
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output
    report = {"schema_version": 1, "project_id": "renderweave",
              "task_id": "RW-20261118-668" if args.feature == "sr" else "RW-20261101-021",
              "run_id": args.run_id, "argv": list(argv) if argv is not None else sys.argv,
              "execution": context, "status": "RUNNING",
              "start_utc": datetime.now(timezone.utc).isoformat(), "requests": [],
              "native_temporal_qualified": False, "production_qualified": False,
              "temporal_mode": "not_yet_admitted", "audio": {"policy": args.audio_policy, "input_streams": []}}
    contract = {"command": args.command, "feature": args.feature, "color_profile": args.color_profile, "codec": args.codec,
                "container": args.container, "timestamp_policy": args.timestamp_policy,
                "generated_fps": args.generated_fps, "generated_vfr_steps": args.vfr_steps,
                "generated_start_pts": args.generated_start_pts,
                "audio_policy": args.audio_policy, "max_frames": args.max_frames,
                "max_output_bytes": args.max_output_mib << 20,
                "media_input_snapshot_maximum_bytes": min(args.max_output_mib, 1024) << 20,
                "unbound_legacy_media": "codec_only_unverified_never_model_input",
                "full_native_temporal_inference": False, "SDK_only_model_execution": True,
                "sdr_temporal_head": args.sdr_temporal_head,
                "sr_depth_range": args.sr_depth_range, "sr_overlap_write": args.sr_overlap_write,
                "sr_original_fp16_backbone": args.sr_original_fp16_backbone,
                "created_before_decode": True}
    atomic_json(output / "adapter_contract.json", contract)
    report["contract_sha256"] = sha256(output / "adapter_contract.json")
    audit = Audit(output / "audit.jsonl")
    code = 0
    try:
        execute(args, report, context, sha256, within, audit, reader=reader)
        report["status"] = "COMPLETED_EXPERIMENTAL_ADAPTER"
    except Exception as error:
        code = 1
        if report["requests"] and report["requests"][-1]["status"] in ("ARRIVED", "ADMITTED"):
            last = report["requests"][-1]
            last["status"] = "FAILED" if last["status"] == "ADMITTED" else "REJECTED_INPUT"
        report.update(status="FAILED", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        audit.emit("failure", error_type=type(error).__name__, message=str(error))
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        report["counts"] = {"arrived": len(report["requests"]),
                            "admitted": sum("admitted_host_ns" in item for item in report["requests"]),
                            "valid_outputs": sum(item["status"] == "VALID_OUTPUT" for item in report["requests"]),
                            "late_outputs": sum(item["status"] == "VALID_LATE_OUTPUT" for item in report["requests"]),
                            "wait_timeouts": sum(bool(item.get("wait_timed_out")) for item in report["requests"]),
                            "delivered_encoded_frames": sum("output_written_host_ns" in item for item in report["requests"]),
                            "failed_or_rejected": sum(item["status"].startswith(("FAILED", "REJECTED")) for item in report["requests"]),
                            "encode_only_frames": sum(item["status"] == "ENCODE_ONLY_NO_MODEL_EXECUTION" for item in report["requests"])}
        report.update(exit_code=code, end_utc=datetime.now(timezone.utc).isoformat())
        audit.close()
        report["audit_sha256"] = sha256(output / "audit.jsonl")
        atomic_json(output / "adapter.json", report)
    print(json.dumps({"status": report["status"], "evidence": str(output / "adapter.json"), "exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
