"""Read-only MIP360/COLMAP input contracts; no training or image resampling."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
from typing import BinaryIO

SOURCE_ROOT = Path(__file__).resolve().parents[2]
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_input(path: Path) -> Path:
    config = json.loads((SOURCE_ROOT / "config/project.json").read_text())
    shared = Path(config["storage_policy"]["shared_assets_root"])
    if shared.resolve(strict=True) != shared:
        raise ValueError("Configured shared asset root changed realpath")
    if not path.is_absolute():
        raise ValueError("Public input must be an explicit absolute path")
    real = path.resolve(strict=True)
    if not real.is_relative_to(shared.resolve(strict=True)):
        raise ValueError("Input realpath escapes configured shared asset root")
    return real


def execution_context(output: Path) -> tuple[Path, dict]:
    """All experiments run in Slurm against a fixed published local commit."""
    config = json.loads((SOURCE_ROOT / "config/project.json").read_text())
    storage = Path(config["remote_root"])
    if not os.environ.get("SLURM_JOB_ID", "").isdecimal():
        raise ValueError("Run the input registration/render experiment inside Slurm")
    if SOURCE_ROOT.parent != storage / "releases":
        raise ValueError("Execute a published immutable project release")
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if SOURCE_ROOT.name != commit:
        raise ValueError("Release path is not bound to its source commit")
    if subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Release source is dirty; repair in the local repository")
    if storage.resolve() != storage:
        raise ValueError("Project storage root changed realpath")
    if not output.is_absolute():
        raise ValueError("Output must be an explicit absolute project path")
    if not output.resolve().is_relative_to(storage / "runs"):
        raise ValueError("Derived input manifests and renders must stay in project runs/")
    evidence = {
        "project_id": "renderweave", "source_commit": commit,
        "source_dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "source_root": str(SOURCE_ROOT), "storage_root": str(storage),
        "command_argv": list(__import__("sys").argv),
    }
    return storage, evidence


def write_new_json(path: Path, value: dict) -> None:
    # O_EXCL preserves earlier experiment evidence, including failures.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated COLMAP/image metadata")
    return value


def unpack(stream: BinaryIO, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, read_exact(stream, struct.calcsize("<" + fmt)))


def image_dimensions(path: Path) -> tuple[int, int, int]:
    """Inspect encoded dimensions without decoding/copying the source image."""
    with path.open("rb") as stream:
        signature = stream.read(8)
        if signature == b"\x89PNG\r\n\x1a\n":
            size, kind = struct.unpack(">I4s", read_exact(stream, 8))
            if kind != b"IHDR" or size != 13:
                raise ValueError("Invalid PNG IHDR")
            width, height, bits, color_type = struct.unpack(">IIBB", read_exact(stream, 10))
            if bits != 8 or color_type != 2:
                raise ValueError("Input registration requires 8-bit RGB PNG")
            if not width or not height:
                raise ValueError("Invalid image dimensions")
            return width, height, 3
        if signature[:2] != b"\xff\xd8":
            raise ValueError("Only RGB JPEG/PNG inputs are supported")
        stream.seek(2)
        while True:
            marker = read_exact(stream, 1)
            if marker != b"\xff":
                raise ValueError("Malformed JPEG marker")
            code = read_exact(stream, 1)[0]
            while code == 0xFF:
                code = read_exact(stream, 1)[0]
            if code in {0xD9, 0xDA}:
                raise ValueError("JPEG has no supported frame header")
            if code == 0x01 or 0xD0 <= code <= 0xD8:
                continue
            length = struct.unpack(">H", read_exact(stream, 2))[0]
            if length < 2:
                raise ValueError("Invalid JPEG segment length")
            if code in {0xC0, 0xC1, 0xC2}:
                if length < 8:
                    raise ValueError("Invalid JPEG frame header")
                bits, height, width, channels = struct.unpack(">BHHB", read_exact(stream, 6))
                if bits != 8 or channels != 3:
                    raise ValueError("Input registration requires 8-bit three-component JPEG")
                if not width or not height:
                    raise ValueError("Invalid image dimensions")
                return width, height, channels
            stream.seek(length - 2, os.SEEK_CUR)


def read_cameras(path: Path) -> dict[int, dict]:
    result = {}
    with path.open("rb") as stream:
        count, = unpack(stream, "Q")
        if count > 100000:
            raise ValueError("Unbounded COLMAP camera inventory")
        for _ in range(count):
            camera_id, model_id, width, height = unpack(stream, "iiQQ")
            if model_id not in CAMERA_MODELS or camera_id in result:
                raise ValueError("Unknown/duplicate COLMAP camera")
            model, num_params = CAMERA_MODELS[model_id]
            params = unpack(stream, "d" * num_params)
            if not width or not height or not all(math.isfinite(x) for x in params):
                raise ValueError("Invalid COLMAP camera values")
            result[camera_id] = {
                "camera_id": camera_id, "model": model,
                "width": width, "height": height, "params": list(params),
            }
        if stream.read(1):
            raise ValueError("Unexpected trailing COLMAP camera data")
    return result


def read_selected_images(path: Path, names: set[str]) -> dict[str, dict]:
    result = {}
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        count, = unpack(stream, "Q")
        if count > 1000000:
            raise ValueError("Unbounded COLMAP image inventory")
        for _ in range(count):
            values = unpack(stream, "i" + "d" * 7 + "i")
            name_bytes = bytearray()
            while True:
                char = read_exact(stream, 1)
                if char == b"\x00":
                    break
                name_bytes.extend(char)
                if len(name_bytes) > 4096:
                    raise ValueError("Unbounded COLMAP image name")
            name = name_bytes.decode("utf-8")
            points, = unpack(stream, "Q")
            skip_bytes = points * 24  # x/y doubles and point3D_id int64.
            if stream.tell() + skip_bytes > file_size:
                raise ValueError("COLMAP points extend past the image file")
            stream.seek(skip_bytes, os.SEEK_CUR)
            if name in names:
                if name in result or not all(math.isfinite(x) for x in values[1:8]):
                    raise ValueError("Duplicate/invalid COLMAP image pose")
                result[name] = {
                    "image_id": values[0], "qvec_wxyz": list(values[1:5]),
                    "tvec": list(values[5:8]), "camera_id": values[8],
                    "name": name, "transform": "world_to_camera",
                    "axes": "x_right_y_down_z_forward", "length_units": "colmap_scene_units",
                }
            if result.keys() == names:
                break
    return result


def pinhole_camera(camera: dict, width: int, height: int) -> list[list[float]]:
    params, model = camera["params"], camera["model"]
    if model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model == "SIMPLE_PINHOLE":
        fx, cx, cy = params
        fy = fx
    else:
        raise ValueError("Undistorted PINHOLE/SIMPLE_PINHOLE camera required; no implicit undistortion")
    sx, sy = width / camera["width"], height / camera["height"]
    return [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]]


def world_to_camera(pose: dict) -> list[list[float]]:
    w, x, y, z = pose["qvec_wxyz"]
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
        raise ValueError("COLMAP pose quaternion is not unit length")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    tx, ty, tz = pose["tvec"]
    return [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), tx],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), ty],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), tz],
        [0.0, 0.0, 0.0, 1.0],
    ]
