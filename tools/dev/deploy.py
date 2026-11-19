#!/usr/bin/env python3
"""Detect CUDA devices and build the native RenderWeave SDK on Windows or Linux."""
from __future__ import annotations

import argparse
import ctypes as c
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = sys.platform == "win32"
CU_COMPUTE_CAPABILITY_MAJOR = 75
CU_COMPUTE_CAPABILITY_MINOR = 76


def detect_cuda():
    if WINDOWS:
        driver = c.WinDLL("nvcuda.dll")
    else:
        try:
            driver = c.CDLL("libcuda.so.1")
        except OSError:
            driver = c.CDLL("/usr/lib/wsl/lib/libcuda.so.1")

    def checked(name, *args):
        status = getattr(driver, name)(*args)
        if status:
            error = c.c_char_p()
            driver.cuGetErrorString(status, c.byref(error))
            detail = error.value.decode() if error.value else str(status)
            raise RuntimeError(f"{name} failed ({detail})")

    checked("cuInit", 0)
    version, count = c.c_int(), c.c_int()
    checked("cuDriverGetVersion", c.byref(version))
    checked("cuDeviceGetCount", c.byref(count))
    devices = []
    for ordinal in range(count.value):
        device, major, minor = c.c_int(), c.c_int(), c.c_int()
        name, raw_uuid = c.create_string_buffer(256), (c.c_ubyte * 16)()
        memory = c.c_size_t()
        checked("cuDeviceGet", c.byref(device), ordinal)
        checked("cuDeviceGetName", name, len(name), device)
        checked("cuDeviceGetAttribute", c.byref(major), CU_COMPUTE_CAPABILITY_MAJOR, device)
        checked("cuDeviceGetAttribute", c.byref(minor), CU_COMPUTE_CAPABILITY_MINOR, device)
        checked("cuDeviceTotalMem_v2", c.byref(memory), device)
        uuid_api = "cuDeviceGetUuid_v2" if hasattr(driver, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
        checked(uuid_api, c.byref(raw_uuid), device)
        devices.append({
            "ordinal": ordinal, "name": name.value.decode("utf-8", "replace"),
            "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid))),
            "compute_capability": f"{major.value}.{minor.value}",
            "architecture": f"{major.value}{minor.value}", "total_memory_bytes": memory.value,
        })
    return {"driver_api_version": version.value, "devices": devices}


def architecture(text):
    value = text.removeprefix("sm_").replace(".", "")
    if not re.fullmatch(r"[1-9][0-9]{1,2}", value):
        raise argparse.ArgumentTypeError("use a CUDA compute capability such as 8.6, 86 or sm_86")
    return value


def windows_tools(env, work):
    installer = Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer"
    vswhere = installer / "vswhere.exe"
    if shutil.which("cl", path=env["PATH"]):
        return
    if not vswhere.is_file():
        raise RuntimeError("Install Visual Studio C++ Build Tools or run from its x64 developer shell")
    installations = subprocess.check_output(
        [vswhere, "-all", "-products", "*", "-property", "installationPath"], encoding="mbcs", env=env
    ).splitlines()
    for entry in installations:
        vs = Path(entry.strip())
        vcvars = vs / "VC/Auxiliary/Build/vcvars64.bat"
        if not vcvars.is_file():
            continue
        batch = work / "load-msvc.cmd"
        batch.write_text(f'@call "{vcvars}" >nul\n@if errorlevel 1 exit /b 1\n@set\n', encoding="mbcs")
        values = subprocess.check_output(
            [env.get("ComSpec", "cmd.exe"), "/d", "/c", batch.name], cwd=work, env=env, encoding="mbcs"
        )
        env.update(line.split("=", 1) for line in values.splitlines() if "=" in line and not line.startswith("="))
        ninja = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"
        env["PATH"] = str(ninja) + os.pathsep + env["PATH"]
        if shutil.which("cl", path=env["PATH"]):
            return
    raise RuntimeError("No installed x64 Visual Studio C++ compiler was found")


def add_windows_dependencies(install, roots, *, nr, sr):
    patterns = ["cudart64_*.dll"]
    if nr:
        patterns += ["cublas64_*.dll", "cublasLt64_*.dll"]
    if nr or sr:
        patterns.append("libcrypto-*.dll")
    paths = [directory for root in roots if root for directory in (root / "bin", root / "Library/bin", root)]
    paths += [Path(directory) for directory in os.get_exec_path()]
    for pattern in patterns:
        candidate = next((dll for path in paths for dll in sorted(path.glob(pattern)) if dll.is_file()), None)
        if candidate is None:
            raise FileNotFoundError(f"Cannot bundle {pattern}. Set --cuda-root and --openssl-root to the build toolkits.")
        shutil.copy2(candidate, install / "bin" / candidate.name)


def cuda_toolkit(explicit, env):
    if explicit:
        return explicit
    if env.get("CUDA_PATH"):
        return Path(env["CUDA_PATH"])
    nvcc = shutil.which("nvcc", path=env["PATH"])
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    if WINDOWS:
        prefixes = sorted((ROOT / "runtime/toolchains").glob("win-cuda-*/prefix"), reverse=True)
        return next((p for p in prefixes if (p / "bin/nvcc.exe").is_file()), None)
    return None


def run(args):
    platform = "windows" if WINDOWS else "linux"
    try:
        detected = detect_cuda()
    except (OSError, RuntimeError) as error:
        detected = {"devices": [], "error": str(error)}
    report = {"project_id": "renderweave", "platform": platform, **detected}
    if args.detect:
        print(json.dumps(report, indent=2))
        return 0 if detected["devices"] else 1

    selected = detected["devices"]
    if args.device is not None:
        selected = [device for device in selected if device["ordinal"] == args.device]
        if not selected:
            raise ValueError(f"CUDA ordinal {args.device} is not visible")
    architectures = sorted(set(args.architecture or [device["architecture"] for device in selected]), key=int)
    if not architectures:
        raise ValueError("No CUDA device detected. Use --architecture for an offline build.")
    if args.device is not None and selected[0]["architecture"] not in architectures:
        raise ValueError("The selected device architecture is missing from the build")
    selected = [device for device in selected if device["architecture"] in architectures]
    nr = not args.probes_only and not args.sr_only
    source = args.sr_reconstruction.resolve(strict=True) if args.sr_reconstruction else None
    if args.sr_only and not source:
        raise ValueError("--sr-only requires --sr-reconstruction")
    if args.probes_only and source:
        raise ValueError("--probes-only cannot include SR reconstruction")
    hopper = architectures == ["90"] and not args.portable
    variant = ("-nr" if nr else "") + ("-sr" if source else "") or "-probes"
    output = (args.output or ROOT / "build/deploy" / (platform + "-sm" + "_".join(architectures) +
              ("a" if hopper else "") + variant)).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("Build and installation paths must be inside the RenderWeave project")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "tmp"
    temporary.mkdir(exist_ok=True)
    env = dict(os.environ, TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
               CUDA_CACHE_PATH=str(output / "cuda-cache"), PYTHONDONTWRITEBYTECODE="1")
    if WINDOWS:
        windows_tools(env, output)
    cuda = cuda_toolkit(args.cuda_root, env)
    openssl = args.openssl_root or (Path(env["OPENSSL_ROOT_DIR"]) if env.get("OPENSSL_ROOT_DIR") else None)
    if WINDOWS and openssl is None:
        private_openssl = ROOT / "runtime/toolchains/win-openssl/prefix/Library"
        if private_openssl.is_dir():
            openssl = private_openssl
    if cuda:
        env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
    cmake = shutil.which("cmake", path=env["PATH"])
    if not cmake and WINDOWS:
        candidate = Path(env.get("ProgramFiles", r"C:\Program Files")) / "CMake/bin/cmake.exe"
        if candidate.is_file():
            cmake = str(candidate)
    if not cmake:
        raise RuntimeError("CMake 3.25 or newer is needed")
    cache = [
        "-DRW_ENABLE_CUDA=ON", "-DRW_BUILD_TESTS=OFF", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(value + "-real" for value in architectures),
        f"-DRW_NATIVE_NR_HOPPER_WGMMA={'ON' if hopper else 'OFF'}",
        f"-DRW_NATIVE_NR_PORTABLE_CUBLAS={'OFF' if hopper and not WINDOWS else 'ON'}",
        f"-DRW_BUILD_NATIVE_NR_EXPERIMENTAL={'ON' if nr else 'OFF'}",
        f"-DRW_BUILD_NATIVE_NR_SDK_EXPERIMENTAL={'ON' if nr else 'OFF'}",
        f"-DRW_BUILD_SR_SOURCE_BACKBONE={'ON' if source else 'OFF'}",
    ]
    if cuda:
        nvcc = cuda / "bin" / ("nvcc.exe" if WINDOWS else "nvcc")
        supported = subprocess.check_output([nvcc, "--list-gpu-code"], text=True, env=env).splitlines()
        missing = [arch for arch in architectures if f"sm_{arch}" not in supported]
        if missing:
            raise ValueError(f"{nvcc} cannot compile {', '.join('SM' + arch for arch in missing)}")
        cache += [f"-DCUDAToolkit_ROOT={cuda}", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
    if openssl:
        cache.append(f"-DOPENSSL_ROOT_DIR={openssl}")
    if source:
        cache += [f"-DRW_SR_K_RECONSTRUCTION_PTX={source}", f"-DPython3_EXECUTABLE={sys.executable}"]
    build, install = output / "build", output / "install"
    targets = ["rw-doctor", "rw-cuda-numeric-probe", "rw_gsplat_adapter"]
    if nr:
        targets.append("rw_nr_sdk")
    if source:
        targets.append("rw_sr_sdk")
        if WINDOWS:
            targets.append("rw-d3d12-sr-replay")
    configure = [cmake, "-S", ROOT, "-B", build]
    if WINDOWS:
        configure += ["-G", "Ninja"]
    subprocess.run([*configure, *cache], check=True, env=env)
    subprocess.run([cmake, "--build", build, "--config", "Release", "--target", *targets,
                    "--parallel", str(args.jobs)], check=True, env=env)
    subprocess.run([cmake, "--install", build, "--config", "Release", "--prefix", install], check=True, env=env)
    if WINDOWS:
        add_windows_dependencies(install, [cuda, openssl], nr=nr, sr=bool(source))
    report.update(cuda_architectures=architectures, hopper_wgmma=hopper and nr,
                  targets=targets, install=str(install), selected_devices=selected)
    (output / "deployment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect", action="store_true", help="print driver-visible CUDA devices")
    parser.add_argument("--device", type=int, help="build for a specific CUDA-visible ordinal")
    parser.add_argument("--architecture", action="append", type=architecture, help="override build targets (repeatable)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probes-only", action="store_true")
    mode.add_argument("--sr-only", action="store_true")
    parser.add_argument("--sr-reconstruction", type=Path, help="include the SR SDK using the original extracted PTX")
    parser.add_argument("--portable", action="store_true", help="use the portable lowering even on SM90")
    parser.add_argument("--cuda-root", type=lambda path: Path(path).resolve())
    parser.add_argument("--openssl-root", type=lambda path: Path(path).resolve())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
