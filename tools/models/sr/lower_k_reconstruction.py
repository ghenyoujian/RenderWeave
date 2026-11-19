#!/usr/bin/env python3
"""Emit CUDA scalar math for the one registered K/LDR/2× reconstruction entry."""

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHA = "5133a026f4355aacd611ee4886a71848cc3e7c5cad4f283ef40ea27ecce4f1de"
REFERENCE_SHA = "1bd949170193f077f75f9c08f8a4a9a6abf21088efbd7795ad8cd463be3517ae"
ENTRY = "hiluma_engine_output_depthreg_mvlo_ldr_max_v2_rel"


def lower(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError("The SR K reconstruction source differs from the registered entry")
    text = source.decode()
    if re.findall(r"\.visible\s+\.entry\s+(\w+)", text) != [ENTRY]:
        raise ValueError("The registered reconstruction source must contain its single fixed entry")
    sys.path.insert(0, str(ROOT / "third_party/mlx_dlss"))
    from ptx_metal import compile_kernel

    reference = compile_kernel(text)
    if hashlib.sha256(reference.encode()).hexdigest() != REFERENCE_SHA:
        raise ValueError("The scalar reference expansion differs from the registered source")
    _, body = compile_kernel(text, fp32_add_rz="__fadd_rz", param_accessor="param_read", scalar_ops={
        "div.approx.ftz.f32": "source_div", "rcp.approx.ftz.f32": "source_rcp",
        "rsqrt.approx.ftz.f32": "source_rsqrt",
    }).split("\n// END HEADER\n", 1)
    for old, new in {
        "threadgroup uchar ": "__shared__ __align__(8) uchar ",
        "threadgroup_position_in_grid.": "blockIdx.",
        "thread_position_in_threadgroup.": "threadIdx.",
        "threadgroup_barrier(mem_flags::mem_threadgroup);": "__syncthreads();",
        "simd_shuffle_xor(": "shuffle_xor(",
        "const device ": "const ",
        "as_type<long>": "as_type<int64_t>",
    }.items():
        body = body.replace(old, new)
    body = re.sub(r"\+\(-(\d+)\)", r"-\1u", body)
    tokens = Counter(re.findall(r"\b\w+\b", body))
    body = re.sub(r"(?m)^(?:uint|ulong) (\w+)=0;\n",
                  lambda match: "" if tokens[match[1]] == 1 else match[0], body)
    return "// Generated scalar body for SR K 310.7.0.\n" + body + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(lower(args.source.read_bytes()))


if __name__ == "__main__":
    main()
