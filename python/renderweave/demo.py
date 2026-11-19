"""Run the smallest end-to-end RenderWeave CPU contract demo."""
from __future__ import annotations

import argparse
import json

from . import Runtime


def run(library: str | None = None) -> dict:
    """Exercise the public Python entry point without executing a model."""
    with Runtime(library=library) as runtime:
        codes = runtime.encode_e4m3([0.0, 1.0, -1.0, 448.0])
        with runtime.temporal_session(session_id=1, epoch=1) as session:
            snapshot = session.snapshot()
        return {
            "project_id": "renderweave",
            "demo": "cpu_contracts",
            "library_path": str(runtime.library_path),
            "capabilities": runtime.capabilities(),
            "e4m3_codes_hex": codes.hex(),
            "e4m3_roundtrip": runtime.decode_e4m3(codes),
            "memory_required_bytes": runtime.memory_required(
                budget_bytes=100, shared_weights_bytes=10,
                bytes_per_slot=5, slot_count=4,
                bytes_per_session=2, session_count=8,
            ),
            "temporal_snapshot": {
                "session_id": snapshot.session_id,
                "epoch": snapshot.epoch,
                "valid": snapshot.valid,
            },
            "native_model_execution": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="explicit librw_core path")
    args = parser.parse_args()
    print(json.dumps(run(args.library), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
