"""Reject input semantics the installed stateless NR media path cannot consume."""
from __future__ import annotations

STATELESS = "stateless_reset_each_frame"
CONTRACT = "rw.video.input_capability.v1"


def sequence_profile(manifest, command):
    if command not in ("encode", "enhance"):
        raise ValueError("Unknown media operation")
    if not isinstance(manifest, dict):
        raise ValueError("A sequence manifest object is required")
    if manifest.get("renderer") != "gsplat" or manifest.get("status") != "COMPLETED":
        raise ValueError("Only a completed registered gsplat sequence is accepted")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("A nonempty registered sequence is required")
    unsupported = "sequence_contract" in manifest
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("A registered frame object is required")
        mode = frame.get("temporal_mode")
        if mode not in (None, STATELESS):
            unsupported = True
        declaration = frame.get("backend_consumption")
        if declaration is not None:
            if not isinstance(declaration, dict):
                raise ValueError("Invalid frame capability declaration")
            if declaration.get("stateless_sdk_compatible") is not True:
                unsupported = True
        if frame.get("source_frame_contract") or frame.get("expected_predecessor") is not None:
            unsupported = True
    if command == "enhance" and unsupported:
        raise ValueError("This sequence requires continuous or unverified input semantics; the current stateless NR SDK cannot consume it")
    return {"contract": CONTRACT, "source_kind": "registered_gsplat_sequence",
            "stateless_sdk_compatible": not unsupported,
            "model_input_profile": STATELESS if not unsupported else "unverified_for_current_sdk",
            "guides_preserved_in_video": False,
            "native_temporal_qualified": False}


def media_profile(provenance, command):
    if command not in ("encode", "enhance") or not isinstance(provenance, dict):
        raise ValueError("Unknown media operation or provenance")
    declaration = provenance.get("input_capability")
    if declaration is None:
        # An explicit mode alone does not identify an original source.  Old
        # receipts remain available to codecs, but cannot qualify a model
        # input, including when wrapped by a newer derivation receipt.
        compatible = False
    else:
        compatible = (isinstance(declaration, dict)
                      and declaration.get("contract") == CONTRACT
                      and declaration.get("stateless_sdk_compatible") is True
                      and declaration.get("model_input_profile") == STATELESS)
    if (provenance.get("temporal_mode") not in (None, STATELESS)
            or "sequence_contract" in provenance or provenance.get("source_frame_contract")):
        compatible = False
    if command == "enhance" and not compatible:
        raise ValueError("Media provenance requires unverified or continuous semantics; encoding must not silently convert it into stateless NR input")
    return {"contract": CONTRACT, "source_kind": "derived_registered_media",
            "stateless_sdk_compatible": compatible,
            "model_input_profile": STATELESS if compatible else "unverified_for_current_sdk",
            "guides_preserved_in_video": False,
            "native_temporal_qualified": False}
