"""CPU-only SR postprocess admission shared with the public C ABI.

This module validates history/launch identity before a future CUDA call. It
never allocates device memory, launches a kernel, or commits history.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


class PostprocessAdmissionError(ValueError):
    pass


@dataclass(frozen=True)
class HistoryBinding:
    resource_id: int
    owner_id: int
    session_id: int
    epoch: int
    frame: int
    codec_fingerprint: int
    ready_event: int
    completion_event: int
    valid: bool = True
    authenticated: bool = True


@dataclass(frozen=True)
class LaunchBinding:
    correlation: int
    graph_generation: int
    module_id: int = 0
    function_id: int = 0
    authenticated: bool = True


@dataclass(frozen=True)
class CatmullRomSubmission:
    session_id: int
    epoch: int
    frame_id: int
    expected_predecessor: int
    output_owner: int
    width: int
    height: int
    x: float
    y: float
    history: HistoryBinding
    launch: LaunchBinding
    output_valid: bool = True


def admit_catmull_rom(request: CatmullRomSubmission) -> None:
    """Raise ``PostprocessAdmissionError`` unless the request is fully bound."""
    if not isinstance(request, CatmullRomSubmission):
        raise TypeError("request must be CatmullRomSubmission")
    if request.session_id <= 0:
        raise PostprocessAdmissionError("missing session")
    if request.epoch <= 0:
        raise PostprocessAdmissionError("missing epoch")
    if request.width <= 0 or request.height <= 0:
        raise PostprocessAdmissionError("invalid geometry")
    if not math.isfinite(request.x) or not math.isfinite(request.y):
        raise PostprocessAdmissionError("nonfinite coordinate")
    history = request.history
    if not history.valid or history.owner_id <= 0:
        raise PostprocessAdmissionError("missing history")
    if history.resource_id <= 0 or not history.authenticated:
        raise PostprocessAdmissionError("missing authenticated history resource")
    if history.ready_event <= 0 or history.completion_event <= 0:
        raise PostprocessAdmissionError("missing history events")
    if history.codec_fingerprint <= 0:
        raise PostprocessAdmissionError("missing history codec")
    if history.session_id != request.session_id or history.epoch != request.epoch:
        raise PostprocessAdmissionError("history owner/session/epoch mismatch")
    if request.expected_predecessor <= 0 or history.frame != request.expected_predecessor:
        raise PostprocessAdmissionError("stale history predecessor")
    if request.frame_id <= request.expected_predecessor:
        raise PostprocessAdmissionError("frame is not after predecessor")
    launch = request.launch
    if not launch.authenticated or launch.correlation <= 0 or launch.graph_generation <= 0:
        raise PostprocessAdmissionError("missing authenticated launch join")
    if not request.output_valid or request.output_owner <= 0:
        raise PostprocessAdmissionError("missing output owner")
    if request.output_owner == history.owner_id:
        raise PostprocessAdmissionError("history/output owner alias")


__all__ = [
    "CatmullRomSubmission",
    "HistoryBinding",
    "LaunchBinding",
    "PostprocessAdmissionError",
    "admit_catmull_rom",
]
