"""Explicit host temporal packet for NR adapter admission.

The packet owns metadata and lineage, while payload ownership remains with the
caller.  It is intentionally separate from the ABI2 CPU contract and from the
experimental previous-colour CUDA entry: accepting a packet does not claim
that the current native graph consumes motion/depth/history.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal
from threading import RLock


Presence = Literal["present", "absent"]


class HistoryCommitError(ValueError):
    """History owner/epoch/physical-completion contract violation."""


@dataclass(frozen=True)
class HistoryState:
    session_id: int
    epoch: int
    owner_id: int
    frame_id: int | None


@dataclass(frozen=True)
class GuidePlane:
    presence: Presence
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    units: str | None = None
    source: str | None = None
    sha256: str | None = None
    validity_mask_sha256: str | None = None
    direction: str | None = None
    lineage_frame_id: int | None = None
    owner_id: int = 0
    owner_session_id: int = 0
    owner_epoch: int = 0
    owner_frame_id: int = 0

    def manifest(self) -> dict[str, Any]:
        if self.presence == "absent":
            return {"presence": "absent", "valid": False, "sha256": None}
        if self.presence != "present":
            raise ValueError("guide presence must be present or absent")
        if not self.shape or any(type(v) is not int or v <= 0 for v in self.shape):
            raise ValueError("present guide shape must be positive integers")
        if not self.dtype or not self.units or not self.source:
            raise ValueError("present guide dtype/units/source are required")
        for name, value in (("sha256", self.sha256),
                            ("validity_mask_sha256", self.validity_mask_sha256)):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA256 digest")
            int(value, 16)
        if self.owner_id <= 0 or self.owner_session_id <= 0:
            raise ValueError("present guide owner identity is required")
        result: dict[str, Any] = {
            "presence": "present", "valid": True, "shape": list(self.shape),
            "dtype": self.dtype, "units": self.units, "source": self.source,
            "sha256": self.sha256, "validity_mask_sha256": self.validity_mask_sha256,
            "owner_id": self.owner_id, "owner_session_id": self.owner_session_id,
            "owner_epoch": self.owner_epoch, "owner_frame_id": self.owner_frame_id,
        }
        if self.direction is not None:
            result["direction"] = self.direction
        if self.lineage_frame_id is not None:
            result["lineage_frame_id"] = self.lineage_frame_id
        return result


@dataclass(frozen=True)
class TemporalPacket:
    session_id: int
    sequence_id: int
    frame_id: int
    epoch: int
    expected_predecessor: int | None
    pts: int
    time_base_num: int
    time_base_den: int
    reset: bool
    color: GuidePlane
    depth: GuidePlane
    motion: GuidePlane
    history: GuidePlane | None
    previous_color: Any = None
    scene_epoch: int = 0
    jitter_explicitly_disabled: bool = True
    exposure_explicit_no_transform: bool = True

    def validate(self, *, width: int, height: int) -> None:
        if self.session_id <= 0 or self.sequence_id <= 0 or self.epoch <= 0:
            raise ValueError("session/sequence/epoch identity is required")
        if self.frame_id < 0 or self.pts < 0 or self.time_base_num <= 0 or self.time_base_den <= 0:
            raise ValueError("frame timeline is invalid")
        if self.reset != (self.frame_id == 0 or self.expected_predecessor is None):
            raise ValueError("reset and predecessor presence disagree")
        self.color.manifest()
        if self.color.presence != "present" or self.color.shape not in ((height, width, 3), (1, height, width, 3)):
            raise ValueError("color must be present HWC RGB at the packet geometry")
        if self.reset:
            if self.motion.presence != "absent":
                raise ValueError("reset packet must omit motion, not provide zero motion")
            if self.history is not None and self.history.presence != "absent":
                raise ValueError("reset packet must omit history")
            if self.previous_color is not None:
                raise ValueError("reset packet cannot carry previous colour")
            return
        if self.expected_predecessor != self.frame_id - 1:
            raise ValueError("packet must reference the immediate predecessor")
        if self.previous_color is None:
            raise ValueError("non-reset packet needs retained previous colour")
        if self.motion.presence != "present":
            raise ValueError("non-reset packet needs an explicit motion guide")
        if self.motion.direction != "current_to_previous":
            raise ValueError("motion direction must be current_to_previous")
        if self.motion.units not in ("pixels", "uv"):
            raise ValueError("motion units must be explicit pixels or uv")
        if self.motion.shape not in ((height, width, 2), (1, height, width, 2)):
            raise ValueError("motion geometry differs from color")
        if self.depth.presence == "present" and self.depth.shape not in ((height, width), (1, height, width)):
            raise ValueError("depth geometry differs from color")
        if self.history is None or self.history.presence != "present":
            raise ValueError("non-reset packet needs an explicit history resource")
        if self.history.lineage_frame_id != self.expected_predecessor:
            raise ValueError("history lineage does not match expected predecessor")
        for guide in (self.motion, self.depth, self.history):
            guide.manifest()

    def eligibility_manifest(self, *, width: int, height: int, dataset: str = "registered") -> dict[str, Any]:
        """Return one frame record for ``nr_temporal_eligibility.py``."""
        self.validate(width=width, height=height)
        return {
            "frame_id": self.frame_id, "pts": self.pts,
            "time_base": {"numerator": self.time_base_num, "denominator": self.time_base_den},
            "reset": self.reset, "reset_reason": "sequence_start" if self.reset else None,
            "scene_epoch": self.scene_epoch,
            "expected_predecessor": self.expected_predecessor,
            "resources": {"color": self.color.manifest(), "depth": self.depth.manifest(),
                          "motion": self.motion.manifest(),
                          "history": None if self.history is None else self.history.manifest()},
            "packet_execution": {"previous_color_lane": self.previous_color is not None,
                                  "motion_depth_history_consumed_by_native_graph": False,
                                  "native_temporal_qualified": False},
            "dataset": dataset,
        }

    @staticmethod
    def manifest_sha256(manifest: dict[str, Any]) -> str:
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()


class NativeHistoryLedger:
    """Host owner ledger paired with the experimental native temporal lane.

    It commits a new history owner only after the caller reports physical
    completion and a valid output. The native graph currently consumes RGB and
    pixel motion; this ledger keeps depth/history lineage explicit without
    claiming that the graph has a vendor history codec.
    """

    def __init__(self, session_id: int, *, epoch: int = 1):
        if type(session_id) is not int or session_id <= 0:
            raise ValueError("session_id must be positive")
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("epoch must be positive")
        self._lock = RLock()
        self._state = HistoryState(session_id, epoch, 0, None)

    def snapshot(self) -> HistoryState:
        with self._lock:
            return self._state

    def reset(self, epoch: int) -> HistoryState:
        with self._lock:
            if type(epoch) is not int or epoch <= self._state.epoch:
                raise HistoryCommitError("reset epoch must advance")
            self._state = HistoryState(self._state.session_id, epoch, 0, None)
            return self._state

    def admit(self, packet: TemporalPacket, *, width: int, height: int) -> None:
        packet.validate(width=width, height=height)
        with self._lock:
            state = self._state
            if packet.session_id != state.session_id or packet.epoch != state.epoch:
                raise HistoryCommitError("packet session or epoch differs")
            if packet.reset:
                if packet.expected_predecessor is not None or state.frame_id is not None:
                    raise HistoryCommitError("reset packet has a live predecessor")
                return
            history = packet.history
            if (state.frame_id is None or history is None or history.presence != "present" or
                    history.owner_id != state.owner_id or
                    history.owner_session_id != state.session_id or
                    history.owner_epoch != state.epoch or
                    history.owner_frame_id != state.frame_id or
                    packet.expected_predecessor != state.frame_id):
                raise HistoryCommitError("history owner or predecessor is stale")

    def commit(self, packet: TemporalPacket, *, width: int, height: int,
               physical_completion: bool, output_valid: bool,
               next_history_owner_id: int) -> HistoryState:
        if type(physical_completion) is not bool or type(output_valid) is not bool:
            raise TypeError("completion flags must be bool")
        if not physical_completion:
            return self.snapshot()
        if not output_valid:
            raise HistoryCommitError("invalid output cannot commit history")
        if type(next_history_owner_id) is not int or next_history_owner_id <= 0:
            raise HistoryCommitError("next history owner must be positive")
        self.admit(packet, width=width, height=height)
        with self._lock:
            state = self._state
            if next_history_owner_id == state.owner_id:
                raise HistoryCommitError("history owner must change on commit")
            self._state = HistoryState(state.session_id, state.epoch,
                                       next_history_owner_id, packet.frame_id)
            return self._state
