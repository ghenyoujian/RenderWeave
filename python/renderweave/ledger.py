"""CPU session lease ledger for fail-closed temporal host adapters.

The ledger never launches CUDA. It owns the ordering rules that a future
native backend must satisfy: physical retirement precedes slot reuse, stale or
invalid outputs cannot commit history, and close waits for all leases.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock


class LedgerError(RuntimeError):
    """Session lease ordering violation."""


@dataclass
class PendingTicket:
    ticket: int
    epoch: int
    generation: int
    predecessor: int | None
    slot: int
    owner_ids: tuple[int, ...]
    physical_event: int
    status: str = "accepted"
    physical_complete: bool = False
    quarantined: bool = False


class SessionLedger:
    """Thread-safe CPU ledger for one session's pending work."""

    def __init__(self, session_id: int, *, epoch: int = 1):
        if type(session_id) is not int or session_id <= 0 or type(epoch) is not int or epoch <= 0:
            raise ValueError("session_id and epoch must be positive integers")
        self._lock = RLock()
        self.session_id = session_id
        self.epoch = epoch
        self.generation = 1
        self.predecessor: int | None = None
        self._next_ticket = 1
        self._pending: dict[int, PendingTicket] = {}
        self._closed = False

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def begin(self, *, epoch: int, predecessor: int | None, slot: int,
              owner_ids: tuple[int, ...], physical_event: int) -> int:
        with self._lock:
            self._open()
            if epoch != self.epoch or predecessor != self.predecessor:
                raise LedgerError("stale epoch or predecessor")
            if type(slot) is not int or slot < 0 or not owner_ids or any(type(x) is not int or x <= 0 for x in owner_ids):
                raise ValueError("slot and owner_ids are invalid")
            if type(physical_event) is not int or physical_event <= 0:
                raise ValueError("physical_event must be positive")
            if any(item.slot == slot and not item.physical_complete for item in self._pending.values()):
                raise LedgerError("slot still has physical work")
            ticket = self._next_ticket
            self._next_ticket += 1
            self._pending[ticket] = PendingTicket(ticket, epoch, self.generation, predecessor,
                                                   slot, tuple(owner_ids), physical_event)
            return ticket

    def mark_running(self, ticket: int) -> None:
        with self._lock:
            item = self._item(ticket)
            if item.status != "accepted":
                raise LedgerError("ticket is not accepted")
            item.status = "running"

    def cancel(self, ticket: int) -> None:
        with self._lock:
            item = self._item(ticket)
            if item.physical_complete:
                raise LedgerError("cannot cancel retired ticket")
            item.status = "cancelled"
            item.quarantined = True

    def timeout(self, ticket: int) -> None:
        with self._lock:
            item = self._item(ticket)
            if item.physical_complete:
                raise LedgerError("cannot timeout retired ticket")
            item.status = "timed_out"
            item.quarantined = True

    def complete(self, ticket: int, *, physical_completion: bool,
                 output_valid: bool, predecessor_match: bool) -> bool:
        """Retire a ticket physically; return whether history may commit."""
        with self._lock:
            item = self._item(ticket)
            if type(physical_completion) is not bool or type(output_valid) is not bool or type(predecessor_match) is not bool:
                raise TypeError("completion flags must be bool")
            if not physical_completion:
                return False
            item.physical_complete = True
            commit = (item.status not in ("cancelled", "timed_out")
                      and output_valid and predecessor_match
                      and item.epoch == self.epoch and item.generation == self.generation
                      and item.predecessor == self.predecessor)
            if commit:
                self.predecessor = item.ticket
            del self._pending[ticket]
            return commit

    def reset(self, epoch: int) -> None:
        with self._lock:
            self._open()
            if type(epoch) is not int or epoch <= self.epoch:
                raise LedgerError("reset epoch must advance")
            self.epoch = epoch
            self.generation += 1
            self.predecessor = None
            # Old pending tickets remain until their physical completion is known.

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._pending:
                raise LedgerError("cannot close with pending physical work")
            self._closed = True

    def _open(self) -> None:
        if self._closed:
            raise LedgerError("session ledger is closed")

    def _item(self, ticket: int) -> PendingTicket:
        if type(ticket) is not int or ticket not in self._pending:
            raise LedgerError("unknown ticket")
        return self._pending[ticket]


class AdapterAdmission:
    """Admission wrapper that refuses tickets before identity/graph checks."""

    def __init__(self, ledger: SessionLedger):
        if not isinstance(ledger, SessionLedger):
            raise TypeError("ledger must be SessionLedger")
        self.ledger = ledger

    def admit(self, *, identity: dict[str, str], native_graph_admitted: bool,
               epoch: int, predecessor: int | None, slot: int,
               owner_ids: tuple[int, ...], physical_event: int) -> int:
        required = ("feature", "model_fingerprint", "device_profile", "input_profile", "implementation_kind")
        if not isinstance(identity, dict) or any(not isinstance(identity.get(key), str) or not identity[key] for key in required):
            raise LedgerError("support identity tuple is incomplete")
        if identity["implementation_kind"] != "native_self" or native_graph_admitted is not True:
            raise LedgerError("native graph admission is not proven")
        return self.ledger.begin(epoch=epoch, predecessor=predecessor, slot=slot,
                                 owner_ids=owner_ids, physical_event=physical_event)

    def complete(self, ticket: int, *, physical_completion: bool,
                 output_valid: bool, predecessor_match: bool) -> bool:
        return self.ledger.complete(ticket, physical_completion=physical_completion,
                                     output_valid=output_valid, predecessor_match=predecessor_match)

    def close(self) -> None:
        self.ledger.close()


__all__ = ["AdapterAdmission", "LedgerError", "PendingTicket", "SessionLedger"]
