"""RenderWeave C ABI bindings. Native model execution is queried, never assumed."""
from ._core import CoreError, Runtime
from .temporal import TemporalError, TemporalLibrary, TemporalSession
from .ledger import AdapterAdmission, LedgerError, SessionLedger
from .temporal_packet import GuidePlane, HistoryCommitError, NativeHistoryLedger, TemporalPacket
from .native_sr import NativeSRRuntime

__version__ = "0.1.0"
__all__ = ["CoreError", "Runtime", "TemporalError", "TemporalLibrary", "TemporalSession", "AdapterAdmission", "LedgerError", "SessionLedger", "GuidePlane", "TemporalPacket", "NativeHistoryLedger", "HistoryCommitError", "CatmullRomSubmission", "HistoryBinding", "LaunchBinding", "PostprocessAdmissionError", "admit_catmull_rom"]
__all__.append("NativeSRRuntime")

from .sr_postprocess import CatmullRomSubmission, HistoryBinding, LaunchBinding, PostprocessAdmissionError, admit_catmull_rom
