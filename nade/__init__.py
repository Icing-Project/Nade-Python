from .audio import AudioStack
from .crypto.noise_wrapper import NoiseXKWrapper
from .modems import LiquidFourFSKModem, LiquidBFSKModem

# Protocol state machine
from .protocol import NadeProtocol, NadeState, Phase

# Engine (bridges protocol to concrete implementations)
from .engine import NadeEngine

# Transport abstraction
from .transport import ITransport, AudioTransport

# Backwards-compatible alias for the previous FourFSK modem name
FourFSKModem = LiquidFourFSKModem

__all__ = [
    # Core (backwards compatible)
    "AudioStack",
    "NoiseXKWrapper",
    # Protocol
    "NadeProtocol",
    "NadeState",
    "Phase",
    # Engine
    "NadeEngine",
    # Transport
    "ITransport",
    "AudioTransport",
    # Modems
    "LiquidBFSKModem",
    "LiquidFourFSKModem",
    "FourFSKModem",
]

__version__ = "0.2.0"
