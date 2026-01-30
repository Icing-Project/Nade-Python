"""Nade Protocol Implementation."""

import sys
from pathlib import Path

# Add vendored packages to sys.path for Windows bundled liquid-dsp
# Only on Windows - Linux/macOS should use pip-installed liquid-dsp
if sys.platform == "win32":
    _vendor_path = Path(__file__).parent / "_vendor"
    if _vendor_path.exists() and str(_vendor_path) not in sys.path:
        sys.path.insert(0, str(_vendor_path))

from .audio import AudioStack
from .crypto.noise_wrapper import NoiseXKWrapper
from .modems import LiquidFourFSKModem, LiquidBFSKModem

# Protocol state machine
from .protocol import NadeProtocol, NadeState, Phase

# Engine (bridges protocol to concrete implementations)
from .engine import NadeEngine

# Transport abstraction
from .transport import ITransport, AudioTransport

# NDA Adapter (for C++ integration)
from .adapters import NDAAdapter

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
    # Adapters (NDA integration)
    "NDAAdapter",
]

__version__ = "0.3.0"  # Updated for NDA integration
