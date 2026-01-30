"""
Actions are outputs from the Nade state machine.

The engine executes actions by calling concrete implementations
(crypto, transport, timers, application callbacks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """Base class for all protocol actions."""
    pass


# === Crypto Actions ===

@dataclass(frozen=True)
class CryptoStartHandshake(Action):
    """Initialize the crypto layer and start handshake."""
    is_initiator: bool


@dataclass(frozen=True)
class CryptoProcessMessage(Action):
    """Process an incoming handshake message."""
    data: bytes


@dataclass(frozen=True)
class CryptoEncrypt(Action):
    """Encrypt plaintext and queue for transmission."""
    plaintext: bytes


@dataclass(frozen=True)
class CryptoDecrypt(Action):
    """Decrypt ciphertext and deliver to application."""
    ciphertext: bytes


# === Transport Actions ===

@dataclass(frozen=True)
class TransportSend(Action):
    """Send an SDU via the transport layer."""
    sdu: bytes


@dataclass(frozen=True)
class TransportFlushHandshake(Action):
    """Flush any pending handshake messages from crypto to transport."""
    pass


# === Timer Actions ===

@dataclass(frozen=True)
class TimerStart(Action):
    """Start or restart a timer."""
    timer_id: str
    duration_ms: int


@dataclass(frozen=True)
class TimerCancel(Action):
    """Cancel an active timer."""
    timer_id: str


# === Application Callbacks ===

@dataclass(frozen=True)
class AppDeliver(Action):
    """Deliver decrypted data to the application."""
    payload: bytes


@dataclass(frozen=True)
class AppNotify(Action):
    """Notify application of a protocol event."""
    event_type: str  # "handshake_complete", "session_expired", "error", etc.
    details: dict[str, Any] = field(default_factory=dict)


# === Logging ===

@dataclass(frozen=True)
class Log(Action):
    """Emit a log message."""
    level: str  # "debug", "info", "warn", "error"
    message: str
