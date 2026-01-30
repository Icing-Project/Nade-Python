"""
Events are inputs to the Nade state machine.

The protocol reacts to events and produces actions.
Events are clock-agnostic: adapters convert their timing model to events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Event:
    """Base class for all protocol events."""
    pass


# === Session Lifecycle ===

@dataclass(frozen=True)
class StartSession(Event):
    """Request to start a new Nade session."""
    role: Literal["initiator", "responder"]


@dataclass(frozen=True)
class StopSession(Event):
    """Request to stop the current session."""
    reason: str = "user_request"


# === Transport Events ===

@dataclass(frozen=True)
class TransportRxReady(Event):
    """Transport has decoded SDU(s) from the physical layer."""
    sdus: tuple[bytes, ...]


@dataclass(frozen=True)
class TransportTxCapacity(Event):
    """Transport signals it has capacity for outgoing data."""
    budget_bytes: int


# === Application Events ===

@dataclass(frozen=True)
class AppSendData(Event):
    """Application wants to send data through the encrypted channel."""
    payload: bytes


# === Timer Events ===

@dataclass(frozen=True)
class TimerExpired(Event):
    """A previously-set timer has expired."""
    timer_id: str  # e.g., "handshake_timeout", "session_expiry"


# === Link Quality Events (Future: adaptive control) ===

@dataclass(frozen=True)
class LinkQualityUpdate(Event):
    """Transport reports link quality metrics for adaptive decisions."""
    snr_db: float | None = None
    packet_loss_rate: float | None = None
    estimated_ber: float | None = None
