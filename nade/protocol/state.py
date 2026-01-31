"""
Nade protocol state representation.

NadeState is immutable (frozen dataclass) to enable pure functional transitions.
The Phase enum represents the current stage in the connection lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal


class Phase(Enum):
    """Connection lifecycle phases."""

    # Initial state - no session active
    IDLE = auto()

    # Discovery phases
    PING_DISCOVERY = auto()       # Sending pings, awaiting response
    AWAIT_PING_RESPONSE = auto()  # Peer pinged us, awaiting their PONG to our PING

    # Handshake phases (initiator path)
    HS_INITIATOR_STARTING = auto()      # Starting handshake, will send M1
    HS_INITIATOR_AWAITING_M2 = auto()   # M1 sent, waiting for M2
    # Note: Initiator transitions directly to ESTABLISHED after receiving M2
    # (crypto completes when M3 is generated)

    # Handshake phases (responder path)
    HS_RESPONDER_STARTING = auto()      # Starting handshake, waiting for M1
    HS_RESPONDER_AWAITING_M3 = auto()   # M2 sent, waiting for M3

    # Active session
    ESTABLISHED = auto()

    # Session ending states (future implementation)
    PAUSED = auto()     # No data, session alive, can resume
    CLOSING = auto()    # Graceful shutdown in progress
    EXPIRED = auto()    # Session timed out, returning to IDLE


@dataclass(frozen=True)
class NadeState:
    """
    Immutable protocol state.

    All state transitions produce a new NadeState instance.
    This enables pure functional protocol logic with no hidden state.
    """

    # Current phase in the connection lifecycle
    phase: Phase = Phase.IDLE

    # Session role (set on StartSession)
    role: Literal["initiator", "responder"] | None = None

    # Handshake progress tracking
    handshake_messages_sent: int = 0
    handshake_messages_received: int = 0

    # Discovery state
    ping_counter: int = 0           # Incrementing ping ID (wraps at 256)
    peer_ping_id: int | None = None # Last ping ID received from peer
    discovery_mode: bool = True     # Auto-discovery enabled

    # Pending outbound data (encrypted SDUs waiting for transport capacity)
    tx_pending: tuple[bytes, ...] = field(default_factory=tuple)

    # Configuration (can be customized per-session)
    # Note: actual values are placeholders; real implementation will use
    # configurable parameters passed at session start
    handshake_timeout_ms: int = 10_000      # 10 seconds
    session_expiry_ms: int = 300_000        # 5 minutes (future)
    keepalive_interval_ms: int = 30_000     # 30 seconds (future)

    # Error state (if any)
    last_error: str | None = None
