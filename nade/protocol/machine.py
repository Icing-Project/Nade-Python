"""
Nade Protocol State Machine.

This is the core protocol logic, implemented as a pure function:
    step(state, event) -> (new_state, actions)

No I/O, no side effects, no clock dependency.
The engine executes the returned actions.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .state import NadeState, Phase
from .events import (
    Event,
    StartSession,
    StopSession,
    TransportRxReady,
    TransportTxCapacity,
    AppSendData,
    TimerExpired,
    LinkQualityUpdate,
)
from .actions import (
    Action,
    CryptoStartHandshake,
    CryptoProcessMessage,
    CryptoEncrypt,
    CryptoDecrypt,
    TransportSend,
    TransportFlushHandshake,
    TimerStart,
    TimerCancel,
    AppDeliver,
    AppNotify,
    Log,
)


# Type alias for the step function signature
StepResult = tuple[NadeState, list[Action]]


class NadeProtocol:
    """
    Pure functional state machine for the Nade protocol.

    Usage:
        state = NadeState()
        state, actions = NadeProtocol.step(state, StartSession(role="initiator"))
        # engine executes actions...
        state, actions = NadeProtocol.step(state, TransportRxReady(sdus=(m2_bytes,)))
        # etc.
    """

    @staticmethod
    def step(state: NadeState, event: Event) -> StepResult:
        """
        Process an event and return (new_state, actions).

        This is the ONLY entry point for protocol logic.
        All state transitions and action generation happens here.
        """

        # Dispatch based on current phase and event type
        handler = _HANDLERS.get((type(state.phase).__name__, state.phase, type(event)))

        if handler:
            return handler(state, event)

        # Try phase-agnostic handlers
        handler = _GLOBAL_HANDLERS.get(type(event))
        if handler:
            return handler(state, event)

        # No handler: no state change, no actions
        return (state, [])


# =============================================================================
# Phase-specific handlers
# =============================================================================

def _handle_idle_start_session(state: NadeState, event: StartSession) -> StepResult:
    """IDLE + StartSession -> begin handshake."""

    if event.role == "initiator":
        return (
            replace(
                state,
                phase=Phase.HS_INITIATOR_STARTING,
                role="initiator",
                handshake_messages_sent=0,
                handshake_messages_received=0,
                last_error=None,
            ),
            [
                Log("info", f"[NadeProtocol] Starting session as initiator"),
                CryptoStartHandshake(is_initiator=True),
                TransportFlushHandshake(),
                TimerStart("handshake_timeout", state.handshake_timeout_ms),
            ]
        )
    else:
        return (
            replace(
                state,
                phase=Phase.HS_RESPONDER_STARTING,
                role="responder",
                handshake_messages_sent=0,
                handshake_messages_received=0,
                last_error=None,
            ),
            [
                Log("info", f"[NadeProtocol] Starting session as responder"),
                CryptoStartHandshake(is_initiator=False),
                TimerStart("handshake_timeout", state.handshake_timeout_ms),
            ]
        )


def _handle_hs_initiator_starting_tx_capacity(state: NadeState, event: TransportTxCapacity) -> StepResult:
    """Initiator starting + TX capacity -> M1 should be flushed, advance to awaiting M2."""
    return (
        replace(state, phase=Phase.HS_INITIATOR_AWAITING_M2, handshake_messages_sent=1),
        [
            Log("info", "[NadeProtocol] Initiator: M1 queued, awaiting M2"),
            TransportFlushHandshake(),
        ]
    )


def _handle_hs_initiator_awaiting_m2_rx(state: NadeState, event: TransportRxReady) -> StepResult:
    """Initiator awaiting M2 + RX -> process M2, send M3, handshake complete.

    Note: In Noise XK, the initiator's handshake completes cryptographically
    after processing M2 (when M3 is generated and cipher states derived).
    We transition to ESTABLISHED here, matching the original behavior.
    M3 will be flushed to transport on the next TX capacity event.
    """
    if not event.sdus:
        return (state, [])

    m2 = event.sdus[0]
    return (
        replace(
            state,
            phase=Phase.ESTABLISHED,
            handshake_messages_received=state.handshake_messages_received + 1,
        ),
        [
            Log("info", f"[NadeProtocol] Initiator: received M2 ({len(m2)} bytes)"),
            CryptoProcessMessage(m2),
            TransportFlushHandshake(),
            TimerCancel("handshake_timeout"),
            Log("info", "[NadeProtocol] Handshake complete (initiator)"),
            AppNotify("handshake_complete", {"role": "initiator"}),
        ]
    )


def _handle_hs_responder_starting_rx(state: NadeState, event: TransportRxReady) -> StepResult:
    """Responder starting + RX -> receive M1, send M2."""
    if not event.sdus:
        return (state, [])

    m1 = event.sdus[0]
    return (
        replace(
            state,
            phase=Phase.HS_RESPONDER_AWAITING_M3,
            handshake_messages_received=1,
        ),
        [
            Log("info", f"[NadeProtocol] Responder: received M1 ({len(m1)} bytes)"),
            CryptoProcessMessage(m1),
            TransportFlushHandshake(),
        ]
    )


def _handle_hs_responder_awaiting_m3_tx_capacity(state: NadeState, event: TransportTxCapacity) -> StepResult:
    """Responder awaiting M3 + TX capacity -> M2 should be flushed."""
    return (
        replace(state, handshake_messages_sent=state.handshake_messages_sent + 1),
        [
            Log("info", "[NadeProtocol] Responder: M2 queued, awaiting M3"),
            TransportFlushHandshake(),
        ]
    )


def _handle_hs_responder_awaiting_m3_rx(state: NadeState, event: TransportRxReady) -> StepResult:
    """Responder awaiting M3 + RX -> receive M3, handshake complete."""
    if not event.sdus:
        return (state, [])

    m3 = event.sdus[0]
    return (
        replace(
            state,
            phase=Phase.ESTABLISHED,
            handshake_messages_received=state.handshake_messages_received + 1,
        ),
        [
            Log("info", f"[NadeProtocol] Responder: received M3 ({len(m3)} bytes)"),
            CryptoProcessMessage(m3),
            TimerCancel("handshake_timeout"),
            Log("info", "[NadeProtocol] Handshake complete (responder)"),
            AppNotify("handshake_complete", {"role": "responder"}),
        ]
    )


def _handle_established_app_send(state: NadeState, event: AppSendData) -> StepResult:
    """ESTABLISHED + AppSendData -> encrypt and queue for TX."""
    return (
        state,
        [
            Log("info", f"[NadeProtocol] Encrypting SDU ({len(event.payload)} bytes)"),
            CryptoEncrypt(event.payload),
        ]
    )


def _handle_established_rx(state: NadeState, event: TransportRxReady) -> StepResult:
    """ESTABLISHED + RX -> decrypt incoming data."""
    actions: list[Action] = []
    for sdu in event.sdus:
        actions.append(Log("info", f"[NadeProtocol] Decrypting SDU ({len(sdu)} bytes)"))
        actions.append(CryptoDecrypt(sdu))
    return (state, actions)


def _handle_established_tx_capacity(state: NadeState, event: TransportTxCapacity) -> StepResult:
    """ESTABLISHED + TX capacity -> flush any pending data."""
    # Future: could dequeue from tx_pending here
    return (state, [])


# =============================================================================
# Global handlers (phase-agnostic)
# =============================================================================

def _handle_timer_expired(state: NadeState, event: TimerExpired) -> StepResult:
    """Handle timer expiry (any phase)."""

    if event.timer_id == "handshake_timeout":
        # Handshake timeout - return to IDLE with error
        return (
            replace(
                state,
                phase=Phase.EXPIRED,
                last_error="handshake_timeout",
            ),
            [
                Log("error", "[NadeProtocol] Handshake timeout"),
                AppNotify("error", {"reason": "handshake_timeout"}),
            ]
        )

    if event.timer_id == "session_expiry":
        # Session expired (future)
        return (
            replace(state, phase=Phase.EXPIRED, last_error="session_expired"),
            [
                Log("info", "[NadeProtocol] Session expired"),
                AppNotify("session_expired", {}),
            ]
        )

    # Unknown timer - log and ignore
    return (state, [Log("warn", f"[NadeProtocol] Unknown timer expired: {event.timer_id}")])


def _handle_stop_session(state: NadeState, event: StopSession) -> StepResult:
    """Handle session stop request (any phase)."""
    return (
        replace(state, phase=Phase.IDLE, role=None, last_error=None),
        [
            TimerCancel("handshake_timeout"),
            TimerCancel("session_expiry"),
            Log("info", f"[NadeProtocol] Session stopped: {event.reason}"),
            AppNotify("session_stopped", {"reason": event.reason}),
        ]
    )


# =============================================================================
# Handler dispatch tables
# =============================================================================

# Phase-specific handlers: (phase_type_name, phase, event_type) -> handler
_HANDLERS: dict[tuple, Callable[[NadeState, Event], StepResult]] = {
    # IDLE
    ("Phase", Phase.IDLE, StartSession): _handle_idle_start_session,

    # Initiator handshake
    ("Phase", Phase.HS_INITIATOR_STARTING, TransportTxCapacity): _handle_hs_initiator_starting_tx_capacity,
    ("Phase", Phase.HS_INITIATOR_AWAITING_M2, TransportRxReady): _handle_hs_initiator_awaiting_m2_rx,
    # Note: Initiator transitions directly to ESTABLISHED after receiving M2

    # Responder handshake
    ("Phase", Phase.HS_RESPONDER_STARTING, TransportRxReady): _handle_hs_responder_starting_rx,
    ("Phase", Phase.HS_RESPONDER_AWAITING_M3, TransportTxCapacity): _handle_hs_responder_awaiting_m3_tx_capacity,
    ("Phase", Phase.HS_RESPONDER_AWAITING_M3, TransportRxReady): _handle_hs_responder_awaiting_m3_rx,

    # Established
    ("Phase", Phase.ESTABLISHED, AppSendData): _handle_established_app_send,
    ("Phase", Phase.ESTABLISHED, TransportRxReady): _handle_established_rx,
    ("Phase", Phase.ESTABLISHED, TransportTxCapacity): _handle_established_tx_capacity,
}

# Global handlers: event_type -> handler (checked if no phase-specific handler)
_GLOBAL_HANDLERS: dict[type, Callable[[NadeState, Event], StepResult]] = {
    TimerExpired: _handle_timer_expired,
    StopSession: _handle_stop_session,
}
