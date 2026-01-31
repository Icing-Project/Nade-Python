"""
Nade Engine - Executes protocol actions.

The engine bridges the pure functional protocol to concrete implementations:
- Crypto (NoiseXKWrapper)
- Transport (AudioTransport or ByteTransport)
- Timers (managed by adapter)
- Application callbacks

The engine is adapter-agnostic: it doesn't know about DryBox vs Desktop App.
Adapters feed events to the engine and handle timer scheduling.
"""
from __future__ import annotations

from typing import Callable, Any
from dataclasses import dataclass

from .protocol import (
    NadeProtocol,
    NadeState,
    Phase,
    Event,
    Action,
    # Actions
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
    SendPing,
    SendPong,
)
from .transport.interface import ITransport
from .crypto.noise_wrapper import NoiseXKWrapper


@dataclass
class TimerRequest:
    """A timer that the adapter should manage."""
    timer_id: str
    duration_ms: int
    started_at_ms: int | None = None  # Set by adapter when started


class NadeEngine:
    """
    Executes protocol actions by calling concrete implementations.

    Usage:
        engine = NadeEngine(crypto=noise_wrapper, transport=audio_transport)
        engine.on_app_data = lambda data: print(f"Received: {data}")
        engine.on_event = lambda event, details: print(f"Event: {event}")

        # Feed events from adapter
        engine.feed_event(StartSession(role="initiator"))
        engine.feed_event(TransportRxReady(sdus=(data,)))
    """

    def __init__(
        self,
        crypto: NoiseXKWrapper,
        transport: ITransport,
        logger: Callable[[str, str], None] | None = None,
    ):
        self._crypto = crypto
        self._transport = transport
        self._logger = logger or (lambda level, msg: None)

        # Protocol state
        self._state = NadeState()

        # Timer requests (adapter reads these and manages actual timing)
        self._pending_timers: dict[str, TimerRequest] = {}
        self._cancelled_timers: set[str] = set()

        # Application callbacks
        self.on_app_data: Callable[[bytes], None] | None = None
        self.on_event: Callable[[str, dict[str, Any]], None] | None = None

    # === Properties ===

    @property
    def state(self) -> NadeState:
        """Current protocol state (read-only)."""
        return self._state

    @property
    def phase(self) -> Phase:
        """Current protocol phase."""
        return self._state.phase

    @property
    def is_established(self) -> bool:
        """Check if session is established (handshake complete)."""
        return self._state.phase == Phase.ESTABLISHED

    @property
    def is_handshake_complete(self) -> bool:
        """Alias for is_established, for compatibility."""
        return self.is_established

    # === Event Feeding ===

    def feed_event(self, event: Event) -> None:
        """
        Feed an event to the protocol state machine.

        The protocol returns actions which are immediately executed.
        """
        new_state, actions = NadeProtocol.step(self._state, event)
        self._state = new_state

        for action in actions:
            self._execute(action)

    # === Action Execution ===

    def _execute(self, action: Action) -> None:
        """Execute a single action."""

        match action:
            case Log(level, message):
                self._logger(level, message)

            case CryptoStartHandshake(is_initiator):
                self._crypto.start_handshake(initiator=is_initiator)
                self._logger("debug", f"[Engine] Crypto handshake started (initiator={is_initiator})")

            case CryptoProcessMessage(data):
                try:
                    self._crypto.process_handshake_message(data)
                    self._logger("debug", f"[Engine] Processed handshake message ({len(data)} bytes)")
                except Exception as e:
                    self._logger("error", f"[Engine] Crypto process failed: {e}")

            case TransportFlushHandshake():
                # Flush any pending handshake messages from crypto to transport
                while True:
                    msg = self._crypto.get_next_handshake_message()
                    if msg is None:
                        break
                    self._transport.queue_tx_sdu(msg)
                    self._logger("debug", f"[Engine] Flushed handshake message ({len(msg)} bytes)")

            case TransportSend(sdu):
                if self._transport.queue_tx_sdu(sdu):
                    self._logger("debug", f"[Engine] Queued SDU ({len(sdu)} bytes)")
                else:
                    self._logger("warn", f"[Engine] Transport TX full, SDU dropped")

            case CryptoEncrypt(plaintext):
                try:
                    ciphertext = self._crypto.encrypt_sdu(b"", plaintext)
                    self._transport.queue_tx_sdu(ciphertext)
                    self._logger("debug", f"[Engine] Encrypted and queued ({len(plaintext)} -> {len(ciphertext)} bytes)")
                except Exception as e:
                    self._logger("error", f"[Engine] Encrypt failed: {e}")

            case CryptoDecrypt(ciphertext):
                try:
                    plaintext = self._crypto.decrypt_sdu(b"", ciphertext)
                    self._logger("debug", f"[Engine] Decrypted ({len(ciphertext)} -> {len(plaintext)} bytes)")
                    if self.on_app_data:
                        self.on_app_data(plaintext)
                except Exception as e:
                    self._logger("error", f"[Engine] Decrypt failed: {e}")

            case AppDeliver(payload):
                if self.on_app_data:
                    self.on_app_data(payload)

            case AppNotify(event_type, details):
                self._logger("info", f"[Engine] App event: {event_type} {details}")
                if self.on_event:
                    self.on_event(event_type, details)

            case TimerStart(timer_id, duration_ms):
                self._pending_timers[timer_id] = TimerRequest(timer_id, duration_ms)
                self._cancelled_timers.discard(timer_id)
                self._logger("debug", f"[Engine] Timer requested: {timer_id} ({duration_ms}ms)")

            case TimerCancel(timer_id):
                self._pending_timers.pop(timer_id, None)
                self._cancelled_timers.add(timer_id)
                self._logger("debug", f"[Engine] Timer cancelled: {timer_id}")

            case SendPing(ping_id):
                # Encode PING frame: NADE magic + type 0x01 + ping_id
                payload = b'\x4E\x41\x44\x45' + bytes([0x01, ping_id])
                self._transport.queue_tx_sdu(payload)
                self._logger("debug", f"[Engine] Sent PING #{ping_id}")

            case SendPong(ping_id):
                # Encode PONG frame: NADE magic + type 0x02 + ping_id
                payload = b'\x4E\x41\x44\x45' + bytes([0x02, ping_id])
                self._transport.queue_tx_sdu(payload)
                self._logger("debug", f"[Engine] Sent PONG #{ping_id}")

            case _:
                self._logger("warn", f"[Engine] Unknown action: {action}")

    # === Timer Management (for adapter) ===

    def get_pending_timers(self) -> dict[str, TimerRequest]:
        """
        Get timers that need to be scheduled.

        The adapter should:
        1. Call this to get pending timers
        2. Schedule them with its timing mechanism
        3. Call acknowledge_timer() once scheduled
        4. Feed TimerExpired events when they fire
        """
        return dict(self._pending_timers)

    def acknowledge_timer(self, timer_id: str, started_at_ms: int) -> None:
        """Mark a timer as scheduled by the adapter."""
        if timer_id in self._pending_timers:
            self._pending_timers[timer_id].started_at_ms = started_at_ms

    def is_timer_cancelled(self, timer_id: str) -> bool:
        """Check if a timer was cancelled (adapter should not fire it)."""
        return timer_id in self._cancelled_timers

    def clear_cancelled_timer(self, timer_id: str) -> None:
        """Clear a timer from the cancelled set after adapter handles it."""
        self._cancelled_timers.discard(timer_id)

    # === Transport Access (for adapter) ===

    def get_tx_samples(self, count: int, t_ms: int):
        """Get PCM samples from transport (adapter calls this for TX)."""
        return self._transport.get_tx_samples(count, t_ms)

    def feed_rx_samples(self, pcm, t_ms: int) -> list[bytes]:
        """Feed PCM samples to transport (adapter calls this for RX)."""
        return self._transport.feed_rx_samples(pcm, t_ms)

    def poll_modem_events(self) -> None:
        """
        Poll modem for ping/pong events and feed to protocol.

        Call this periodically from adapter's worker loop.
        """
        from .protocol.events import PingReceived, PongReceived

        # Access modem through transport
        modem = getattr(self._transport, 'modem', None)
        if modem is None:
            return

        events = modem.get_pending_ping_events()
        for event_type, ping_id in events:
            if event_type == 'ping_received':
                self.feed_event(PingReceived(ping_id=ping_id))
            elif event_type == 'pong_received':
                self.feed_event(PongReceived(ping_id=ping_id))
