"""
NDA Adapter for Nade-Python text messaging modem.

This adapter wraps Nade's AudioStack, NoiseXKWrapper, and NadeEngine to provide
a simple interface for C++ NDA plugins via pybind11.

Key Features:
- Separate from Nade core (zero changes to existing Nade files)
- Handles resampling (48kHz <-> 8kHz)
- Manages text message queuing
- Thread-safe (called from Python worker thread)
- Error codes instead of exceptions (for C++ boundary)

Usage (from C++ via pybind11):
    adapter = NDAAdapter(private_key, local_pub, remote_pub, 48000)
    adapter.send_text_message("Hello")
    fsk_audio = adapter.get_tx_audio(10.67)  # Generate FSK
    adapter.process_rx_audio(fsk_audio, 10.67)  # Demodulate
    messages = adapter.get_received_messages()
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
from scipy import signal

# Nade components
from nade.audio import AudioStack
from nade.crypto.noise_wrapper import NoiseXKWrapper
from nade.transport import AudioTransport
from nade.engine import NadeEngine
from nade.protocol import StartSession, StopSession, AppSendData, TransportRxReady

# Dissononce key types
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.x25519.private import PrivateKey


def _bytes_to_keypair(private_key: bytes, public_key: bytes) -> KeyPair:
    """Convert raw bytes to dissononce KeyPair."""
    priv = PrivateKey(private_key)
    pub = PublicKey(public_key)
    # Note: KeyPair constructor is (public, private) not (private, public)
    return KeyPair(pub, priv)


def _bytes_to_public_key(public_key: bytes) -> PublicKey:
    """Convert raw bytes to dissononce PublicKey."""
    return PublicKey(public_key)


class NDAAdapter:
    """
    Adapter for NDA C++ plugins to use Nade-Python modem.

    This class wraps AudioStack and NoiseXKWrapper to provide a simple
    interface for text messaging over radio.

    Note: This is NOT a singleton. C++ NadeExternalIO manages the singleton
    pattern and creates ONE instance of this adapter.

    Thread Safety:
        Called only from Python worker thread (owned by C++ NadeExternalIO).
        No internal locking needed.
    """

    def __init__(
        self,
        x25519_private_key: bytes,
        x25519_local_public: bytes,
        x25519_peer_public: bytes,
        nda_sample_rate: int = 48000,
        modem_mode: str = "4fsk",
        is_initiator: bool = True,
    ):
        """
        Initialize NDA adapter.

        Args:
            x25519_private_key: 32-byte local private key
            x25519_local_public: 32-byte local public key
            x25519_peer_public: 32-byte peer public key
            nda_sample_rate: NDA's sample rate (typically 48000)
            modem_mode: FSK mode - "bfsk" or "4fsk"
            is_initiator: True if this side initiates handshake
        """
        # Validate keys
        if len(x25519_private_key) != 32:
            raise ValueError("Private key must be 32 bytes")
        if len(x25519_local_public) != 32:
            raise ValueError("Local public key must be 32 bytes")
        if len(x25519_peer_public) != 32:
            raise ValueError("Peer public key must be 32 bytes")

        # Sample rates
        self.nda_sample_rate = nda_sample_rate
        self.nade_sample_rate = 8000  # Fixed for current Nade modems
        self.resample_ratio = self.nda_sample_rate / self.nade_sample_rate

        # Convert keys to dissononce types
        keypair = _bytes_to_keypair(x25519_private_key, x25519_local_public)
        peer_pubkey = _bytes_to_public_key(x25519_peer_public)

        # Logging callback
        self._log_messages: List[str] = []
        def logger(level: str, msg: str) -> None:
            self._log_messages.append(f"[{level}] {msg}")

        # Initialize Nade components
        # NoiseXKWrapper for crypto
        self.noise = NoiseXKWrapper(
            keypair=keypair,
            peer_pubkey=peer_pubkey,
            debug_callback=lambda msg: logger("noise", msg),
        )

        # AudioTransport wraps AudioStack (modem)
        modem_cfg = {
            "sample_rate_hz": self.nade_sample_rate,
            "block_size": 160,  # 20ms at 8kHz
        }
        self.transport = AudioTransport(
            modem=modem_mode,
            modem_cfg=modem_cfg,
            logger=logger,
        )

        # NadeEngine bridges protocol to concrete implementations
        self.engine = NadeEngine(
            crypto=self.noise,
            transport=self.transport,
            logger=logger,
        )

        # Setup callbacks for received data
        self.received_messages: List[str] = []
        self.engine.on_app_data = self._on_app_data

        # State
        self.is_transmitting = False
        self.session_started = False
        self._is_initiator = is_initiator

        # Start session
        self._start_session()

    def _start_session(self) -> None:
        """Start Nade session (initiator or responder)."""
        role = "initiator" if self._is_initiator else "responder"
        self.engine.feed_event(StartSession(role=role))
        self.session_started = True

    def _on_app_data(self, data: bytes) -> None:
        """Callback when decrypted data is received."""
        try:
            text = data.decode("utf-8", errors="replace")
            self.received_messages.append(text)
        except Exception:
            pass

    # =========================================================================
    # TEXT MESSAGING API
    # =========================================================================

    def send_text_message(self, text: str) -> Dict[str, Any]:
        """
        Queue text message for transmission.

        Args:
            text: Text message (UTF-8, max 256 characters)

        Returns:
            Dictionary with:
            - 'success': bool
            - 'error': str (if success=False)

        Example:
            result = adapter.send_text_message("Hello")
            if result['success']:
                print("Message queued")
        """
        # Validate length
        if len(text) > 256:
            return {"success": False, "error": "Message too long (max 256 chars)"}

        # Encode as UTF-8
        try:
            text_bytes = text.encode("utf-8")
        except Exception as e:
            return {"success": False, "error": f"UTF-8 encoding error: {e}"}

        if len(text_bytes) > 256:
            return {"success": False, "error": "UTF-8 bytes exceed 256"}

        # Ensure session is established
        if not self.engine.is_established:
            # Handshake not complete yet - queue the message anyway
            # The engine will handle it once handshake completes
            pass

        # Feed AppSendData event to engine
        try:
            self.engine.feed_event(AppSendData(payload=text_bytes))
            self.is_transmitting = True
            return {"success": True, "error": ""}
        except Exception as e:
            return {"success": False, "error": f"Engine error: {e}"}

    def get_received_messages(self) -> List[str]:
        """
        Retrieve and clear all decoded messages.

        Returns:
            List of UTF-8 decoded messages.
        """
        messages = self.received_messages.copy()
        self.received_messages.clear()
        return messages

    # =========================================================================
    # AUDIO PROCESSING (Called from Python worker thread)
    # =========================================================================

    def get_tx_audio(self, duration_ms: float) -> np.ndarray:
        """
        Generate FSK audio for transmission (pre-buffering).

        Called from Python worker thread to fill pre-buffer.

        Args:
            duration_ms: Buffer duration in milliseconds

        Returns:
            FSK audio at NDA sample rate (float32, range [-1.0, +1.0])
            Returns zeros if no data to transmit.
        """
        # Calculate samples at 8kHz (Nade's rate)
        samples_8k = int(duration_ms * self.nade_sample_rate / 1000.0)

        # Get FSK from engine/transport
        fsk_int16 = self.engine.get_tx_samples(samples_8k, int(duration_ms))

        if fsk_int16 is None or len(fsk_int16) == 0:
            # No TX data - return silence at NDA sample rate
            self.is_transmitting = False
            samples_nda = int(duration_ms * self.nda_sample_rate / 1000.0)
            return np.zeros(samples_nda, dtype=np.float32)

        # Convert int16 -> float32
        fsk_float32 = fsk_int16.astype(np.float32) / 32768.0

        # Resample 8kHz -> NDA rate
        num_samples_nda = int(len(fsk_float32) * self.resample_ratio)
        if num_samples_nda > 0:
            fsk_nda = signal.resample(fsk_float32, num_samples_nda)
        else:
            fsk_nda = np.zeros(0, dtype=np.float32)

        return fsk_nda.astype(np.float32)

    def process_rx_audio(self, audio: np.ndarray, duration_ms: float) -> None:
        """
        Process received FSK audio (post-buffering).

        Called from Python worker thread after C++ buffers RX audio.

        Args:
            audio: FSK audio at NDA sample rate (float32)
            duration_ms: Buffer duration
        """
        if audio is None or len(audio) == 0:
            return

        # Resample NDA rate -> 8kHz
        num_samples_8k = int(len(audio) / self.resample_ratio)
        if num_samples_8k <= 0:
            return

        audio_8k = signal.resample(audio, num_samples_8k)

        # Convert float32 -> int16
        audio_int16 = (audio_8k * 32768.0).clip(-32768, 32767).astype(np.int16)

        # Feed to engine/transport
        sdus = self.engine.feed_rx_samples(audio_int16, int(duration_ms))

        # Process any received SDUs through protocol
        if sdus:
            self.engine.feed_event(TransportRxReady(sdus=tuple(sdus)))

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_rx_signal_quality(self) -> float:
        """Get RX signal quality [0.0, 1.0]."""
        # TODO: Implement based on modem's demodulator stats
        return 0.0

    def is_rx_synchronized(self) -> bool:
        """Check if demodulator is locked to signal."""
        # TODO: Implement based on modem's sync state
        return False

    def is_transmitting_active(self) -> bool:
        """Check if TX buffer has data."""
        return self.is_transmitting and self.transport.has_pending_tx()

    def get_mode(self) -> str:
        """Get current mode ('tx', 'rx', or 'idle')."""
        if self.is_transmitting:
            return "tx"
        return "rx"

    def is_session_established(self) -> bool:
        """Check if Noise handshake is complete."""
        return self.engine.is_established

    def get_log_messages(self) -> List[str]:
        """Get and clear log messages (for debugging)."""
        msgs = self._log_messages.copy()
        self._log_messages.clear()
        return msgs
