# DryBox v1-compatible adapter for Nade-Python
#
# Architecture:
# - Audio mode: Uses NadeEngine (state machine) for protocol logic
# - Byte mode: Full NoiseXK handshake + encrypted SDU pass-through
#
# The adapter converts DryBox's clock-driven model (push_tx_block/pull_rx_block)
# to the event-driven NadeEngine interface (for audio mode).

from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional, Tuple
import os
import json

from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.x25519.private import PrivateKey

try:
    import numpy as np
except Exception:
    np = None  # Audio mode requires numpy

# Nade imports
from nade.crypto.noise_wrapper import NoiseXKWrapper
from nade.transport import AudioTransport
from nade.engine import NadeEngine
from nade.protocol import (
    StartSession,
    TransportRxReady,
    TransportTxCapacity,
    AppSendData,
    TimerExpired,
)

ABI_VERSION = "dbx-v1"
HANDSHAKE_TAG = 0x20
SDU_TAG = 0x30
SDU_MAX_BYTES = 1024
INITIATOR_SIDE = "L"


def _merged_nade_cfg(base: dict | None) -> dict:
    """Merge base nade cfg with optional JSON in NADE_ADAPTER_CFG."""
    cfg: dict = (base or {}).copy()
    try:
        env_cfg = os.environ.get("NADE_ADAPTER_CFG")
        if env_cfg:
            parsed = json.loads(env_cfg)
            if isinstance(parsed, dict):
                cfg.update(parsed)
    except Exception:
        pass
    return cfg


class Adapter:
    """
    DryBox adapter (v1) for Nade.

    Audio mode: Delegates to NadeEngine (state machine architecture)
    Byte mode: Full NoiseXK handshake + encrypted SDU pass-through
    """

    def __init__(self):
        self.cfg: dict = {}
        self.ctx: Optional[Any] = None
        self.mode: str = "byte"
        self.side: str = "L"
        self.crypto: dict = {}

        # ByteLink state with full NoiseXK
        self._byte_t_ms: int = 0
        self._byte_txq: Deque[Tuple[bytes, int]] = deque()
        self._byte_noise: Optional[NoiseXKWrapper] = None
        self._byte_pending_handshake: bool = True
        self._byte_handshake_complete: bool = False

        # Audio mode: NadeEngine
        self._engine: Optional[NadeEngine] = None
        self._active_timers: dict[str, int] = {}  # timer_id -> expiry_ms

    # ---- Capabilities ----
    def nade_capabilities(self) -> dict:
        return {
            "abi_version": ABI_VERSION,
            "bytelink": True,
            "audioblock": True,
            "sdu_max_bytes": SDU_MAX_BYTES,
            "audioparams": {"sr": 8000, "block": 160},
        }

    # ---- Lifecycle ----
    def init(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.side = self.cfg.get("side", "L")
        self.mode = self.cfg.get("mode", "audio")
        self.crypto = self.cfg.get("crypto", {}) or {}
        self.nade_cfg = _merged_nade_cfg(self.cfg.get("nade", {}) or {})

    def start(self, ctx):
        self.ctx = ctx

        if self.mode == "audio":
            self._start_audio_mode()
        else:
            self._start_byte_mode()

    def _start_audio_mode(self):
        """Initialize audio mode using NadeEngine."""
        self._log("info", f"[Adapter] start() side={self.side} mode=audio")

        # Build crypto
        priv_raw = self.crypto.get("priv")
        pub_raw = self.crypto.get("pub")
        peer_pub_raw = self.crypto.get("peer_pub")

        self._log("info", f"[Noise] Local pub={pub_raw.hex()} peer_pub={peer_pub_raw.hex() if peer_pub_raw else None}")

        local_kp = KeyPair(PublicKey(bytes(pub_raw)), PrivateKey(bytes(priv_raw)))
        peer_pub_obj = PublicKey(bytes(peer_pub_raw)) if peer_pub_raw else None

        noise = NoiseXKWrapper(
            keypair=local_kp,
            peer_pubkey=peer_pub_obj,
            debug_callback=lambda *args: self._log("debug", "[Noise dbg] " + str(args[-1])),
        )

        # Build transport
        modem_cfg = (self.nade_cfg or {}).get("modem_cfg") or {}
        modem_name = (self.nade_cfg or {}).get("modem", "bfsk")

        self._log("info", f"[Adapter] Creating AudioTransport modem={modem_name} cfg={modem_cfg}")

        transport = AudioTransport(
            modem=modem_name,
            modem_cfg=modem_cfg,
            logger=self._log,
        )

        # Build engine
        self._engine = NadeEngine(
            crypto=noise,
            transport=transport,
            logger=self._log,
        )

        # Wire callbacks
        self._engine.on_app_data = self._on_decrypted_data
        self._engine.on_event = self._on_protocol_event

        # Start session
        role = "initiator" if self.side == INITIATOR_SIDE else "responder"
        self._engine.feed_event(StartSession(role=role))

        # Schedule initial timers
        self._sync_timers(t_ms=0)

    def _start_byte_mode(self):
        """Initialize byte mode with full NoiseXK."""
        self._log("info", f"[Adapter] start() side={self.side} mode=byte")

        priv_raw = self.crypto.get("priv")
        pub_raw = self.crypto.get("pub")
        peer_pub_raw = self.crypto.get("peer_pub")

        self._log("info", f"[Noise] Local pub={pub_raw.hex()} peer_pub={peer_pub_raw.hex() if peer_pub_raw else None}")

        local_kp = KeyPair(PublicKey(bytes(pub_raw)), PrivateKey(bytes(priv_raw)))
        peer_pub_obj = PublicKey(bytes(peer_pub_raw)) if peer_pub_raw else None

        self._byte_noise = NoiseXKWrapper(
            keypair=local_kp,
            peer_pubkey=peer_pub_obj,
            debug_callback=lambda *args: self._log("debug", "[Noise dbg] " + str(args[-1])),
        )

        self._byte_pending_handshake = True
        self._byte_handshake_complete = False
        self._byte_txq.clear()

    def stop(self) -> None:
        pass

    # ---- Timers ----
    def on_timer(self, t_ms: int) -> None:
        if self.mode != "audio":
            # ByteLink mode: start handshake on first timer tick
            self._byte_t_ms = t_ms

            if self._byte_pending_handshake:
                self._byte_pending_handshake = False
                self._byte_noise.start_handshake(initiator=(self.side == INITIATOR_SIDE))
                self._log("info", f"[Noise] Starting NoiseXK handshake (initiator={self.side == INITIATOR_SIDE})")

            # Queue handshake messages
            if self._byte_noise:
                hs_msg = self._byte_noise.get_next_handshake_message()
                if hs_msg:
                    self._log("info", f"[NoiseXK] TX handshake msg len={len(hs_msg)} hex={hs_msg.hex()[:32]}...")
                    self._byte_txq.append((bytes([HANDSHAKE_TAG]) + hs_msg, self._byte_t_ms))

                # Update completion status
                if not self._byte_handshake_complete and self._byte_noise.handshake_complete:
                    self._byte_handshake_complete = True
                    self._log("info", "[NoiseXK] Handshake complete")
            return

    # ---- BYTELINK ----
    def poll_link_tx(self, budget: int):
        if self.mode == "audio":
            return []
        out: List[Tuple[bytes, int]] = []
        while self._byte_txq and len(out) < budget:
            out.append(self._byte_txq.popleft())
        return out

    def on_link_rx(self, sdu: bytes):
        if self.mode == "audio" or not sdu:
            return

        # Handle handshake messages
        if sdu[0] == HANDSHAKE_TAG:
            hs_data = sdu[1:]
            self._log("info", f"[NoiseXK] RX handshake message len={len(hs_data)} hex={hs_data.hex()[:32]}...")

            if not self._byte_noise:
                self._log("error", "[Adapter] RX handshake but _byte_noise is None!")
                return

            try:
                self._byte_noise.process_handshake_message(hs_data)
            except Exception as e:
                self._log("error", f"[NoiseXK] ERROR while processing handshake message: {e}")
                return

            self._log("msg", f"[NoiseXK] Handshake state: complete={self._byte_noise.handshake_complete}")

            if self._byte_noise.handshake_complete:
                self._byte_handshake_complete = True
                self._log("info", "[NoiseXK] Handshake complete")

            return

        # Handle encrypted SDUs
        if sdu[0] == SDU_TAG:
            if not self._byte_handshake_complete:
                self._log("warn", "[Nade] Received SDU but handshake is incomplete. Dropped.")
                return

            ct = sdu[1:]
            try:
                plaintext = self._byte_noise.decrypt_sdu(b"", ct)
                self._log("info", f"[Nade] RX decrypted SDU len={len(plaintext)} text={plaintext!r}")
                self._log("msg", {"Received a message": plaintext.decode("utf-8", errors="ignore")})
            except Exception as e:
                self._log("error", f"[NoiseXK] decrypt_sdu FAILED: {e} (len={len(ct)})")

    # ---- AUDIOBLOCK TX ----
    def push_tx_block(self, t_ms: int):
        if not self._engine:
            return np.zeros(160, dtype=np.int16) if np else None

        # Check and fire expired timers
        self._check_timers(t_ms)

        # Signal TX capacity to protocol
        self._engine.feed_event(TransportTxCapacity(budget_bytes=255))

        # Sync any new timers
        self._sync_timers(t_ms)

        # Get PCM from transport
        return self._engine.get_tx_samples(160, t_ms)

    # ---- AUDIOBLOCK RX ----
    def pull_rx_block(self, pcm, t_ms: int):
        if not self._engine:
            return

        # Feed samples to transport
        sdus = self._engine.feed_rx_samples(pcm, t_ms)

        # If frames decoded, feed to protocol
        if sdus:
            self._engine.feed_event(TransportRxReady(tuple(sdus)))

    # ---- API: send SDU ----
    def send_sdu(self, data: bytes):
        if self.mode == "audio":
            if not self._engine:
                self._log("warn", "[Adapter] send_sdu called but engine not initialized")
                return

            if not self._engine.is_established:
                self._log("warn", "[Adapter] send_sdu called but handshake incomplete")
                return

            self._engine.feed_event(AppSendData(payload=data))

        else:  # byte mode
            if not self._byte_noise or not self._byte_handshake_complete:
                self._log("warn", "[Nade] Tried to send SDU but handshake is incomplete. Dropped.")
                return

            try:
                ct = self._byte_noise.encrypt_sdu(b"", data)
            except Exception as e:
                self._log("error", f"[NoiseXK] encrypt_sdu FAILED: {e} plaintext={data!r}")
                return

            self._log("info", f"[Nade] Queue TX SDU plaintext_len={len(data)} ct_len={len(ct)}")
            self._byte_txq.append((bytes([SDU_TAG]) + ct, self._byte_t_ms))

    def is_handshake_complete(self) -> bool:
        """Query handshake state."""
        if self.mode == "audio":
            if not self._engine:
                return False
            return self._engine.is_established
        else:  # byte mode
            if not self._byte_noise:
                return False
            return self._byte_noise.handshake_complete

    # ---- Timer Management ----
    def _sync_timers(self, t_ms: int):
        """Sync pending timers from engine to adapter's active timers."""
        if not self._engine:
            return

        for timer_id, req in self._engine.get_pending_timers().items():
            if timer_id not in self._active_timers and req.started_at_ms is None:
                # New timer: schedule it
                expiry = t_ms + req.duration_ms
                self._active_timers[timer_id] = expiry
                self._engine.acknowledge_timer(timer_id, t_ms)
                self._log("debug", f"[Adapter] Timer scheduled: {timer_id} expires at {expiry}ms")

    def _check_timers(self, t_ms: int):
        """Check for expired timers and fire events."""
        if not self._engine:
            return

        expired = [tid for tid, expiry in self._active_timers.items() if t_ms >= expiry]
        for timer_id in expired:
            if not self._engine.is_timer_cancelled(timer_id):
                self._log("debug", f"[Adapter] Timer expired: {timer_id} at {t_ms}ms")
                self._engine.feed_event(TimerExpired(timer_id=timer_id))
            self._active_timers.pop(timer_id, None)
            self._engine.clear_cancelled_timer(timer_id)

    # ---- Callbacks ----
    def _on_decrypted_data(self, data: bytes):
        """Called by engine when decrypted data is available."""
        self._log("info", f"[Adapter] RX decrypted SDU len={len(data)} text={data!r}")
        self._log("msg", {"Received a message: ": data.decode("utf-8", errors="ignore")})

    def _on_protocol_event(self, event_type: str, details: dict):
        """Called by engine on protocol events."""
        self._log("info", f"[Adapter] Protocol event: {event_type} {details}")

        if event_type == "handshake_complete":
            self._log("msg", "[Adapter] Handshake complete!")

    # ---- Logging ----
    def _log(self, level: str, payload: Any) -> None:
        if not self.ctx:
            return
        if level == "metric" and isinstance(payload, dict):
            self.ctx.emit_event("metric", payload)
        elif isinstance(payload, str):
            self.ctx.emit_event("log", {"level": level, "msg": payload})
        else:
            self.ctx.emit_event("log", {"level": str(level), "msg": str(payload)})
