# DryBox v1-compatible adapter for Nade-Python
# - Zero-arg __init__ (runner instantiates with cls())
# - init(cfg) to receive side/mode/crypto
# - start(ctx) to receive AdapterCtx (now_ms(), emit_event(), rng, config)
# - Byte mode: minimal handshake + SDU pass-through mock
# - Audio mode: silence source/sink mock (160 @ 8kHz)

from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional, Tuple
import os
import json
from nade.audio import AudioStack
from nade.crypto.noise_wrapper import NoiseXKWrapper
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.x25519.private import PrivateKey

try:
    import numpy as np
except Exception:
    np = None  # Audio mock works only if numpy is present

ABI_VERSION = "dbx-v1"
HANDSHAKE_TAG = 0x20
SDU_MAX_BYTES = 1024


def _merged_nade_cfg(base: dict | None) -> dict:                                    # TODO: remove when switching to DryBox controls OR effective statemachine
    """Merge base nade cfg with optional JSON in NADE_ADAPTER_CFG.
    Keeps behavior identical to previous inline merge (shallow update, best-effort).
    """
    cfg: dict = (base or {}).copy()
    try:
        env_cfg = os.environ.get("NADE_ADAPTER_CFG")
        if env_cfg:
            parsed = json.loads(env_cfg)
            if isinstance(parsed, dict):
                cfg.update(parsed)
    except Exception:
        # Preserve previous silent-fail behavior
        pass
    return cfg


class Adapter:
    """
    DryBox adapter (v1) for Nade.
    __init__(): zero-arg constructor
    init(cfg): receive side/mode/crypto (and nade cfg)
    start(ctx): receive AdapterCtx (now_ms(), emit_event(), rng, config)
    Byte mode: minimal handshake + SDU pass-through mock (no crypto)
    Audio mode: uses AudioStack (8k/160) and emits text/log/metric events
    """


    def __init__(self):
        self.cfg: dict = {}
        self.ctx: Optional[Any] = None
        self.mode: str = "byte"
        self.side: str = "L"
        self.crypto: dict = {}
        # ByteLink state (mock)
        self._byte_started: bool = False
        self._byte_done: bool = False
        self._byte_t_ms: int = 0
        self._byte_txq: Deque[Tuple[bytes, int]] = deque()
        # Audio state
        self._audio_stack: Optional[AudioStack] = None
        self._noise: Optional[NoiseXKWrapper] = None
        self._handshake_started = False
        self._pending_handshake = True

        # TX queue for encrypted SDUs (Noise inside Audio)
        self._audio_tx_sdu_q: Deque[bytes] = deque()


    # ---- capabilities (runner les lit avant de démarrer l’I/O) ----
    def nade_capabilities(self) -> dict:
        return {
            "abi_version": ABI_VERSION,
            "bytelink": True,
            "audioblock": True,
            "sdu_max_bytes": SDU_MAX_BYTES,
            "audioparams": {"sr": 8000, "block": 160},
        }

    # ---- lifecycle ----
    def init(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.side = self.cfg.get("side", "L")
        self.mode = self.cfg.get("mode", "audio")
        self.crypto = self.cfg.get("crypto", {}) or {}
        # Base config plus optional env override: NADE_ADAPTER_CFG='{"modem":"bfsk","modem_cfg":{...}}'
        self.nade_cfg = _merged_nade_cfg(self.cfg.get("nade", {}) or {})

    def start(self, ctx):
        self.ctx = ctx

        if self.mode == "audio":

            self._audio_logger("info", f"[Adapter] start() side={self.side} mode=audio")

            modem_cfg = (self.nade_cfg or {}).get("modem_cfg") or {}
            modem_name = (self.nade_cfg or {}).get("modem", "bfsk")

            self._audio_logger("info", f"[Adapter] Creating AudioStack modem={modem_name} cfg={modem_cfg}")

            self._audio_stack = AudioStack(
                modem=modem_name,
                modem_cfg=modem_cfg,
                logger=self._audio_logger,
            )

            # ------- Noise config logging --------
            priv_raw = self.crypto.get("priv")
            pub_raw = self.crypto.get("pub")
            peer_pub_raw = self.crypto.get("peer_pub")

            self._audio_logger("info", f"[Noise] Local pub={pub_raw.hex()} peer_pub={peer_pub_raw.hex() if peer_pub_raw else None}")

            # ------- Keypair setup ----------
            local_kp = KeyPair(PublicKey(bytes(pub_raw)), PrivateKey(bytes(priv_raw)))
            peer_pub_obj = PublicKey(bytes(peer_pub_raw)) if peer_pub_raw else None

            self._noise = NoiseXKWrapper(
                keypair=local_kp,
                peer_pubkey=peer_pub_obj,
                debug_callback=lambda msg: self._audio_logger("debug", "[Noise dbg] " + msg),
            )

        else:
            self._audio_logger("info", "[Adapter] start() mode=byte")
            self._byte_started = False
            self._byte_done = False
            self._byte_txq.clear()

    def stop(self) -> None:
        pass

    # ---- timers ----
    def on_timer(self, t_ms: int) -> None:
        if self.mode != "audio":
            # ByteLink mock timer
            self._byte_t_ms = t_ms
            if not self._byte_started:
                self._byte_started = True
                if self.side == "L":  # initiator sends HS1
                    self._byte_txq.append((bytes([HANDSHAKE_TAG]) + b"HS1", self._byte_t_ms))
        return

    # ----------------------------------------------------------------------
    # BYTELINK
    # ----------------------------------------------------------------------
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
        if sdu[0] == HANDSHAKE_TAG:
            hs = sdu[1:]
            if hs == b"HS1":
                if not self._byte_done:
                    self._byte_txq.append((bytes([HANDSHAKE_TAG]) + b"HS2", self._byte_t_ms))
                    self._byte_done = True
            elif hs == b"HS2":
                self._byte_done = True
            return
        if self._byte_done:
            # Plaintext passthrough; previously queued internally but never used externally
            # Keep behavior no-op for external surfaces
            pass

    # ----------------------------------------------------------------------
    # AUDIOBLOCK TX (device → modem → PCM)
    # ----------------------------------------------------------------------
    def pull_tx_block(self, t_ms):

        if not self._audio_stack:
            return np.zeros(160, dtype=np.int16) if np else None

        # ---- Handshake TX ----
        if self._noise and not self._noise.handshake_complete:
            hs_msg = self._noise.get_next_handshake_message()
            if hs_msg:
                self._audio_logger("info",
                    f"[NoiseXK] TX handshake msg len={len(hs_msg)} hex={hs_msg.hex()[:32]}...")
                self._audio_stack.tx_enqueue(hs_msg)
            else:
                self._audio_logger("debug", "[NoiseXK] No handshake message ready yet")

        # ---- Encrypted SDU TX ----
        if self._audio_tx_sdu_q:
            sdu = self._audio_tx_sdu_q.popleft()
            self._audio_logger("info",
                f"[Nade] TX encrypted SDU len={len(sdu)} hex={sdu.hex()[:32]}...")
            self._audio_stack.tx_enqueue(sdu)

        pcm_out = self._audio_stack.pull_tx_block(t_ms)

        # === Normalize PCM ===
        if pcm_out is not None and pcm_out.size > 0:
            max_abs = np.max(np.abs(pcm_out))
            if max_abs > 0:
                pcm_out = (pcm_out.astype(np.float32) / max_abs) * 32767
                pcm_out = pcm_out.astype(np.int16)

        return pcm_out
    
    # ----------------------------------------------------------------------
    # AUDIOBLOCK RX (PCM → modem → encrypted SDUs → decrypted SDUs)
    # ----------------------------------------------------------------------
    def push_rx_block(self, pcm, t_ms):

        if not self._audio_stack:
            self._audio_logger("debug",f"not self._audio_stack")
            return
        
        pcm_float = pcm.astype(np.float32) / 32767.0

        self._audio_stack.push_rx_block(pcm_float, t_ms)

        payloads = self._audio_stack.pop_rx_frames()

        if not payloads:
            self._audio_logger("debug",f"not payloads")
            return

        for frame in payloads:
            self._audio_logger("debug",f"frame in payloads")

            self._audio_logger(
                "info",
                f"[Adapter] RX frame len={len(frame)} hex={frame.hex()[:32]}..."
            )

            if not self._noise:
                self._audio_logger("error", "[Adapter] RX frame but _noise is None!")
                continue

            # -------- handshake path --------
            if self._pending_handshake:
                    self._pending_handshake = False
                    self._noise.start_handshake(initiator=(self.side == "L"))
                    self._audio_logger("info", f"[Noise] Starting NoiseXK handshake (initiator={self.side == 'L'})")
            elif not self._noise.handshake_complete:
                self._audio_logger("info", "[NoiseXK] RX handshake message")

                try:
                    self._noise.process_handshake_message(frame)
                except Exception as e:
                    self._audio_logger("error",
                        f"[NoiseXK] ERROR while processing handshake message: {e}")
                    continue

                self._audio_logger("info",
                    f"[NoiseXK] Handshake state: complete={self._noise.handshake_complete}")

                if self._noise.handshake_complete:
                    self._audio_logger("info", "[NoiseXK] Handshake COMPLETED")
                    self.ctx.emit_event("handshake_complete", {"side": self.side})

                continue

            # -------- encrypted SDU path --------
            try:
                plaintext = self._noise.decrypt_sdu(b"", frame)
                self._audio_logger(
                    "info",
                    f"[Nade] RX decrypted SDU len={len(plaintext)} text={plaintext!r}"
                )
                self.ctx.emit_event("text_rx",
                    {"text": plaintext.decode("utf-8", errors="ignore")})
            except Exception as e:
                self._audio_logger("error",
                    f"[NoiseXK] decrypt_sdu FAILED: {e} (len={len(frame)})")
    
    # ----------------------------------------------------------------------
    # API: external system injects SDUs into Audio mode
    # ----------------------------------------------------------------------
    def send_sdu(self, data: bytes):

        if not self._noise or not self._noise.handshake_complete:
            self._audio_logger("warn",
                f"[Nade] Tried to send SDU but handshake is incomplete. Dropped.")
            return

        try:
            ct = self._noise.encrypt_sdu(b"", data)
        except Exception as e:
            self._audio_logger("error",
                f"[NoiseXK] encrypt_sdu FAILED: {e} plaintext={data!r}")
            return

        self._audio_logger(
            "info",
            f"[Nade] Queue TX SDU plaintext_len={len(data)} ct_len={len(ct)}"
        )

        self._audio_tx_sdu_q.append(ct)

    
    # ----------------------------------------------------------------------
    # Logger
    # ----------------------------------------------------------------------
    def _audio_logger(self, level: str, payload: Any) -> None:
        if level == "metric" and isinstance(payload, dict):
            self.ctx.emit_event("metric", payload)
        elif isinstance(payload, str):
            self.ctx.emit_event("log", {"level": "info", "msg": payload})
        else:
            self.ctx.emit_event("log", {"level": str(level), "msg": str(payload)})
