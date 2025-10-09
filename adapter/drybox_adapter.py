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


def _auto_text(nade_cfg: dict | None) -> str:
    """Return demo text, unchanged behavior (prefer nade_cfg['auto_text'] if str)."""
    auto = (nade_cfg or {}).get("auto_text")
    return auto if isinstance(auto, str) else "Hello from Nade-Python Audio mode!"


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
        self.side: str = "L"  # "L"/"R" fourni par runner
        self.crypto: dict = {}
        # ByteLink state (mock)
        self._byte_started: bool = False
        self._byte_done: bool = False
        self._byte_t_ms: int = 0
        self._byte_txq: Deque[Tuple[bytes, int]] = deque()
        # Audio state
        self._audio_stack: Optional[AudioStack] = None

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

    def start(self, ctx: Any) -> None:
        self.ctx = ctx
        if self.mode == "audio":
            modem_cfg = (self.nade_cfg or {}).get("modem_cfg") or {}
            modem_name = (self.nade_cfg or {}).get("modem", "bfsk")
            self._audio_stack = AudioStack(modem=modem_name, modem_cfg=modem_cfg, logger=self._audio_logger)
            # auto-send demo text
            msg = _auto_text(self.nade_cfg)
            self._audio_stack.queue_text(msg)
            self.ctx.emit_event("text_tx", {"text": msg})
        else:
            # reset byte-link state
            self._byte_started = False
            self._byte_done = False
            self._byte_t_ms = 0
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
        else:
            if self._audio_stack is not None:
                self._audio_stack.on_timer(t_ms)

    # ---- ByteLink I/O ----
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

    # ---- AudioBlock I/O ----
    def pull_tx_block(self, t_ms: int):
        if self._audio_stack is not None:
            blk = self._audio_stack.pull_tx_block(t_ms)
            # Optional capture hook kept disabled (behavior unchanged)
            return blk
        # silent fallback if audio not active
        if np is None:
            return None
        return np.zeros(160, dtype=np.int16)

    def push_rx_block(self, pcm, t_ms: int):
        if self._audio_stack is None:
            return
        self._audio_stack.push_rx_block(pcm, t_ms)
        texts = self._audio_stack.pop_received_texts()
        for txt in texts:
            self.ctx.emit_event("log", {"level": "info", "msg": f"[Nade] RX TEXT: {txt}"})
            self.ctx.emit_event("text_rx", {"text": txt})

    # ---- logger for AudioStack ----
    def _audio_logger(self, level: str, payload: Any) -> None:
        if level == "metric" and isinstance(payload, dict):
            self.ctx.emit_event("metric", payload)
        elif isinstance(payload, str):
            self.ctx.emit_event("log", {"level": "info", "msg": payload})
        else:
            self.ctx.emit_event("log", {"level": str(level), "msg": str(payload)})
