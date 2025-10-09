# nade/adapter/drybox_adapter.py
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


class _MockByteLink:
    """
    ByteLink mock endpoint.
    - Simple 2-way handshake (TAG + "HS1"/"HS2") pour synchroniser les deux bouts.
    - Pas de chiffrement réel (pass-through). Sert de squelette pour brancher Noise plus tard.
    - API DryBox attendue:
        on_timer(t_ms)
        poll_link_tx(budget) -> List[bytes] | List[Tuple[bytes,int]]
        on_link_rx(sdu: bytes)
    """

    def __init__(self, side: str, key_id: Optional[str] = None, peer_key_id: Optional[str] = None):
        # Runner fournit "L" ou "R"
        self.side = side  # "L"/"R"
        self.is_initiator = (self.side == "L")  # convention: L initie
        self.key_id = key_id
        self.peer_key_id = peer_key_id

        self._t_ms = 0
        self._started = False
        self._done = False

        self._txq: Deque[Tuple[bytes, int]] = deque()
        self._rx_plain: Deque[bytes] = deque()

    # ---- DryBox hooks ----
    def on_timer(self, t_ms: int) -> None:
        self._t_ms = t_ms
        if not self._started:
            self._started = True
            # L initie: envoie HS1
            if self.is_initiator:
                payload = b"HS1"
                self._txq.append((bytes([HANDSHAKE_TAG]) + payload, self._t_ms))

    def poll_link_tx(self, budget: int) -> List[Tuple[bytes, int]]:
        out: List[Tuple[bytes, int]] = []
        while self._txq and len(out) < budget:
            out.append(self._txq.popleft())
        return out

    def on_link_rx(self, sdu: bytes) -> None:
        if not sdu:
            return
        # Handshake message?
        if sdu[0] == HANDSHAKE_TAG:
            hs = sdu[1:]
            if hs == b"HS1":
                # Répond HS2 côté récepteur; handshake complet des deux côtés
                if not self._done:
                    self._txq.append((bytes([HANDSHAKE_TAG]) + b"HS2", self._t_ms))
                    self._done = True
            elif hs == b"HS2":
                # Initiateur reçoit l'ACK → done
                self._done = True
            return

        # Donnée normale (pass-through plaintext)
        if self._done:
            self._rx_plain.append(sdu)

    # ---- Helpers applicatifs (si nécessaire plus tard) ----
    def app_send_plain(self, data: bytes) -> None:
        if not self._done:
            return  # ignorer tant que handshake non terminé
        # Pas de chiffrement → SDU = data (tu brancheras Noise ici ensuite)
        self._txq.append((data, self._t_ms))

    def app_recv_all(self) -> List[bytes]:
        items = list(self._rx_plain)
        self._rx_plain.clear()
        return items


class _NadeAudioPort:
    SR = 8000
    BLK = 160

    def __init__(self, emit_event, nade_cfg: dict | None = None):
        self.emit_event = emit_event  # emit_event(type, payload)
        modem_cfg = (nade_cfg or {}).get("modem_cfg") or {}
        # Use a conservative default (bfsk @ 80 sps) for initial audio smoke tests.
        # Can be overridden via nade_cfg["modem"].
        modem_name = (nade_cfg or {}).get("modem", "bfsk")
        self.stack = AudioStack(modem=modem_name, modem_cfg=modem_cfg, logger=self._logger)

    def _logger(self, level: str, payload: Any):
        # Unifie: niveau texte -> "log", objets -> "metric"
        if level == "metric" and isinstance(payload, dict):
            self.emit_event("metric", payload)  # le runner peut mapper vers metrics.csv
        elif isinstance(payload, str):
            self.emit_event("log", {"level": "info", "msg": payload})
        else:
            self.emit_event("log", {"level": str(level), "msg": str(payload)})

    def pull_tx_block(self, t_ms: int):
        blk = self.stack.pull_tx_block(t_ms)
        # capture optionnelle (ex: symboles non dispo → on garde PCM)
        # self.emit_event("capture", {"layer": "bearer", "event": "tx", "pcm_i16_le": blk.tobytes().hex()})
        return blk

    def push_rx_block(self, pcm, t_ms: int) -> None:
        self.stack.push_rx_block(pcm, t_ms)
        texts = self.stack.pop_received_texts()
        for txt in texts:
            # Deux events: un log humain + un event structuré "text_rx"
            self.emit_event("log", {"level": "info", "msg": f"[Nade] RX TEXT: {txt}"})
            self.emit_event("text_rx", {"text": txt})

    def on_timer(self, t_ms: int) -> None:
        self.stack.on_timer(t_ms)

    # Contrôle runtime: reconfig/modem, injection texte, etc.
    def control(self, cmd: dict) -> None:
        if "send_text" in cmd:
            self.stack.queue_text(str(cmd["send_text"]))
            self.emit_event("text_tx", {"text": str(cmd["send_text"])})
        if "modem_cfg" in cmd:
            self.stack.reconfigure(modem_cfg=cmd["modem_cfg"])
            self.emit_event("log", {"level": "info", "msg": f"Audio modem reconfigured: {cmd['modem_cfg']}"})


class Adapter:
    """
    DryBox adapter (v1) pour Nade.
    Compatible avec le Runner:
      - __init__() sans arguments
      - init(cfg: dict)  ← reçoit side/mode/crypto/...
      - start(ctx)       ← reçoit AdapterCtx (now_ms(), emit_event(), rng, config)
      - stop()
      - Byte mode: on_timer, poll_link_tx, on_link_rx
      - Audio mode: pull_tx_block, push_rx_block
    """

    def __init__(self):
        self.cfg: dict = {}
        self.ctx: Optional[Any] = None
        self.mode: str = "byte"
        self.side: str = "L"  # "L"/"R" fourni par runner
        self.crypto: dict = {}
        self._byte: Optional[_MockByteLink] = None
        self._audio: Optional[_NadeAudioPort] = None

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
        # Base config (can be provided by the runner in future)
        nade_cfg = self.cfg.get("nade", {}) or {}
        # Optional env override for quick local sweeps: NADE_ADAPTER_CFG='{"modem":"bfsk","modem_cfg":{...}}'
        try:
            env_cfg = os.environ.get("NADE_ADAPTER_CFG")
            if env_cfg:
                nade_cfg_env = json.loads(env_cfg)
                if isinstance(nade_cfg_env, dict):
                    nade_cfg.update(nade_cfg_env)
        except Exception:
            pass
        self.nade_cfg = nade_cfg

    def start(self, ctx: Any) -> None:
        self.ctx = ctx
        if self.mode == "audio":
            self._audio = _NadeAudioPort(self.ctx.emit_event, self.nade_cfg)
            # Option demo: auto-send message if provided in config
            auto = (self.nade_cfg or {}).get("auto_text")
            msg = None
            if auto and isinstance(auto, str):
                msg = auto
            else:
                msg = "Hello from Nade-Python Audio mode!"
            self._audio.control({"send_text": msg})
        else:
            self._byte = _MockByteLink(
                side=self.side,
                key_id=self.crypto.get("key_id"),
                peer_key_id=self.crypto.get("peer_key_id"),
            )

    def stop(self) -> None:
        pass

    # ---- timers ----
    def on_timer(self, t_ms: int) -> None:
        if self._byte:
            self._byte.on_timer(t_ms)
        if self._audio:
            self._audio.on_timer(t_ms)

    # ---- ByteLink I/O ----
    def poll_link_tx(self, budget: int):
        if not self._byte:
            return []
        return self._byte.poll_link_tx(budget)

    def on_link_rx(self, sdu: bytes):
        if self._byte:
            self._byte.on_link_rx(sdu)

    # ---- AudioBlock I/O ----
    def pull_tx_block(self, t_ms: int):
        if not self._audio:
            # fallback silencieux si audio non actif
            if np is None:
                return None
            return np.zeros(160, dtype=np.int16)
        return self._audio.pull_tx_block(t_ms)

    def push_rx_block(self, pcm, t_ms: int):
        if self._audio:
            self._audio.push_rx_block(pcm, t_ms)
