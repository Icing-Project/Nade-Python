# nade/adapter/drybox_adapter.py
from typing import Optional, Any, List, Tuple
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.x25519.private import PrivateKey
from nade.modes.mode_a import NadeByteLink
from nade.modes.mode_b import NadeAudioPort
import numpy as np

ABI_VERSION = "dbx-v1"

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
        self.side: str = "L"
        self.crypto: dict = {}
        self.nade: Optional[Any] = None

    def nade_capabilities(self) -> dict:
        return {
            "abi_version": ABI_VERSION,
            "bytelink": True,
            "audioblock": True,
            "audioparams": {"sr": 8000, "block": 160},
            "sdu_max_bytes": 1024,
        }

    def init(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.side = self.cfg.get("side", "L")
        self.mode = self.cfg.get("mode", "audio")
        self.crypto = self.cfg.get("crypto", {}) or {}
        
        # Parse crypto (aligned with runner's l_crypto/r_crypto)
        crypto = self.crypto
        local_priv = crypto.get("priv")
        local_pub = crypto.get("pub")
        peer_pub_bytes = crypto.get("peer_pub")
        
        self.local_kp = KeyPair(PublicKey(local_pub), PrivateKey(local_priv)) if local_priv and local_pub else None
        self.peer_pub = PublicKey(peer_pub_bytes) if peer_pub_bytes else None
        
        # Debug print
        local_pub_bytes = self.local_kp.public.data if self.local_kp else None
        peer_pub_bytes = self.peer_pub.data if self.peer_pub else None
        print(
            f"[Adapter:{self.side}] using local_pub={(local_pub_bytes.hex() if local_pub_bytes else '<none>')} "
            f"peer_pub={(peer_pub_bytes.hex() if peer_pub_bytes else '<none>')}"
        )

    def start(self, ctx: Any) -> None:
        self.ctx = ctx
        if self.mode == "audio":
            self.nade = NadeAudioPort(
                side=self.side,
                local_kp=self.local_kp,
                peer_pub=self.peer_pub,
                debug=lambda *args: self.ctx.emit_event("log", {"level": "info", "msg": str(args)})
            )
            # Demo: Send a test message
            self.nade.control({"send_text": "Hello from Nade-Python Audio mode!"})
        else:
            self.nade = NadeByteLink(
                side=self.side,
                local_kp=self.local_kp,
                peer_pub=self.peer_pub,
                debug=lambda *args: self.ctx.emit_event("log", {"level": "info", "msg": str(args)})
            )

    def stop(self) -> None:
        self.nade = None

    def on_timer(self, t_ms: int) -> None:
        if self.nade:
            self.nade.on_timer(t_ms)

    def poll_link_tx(self, budget: int) -> List[Tuple[bytes, int]]:
        if not self.nade or self.mode != "byte":
            return []
        return getattr(self.nade, "poll_link_tx", lambda b: [])(budget)

    def on_link_rx(self, sdu: bytes, t_ms: int) -> None:
        if self.nade and self.mode == "byte":
            fn = getattr(self.nade, "on_link_rx", None)
            if fn:
                fn(sdu, t_ms)

    def pull_tx_block(self, t_ms: int) -> Optional[np.ndarray]:
        if not self.nade or self.mode != "audio":
            return np.zeros(160, dtype=np.int16) if np else None
        return self.nade.pull_tx_block(t_ms)

    def push_rx_block(self, pcm: np.ndarray, t_ms: int) -> None:
        if self.nade and self.mode == "audio":
            fn = getattr(self.nade, "push_rx_block", None)
            if fn:
                fn(pcm, t_ms)