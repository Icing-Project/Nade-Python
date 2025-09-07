# nade_adapter.py
from typing import Optional, Any
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from nade.mode_a import NadeByteLink
from nade.mode_b import NadeAudioPort

ABI_VERSION = "1.0"


def nade_capabilities():
    return {
        "abi_version": ABI_VERSION,
        "bytelink": True,
        "audioblock": True,
        "audioparams": {"sr": 8000, "block": 160},
    }


class Adapter:
    """Thin adapter for test harness / DryBox."""

    def __init__(self, side: str, keypair: KeyPair, peer_pub: Optional[PublicKey] = None):
        self.side = side
        self.local_kp = keypair
        self.peer_pub = peer_pub
        self.ctx: Optional[Any] = None
        self.nade = None

        # Debug print (best-effort access to typed PublicKey)
        try:
            local_pub_bytes = self.local_kp.public.data
        except Exception:
            local_pub_bytes = bytes(self.local_kp.public)
        try:
            peer_pub_bytes = self.peer_pub.data if self.peer_pub is not None else None
        except Exception:
            peer_pub_bytes = bytes(self.peer_pub) if self.peer_pub is not None else None

        print(
            f"[Adapter:{self.side}] using local_pub={local_pub_bytes.hex()} "
            f"peer_pub={(peer_pub_bytes.hex() if peer_pub_bytes else '<none>')}"
        )

    def start(self, ctx: Optional[Any]):
        """
        DryBox calls this with scenario context (ctx.cfg expected but optional).
        Accepts contexts that don't expose .cfg (e.g. simple DummyCtx used in tests).
        """
        self.ctx = ctx

        # Read mode from ctx.cfg safely — support None ctx or ctx without cfg
        cfg = {}
        if ctx is not None:
            # ctx.cfg might be a dict-like or attribute
            cfg = getattr(ctx, "cfg", cfg) or cfg

        scenario_mode = cfg.get("mode", "byte")

        if scenario_mode == "byte":
            self.nade = NadeByteLink(
                side=self.side,
                local_kp=self.local_kp,
                peer_pub=self.peer_pub,
                debug=print,
            )
        elif scenario_mode == "block":
            self.nade = NadeAudioPort(
                side=self.side,
                local_kp=self.local_kp,
                peer_pub=self.peer_pub,
                debug=print,
            )
        else:
            raise ValueError(f"Unsupported mode from scenario: {scenario_mode}")

    # ---------------- passthroughs ----------------
    def on_timer(self, t_ms: int):
        if self.nade is None:
            return
        self.nade.on_timer(t_ms)

    def poll_link_tx(self, budget: int):
        if self.nade is None:
            return []
        # Byte mode only — audio/block mode will not use this path
        return getattr(self.nade, "poll_link_tx", lambda b: [])(budget)

    def on_link_rx(self, sdu: bytes, t_ms: int):
        if self.nade is None:
            return
        fn = getattr(self.nade, "on_link_rx", None)
        if fn is not None:
            fn(sdu, t_ms)

    # Audio mode passthroughs (DryBox will call these when in block mode)
    def pull_tx_block(self, t_ms: int):
        if self.nade is None:
            # return silence by default (160 samples @ 8kHz)
            try:
                import numpy as _np
                return _np.zeros(160, dtype=_np.int16)
            except Exception:
                return None
        fn = getattr(self.nade, "pull_tx_block", None)
        if fn is None:
            # Not an audio mode endpoint — return silence
            import numpy as _np
            return _np.zeros(160, dtype=_np.int16)
        return fn(t_ms)

    def push_rx_block(self, pcm: "np.ndarray[int16]", t_ms: int):
        if self.nade is None:
            return
        fn = getattr(self.nade, "push_rx_block", None)
        if fn is None:
            return
        return fn(pcm, t_ms)
