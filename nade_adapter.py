from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from mode_a.mode_a import NadeByteLink
from typing import Optional

ABI_VERSION = "1.0"

def nade_capabilities():
    return {
        "abi_version": ABI_VERSION,
        "bytelink": True,
        "audioblock": False,
        "audioparams": {"sr": 8000, "block": 160},
    }


class Adapter:
    """Thin adapter for test harness / DryBox."""

    def __init__(self, side: str, keypair: KeyPair, peer_pub: Optional[PublicKey] = None):
        self.side = side
        self.ctx = None

        self.local_kp = keypair
        self.peer_pub = peer_pub

        # Debug print
        local_pub_bytes = self.local_kp.public.data
        peer_pub_bytes = self.peer_pub.data if self.peer_pub else None
        print(f"[Adapter:{self.side}] using local_pub={local_pub_bytes.hex()} "
              f"peer_pub={(peer_pub_bytes.hex() if peer_pub_bytes else '<none>')}")

        scenario_mode = self.get_scenario_mode()
        if scenario_mode == "byte":
            self.nade = NadeByteLink(
                side=self.side,
                local_kp=self.local_kp,
                peer_pub=self.peer_pub,
                debug=print
            )
        elif scenario_mode == "block":
            raise NotImplementedError("Block mode not yet implemented")
        else:
            raise ValueError(f"Unsupported mode from scenario: {scenario_mode}")

    def get_scenario_mode(self) -> str:
        """Retrieve the mode from scenario configuration (cfg or YAML)."""
        if self.ctx and hasattr(self.ctx, "cfg") and "mode" in self.ctx.cfg:
            return self.ctx.cfg["mode"]
        return "byte"  # default fallback

    def start(self, ctx):
        self.ctx = ctx

    def on_timer(self, t_ms: int):
        self.nade.on_timer(t_ms)

    def poll_link_tx(self, budget: int):
        return self.nade.poll_link_tx(budget)

    def on_link_rx(self, sdu: bytes, t_ms: int):
        self.nade.on_link_rx(sdu, t_ms)
