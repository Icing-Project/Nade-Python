# nade_adapter.py
from typing import Optional
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.dh.x25519.private import PrivateKey
from dissononce.dh.x25519.public import PublicKey
from mode_a.mode_a import NadeByteLink

class Adapter:
    def __init__(self, cfg):
        # set side early
        self.side = cfg.get("side", "left")
        self.ctx = None
        self.mode = cfg.get("mode", "byte")

        if self.mode != "byte":
            raise ValueError(f"Unsupported mode: {self.mode}")

        dh = X25519DH()

        # Try to load KeyPair from scenario if provided
        hex_priv = cfg.get("static_key")
        hex_pub = cfg.get("static_pub")

        local_kp: Optional[KeyPair] = None

        if hex_priv and hex_pub:
            try:
                sk_bytes = bytes.fromhex(hex_priv)
                pk_bytes = bytes.fromhex(hex_pub)

                # Wrap into typed objects (PrivateKey/PublicKey) then KeyPair
                local_priv = PrivateKey(sk_bytes)
                local_pub = PublicKey(pk_bytes)
                tentative_kp = KeyPair(local_priv, local_pub)

                # Try to read back the public bytes from KeyPair to confirm match
                try:
                    actual_pub = tentative_kp.public.data
                except Exception:
                    # fallback to bytes() if .data not available
                    actual_pub = bytes(tentative_kp.public)

                if actual_pub.hex().lower() != pk_bytes.hex().lower():
                    print(f"[Adapter:{self.side}] WARNING: provided static_pub != tentative KeyPair.public (mismatch). Falling back to freshly generated KeyPair.")
                    local_kp = None
                else:
                    local_kp = tentative_kp
            except Exception as e:
                print(f"[Adapter:{self.side}] WARNING: failed to construct KeyPair from scenario keys: {e}. Falling back to freshly generated KeyPair.")
                local_kp = None

        # If we didn't get a valid KeyPair from scenario, generate one (reliable)
        if local_kp is None:
            local_kp = dh.generate_keypair()
            print(f"[Adapter:{self.side}] Info: generated fresh KeyPair for local testing (no reliable scenario KeyPair).")

        self.local_kp = local_kp

        # Peer public key from scenario (optional) — may be overridden by test harness wiring
        hex_peer = cfg.get("peer_pub")
        if hex_peer:
            try:
                peer_bytes = bytes.fromhex(hex_peer)
                self.peer_pub = PublicKey(peer_bytes)
            except Exception:
                print(f"[Adapter:{self.side}] WARNING: invalid peer_pub in scenario; ignoring.")
                self.peer_pub = None
        else:
            self.peer_pub = None

        # Debug info: show the final local/public in use (read from the typed KeyPair)
        try:
            local_pub_bytes = self.local_kp.public.data
        except Exception:
            local_pub_bytes = bytes(self.local_kp.public)
        peer_dbg = getattr(self.peer_pub, "data", None) or (bytes(self.peer_pub) if self.peer_pub is not None else None)
        print(f"[Adapter:{self.side}] using local_pub={local_pub_bytes.hex()} peer_pub={(peer_dbg.hex() if peer_dbg else '<none>')}")

        # instantiate NadeByteLink (pass typed KeyPair & PublicKey if present)
        self.nade = NadeByteLink(
            side=self.side,
            local_kp=self.local_kp,
            peer_pub=self.peer_pub,
            debug=print
        )

    def start(self, ctx):
        self.ctx = ctx

    def on_timer(self, t_ms: int):
        self.nade.on_timer(t_ms)

    def poll_link_tx(self, budget: int):
        return self.nade.poll_link_tx(budget)

    def on_link_rx(self, sdu: bytes, t_ms: int):
        self.nade.on_link_rx(sdu, t_ms)
