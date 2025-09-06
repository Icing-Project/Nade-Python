# test_mode_a.py
import sys, os, yaml, random, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nade_adapter import Adapter

# dissononce wrappers
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.dh.x25519.private import PrivateKey
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.keypair import KeyPair

# cryptography used only to compute the corresponding public key bytes from private bytes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization

TICK_MS = 10

def make_keypair_from_rng(dh: X25519DH, rng: random.Random) -> KeyPair:
    """
    Deterministic generation of a KeyPair from rng (seeded).
    We generate 32 random bytes (rng.randbytes), build a cryptography X25519PrivateKey
    to derive the public bytes (correct serialization), then wrap both into dissononce
    PrivateKey/PublicKey and KeyPair.
    """
    sk_bytes = rng.randbytes(32)
    # derive public bytes with cryptography (this ensures canonical raw bytes)
    priv_crypto = X25519PrivateKey.from_private_bytes(sk_bytes)
    pub_bytes = priv_crypto.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

    # wrap into dissononce typed objects
    local_priv = PrivateKey(sk_bytes)
    local_pub = PublicKey(pub_bytes)
    kp = KeyPair(local_priv, local_pub)
    return kp

def main():
    # read scenario for duration/seed/bearer etc.
    with open("scenario.yaml") as f:
        scenario = yaml.safe_load(f)

    seed = scenario.get("seed", None)
    if seed is None:
        raise RuntimeError("scenario.yaml must include a 'seed' integer for deterministic key generation")

    rng = random.Random(seed)
    dh = X25519DH()

    # make deterministic keypairs
    left_kp = make_keypair_from_rng(dh, rng)
    right_kp = make_keypair_from_rng(dh, rng)

    # instantiate adapters and pass typed KeyPair + PublicKey objects
    left = Adapter(side="left", keypair=left_kp, peer_pub=right_kp.public)
    right = Adapter(side="right", keypair=right_kp, peer_pub=left_kp.public)

    print("[test] wired peer_pub from in-memory KeyPairs (deterministic by seed)")

    # minimal dummy context (DryBox provides a ctx with metrics; not needed here)
    class DummyCtx:
        def __init__(self, name):
            self.name = name
        def deliver(self, sdu, t_ms):
            print(f"[{self.name}] delivered SDU @ {t_ms}ms: {sdu.hex()}")

    left.start(DummyCtx("left"))
    right.start(DummyCtx("right"))

    did_send_app = False
    # simulation loop
    for t in range(0, scenario["duration_ms"], TICK_MS):
        left.on_timer(t)
        right.on_timer(t)

        # left -> right
        for sdu, _ts in left.poll_link_tx(16):
            # In DryBox this would be enqueued through bearer (loss/jitter). Here direct:
            right.on_link_rx(sdu, t)

        # right -> left
        for sdu, _ts in right.poll_link_tx(16):
            left.on_link_rx(sdu, t)

        # once handshake complete on both sides, send a single app SDU
        if left.nade.handshake_done and right.nade.handshake_done and not did_send_app:
            print("[test] both handshakes complete — sending app SDU from left -> right")
            left.nade.app_send_plain_sdu(b"hello from left")
            did_send_app = True

        # collect app-layer plaintexts (decrypted by mode_a)
        left_msgs = left.nade.app_recv_plain_all()
        right_msgs = right.nade.app_recv_plain_all()
        for m in left_msgs:
            print("[app left] got:", m)
        for m in right_msgs:
            print("[app right] got:", m)

        # tiny sleep so logs are readable if you run interactively (optional)
        # time.sleep(0.0001)

    print("done")

if __name__ == "__main__":
    main()
