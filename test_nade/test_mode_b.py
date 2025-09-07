import numpy as np
import sys, os
from dissononce.dh.x25519.x25519 import X25519DH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nade_adapter import Adapter


def main():
    # Deterministic keypairs for left/right (so runs are reproducible)
    dh = X25519DH()
    left_kp = dh.generate_keypair()
    right_kp = dh.generate_keypair()

    # Wire peer pubs (in real DryBox, cfg provides this)
    left = Adapter("left", keypair=left_kp, peer_pub=right_kp.public)
    right = Adapter("right", keypair=right_kp, peer_pub=left_kp.public)

    # Force mode = "block" for testing
    ctx_left = type("Ctx", (), {"cfg": {"mode": "block"}})()
    ctx_right = type("Ctx", (), {"cfg": {"mode": "block"}})()

    left.start(ctx_left)
    right.start(ctx_right)


    # Simulated time loop
    t_ms = 0
    for _ in range(200):  # run 200 ticks
        t_ms += 20  # 20 ms per audio block @ 8 kHz (160 samples)

        left.on_timer(t_ms)
        right.on_timer(t_ms)

        # Transmit one block each side
        block_left = left.nade.pull_tx_block(t_ms)
        block_right = right.nade.pull_tx_block(t_ms)

        # Deliver to the other side (ideal channel: just copy arrays)
        right.nade.push_rx_block(np.copy(block_left), t_ms)
        left.nade.push_rx_block(np.copy(block_right), t_ms)

    print("done")


if __name__ == "__main__":
    main()
