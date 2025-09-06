# test_mode_a.py
import sys, os, yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nade_adapter import Adapter
from dissononce.dh.x25519.public import PublicKey

TICK_MS = 10

def main():
    with open("scenario.yaml") as f:
        scenario = yaml.safe_load(f)

    left = Adapter(scenario["left_nade"])
    right = Adapter(scenario["right_nade"])

    # If scenario supplied peer_pub bytes they were used;
    # for local testing we'll ensure peer_pub objects reference
    # the actual in-memory KeyPairs so Dissononce sees the exact same bytes.
    # The adapters expose .nade.local_kp (dissononce KeyPair) and .nade.peer_pub.
    # If peer_pub is missing / mismatched, override using the real KeyPair.public.

    # Set peer on left to right's public (typed object)
    try:
        left_pub_obj = right.nade.local_kp.public  # dissononce PublicKey object
        right_pub_obj = left.nade.local_kp.public
        left.nade.peer_pub = left_pub_obj
        right.nade.peer_pub = right_pub_obj

        # Make the Noise wrapper use the same PublicKey object
        if hasattr(left.nade, "noise"):
            left.nade.noise.peer_pubkey = left_pub_obj
        if hasattr(right.nade, "noise"):
            right.nade.noise.peer_pubkey = right_pub_obj

        print("[test] wired peer_pub from in-memory KeyPairs for deterministic local test")
    except Exception as e:
        print("[test] could not wire peer pubs automatically:", e)

    class DummyCtx:
        def __init__(self, name):
            self.name = name
        def deliver(self, sdu, t_ms):
            print(f"[{self.name}] delivered SDU @ {t_ms}ms: {sdu.hex()}")

    left.start(DummyCtx("left"))
    right.start(DummyCtx("right"))

    # Run simulation
    for t in range(0, scenario["duration_ms"], TICK_MS):
        left.on_timer(t)
        right.on_timer(t)

        # left -> right
        for sdu, _ts in left.poll_link_tx(16):
            right.on_link_rx(sdu, t)

        # right -> left
        for sdu, _ts in right.poll_link_tx(16):
            left.on_link_rx(sdu, t)

        # Application-level messages
        left_msgs = left.nade.app_recv_plain_all()
        right_msgs = right.nade.app_recv_plain_all()
        for m in left_msgs:
            print("[app left] got:", m)
        for m in right_msgs:
            print("[app right] got:", m)

    print("done")

if __name__ == "__main__":
    main()
