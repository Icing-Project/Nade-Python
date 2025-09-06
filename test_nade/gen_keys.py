# gen_keys.py
import yaml
from dissononce.dh.x25519.x25519 import X25519DH

def generate_keypair_hex():
    dh = X25519DH()
    kp = dh.generate_keypair()
    # kp.private.data and kp.public.data are bytes
    priv_hex = kp.private.data.hex()
    pub_hex = kp.public.data.hex()
    return priv_hex, pub_hex

def main():
    left_priv, left_pub = generate_keypair_hex()
    right_priv, right_pub = generate_keypair_hex()

    scenario = {
        "mode": "byte",
        "name": "auto_generated_mode_a",
        "duration_ms": 5000,
        "bearer": {"mtu": 256, "type": "loopback"},
        "left_nade": {
            "mode": "byte",
            "side": "left",
            "static_key": left_priv,
            "static_pub": left_pub,
            "peer_pub": right_pub,
        },
        "right_nade": {
            "mode": "byte",
            "side": "right",
            "static_key": right_priv,
            "static_pub": right_pub,
            "peer_pub": left_pub,
        },
    }

    with open("scenario.yaml", "w") as f:
        yaml.safe_dump(scenario, f)

    print("Wrote scenario.yaml with fresh keypairs (left & right).")
    print("Left pub:", left_pub)
    print("Right pub:", right_pub)

if __name__ == "__main__":
    main()
