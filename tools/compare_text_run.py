import json
import sys
from pathlib import Path
from difflib import SequenceMatcher


def load_events(path: Path):
    tx = []
    rx = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            ev = json.loads(line)
            typ = ev.get("type")
            payload = ev.get("payload", {})
            if typ == "text_tx":
                tx.append((ev.get("t_ms", 0), ev.get("side", "?"), payload.get("text", "")))
            elif typ == "text_rx":
                rx.append((ev.get("t_ms", 0), ev.get("side", "?"), payload.get("text", "")))
    return tx, rx


def compare(tx_list, rx_list):
    # Build by-side maps of the last text
    out = []
    for side in ("L", "R"):
        tx_texts = [t for (_, s, t) in tx_list if s == side]
        rx_texts = [t for (_, s, t) in rx_list if s == side]
        tx_text = tx_texts[-1] if tx_texts else ""
        rx_text = rx_texts[-1] if rx_texts else ""
        ratio = SequenceMatcher(a=tx_text, b=rx_text).ratio() if rx_text else 0.0
        out.append((side, tx_text, rx_text, ratio))
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/compare_text_run.py <events.jsonl>")
        sys.exit(2)
    path = Path(sys.argv[1])
    tx, rx = load_events(path)
    res = compare(tx, rx)
    for side, tx_text, rx_text, ratio in res:
        print(f"Side {side}:")
        print(f"  TX: {tx_text}")
        print(f"  RX: {rx_text}")
        print(f"  Similarity: {ratio:.3f}")


if __name__ == "__main__":
    main()

