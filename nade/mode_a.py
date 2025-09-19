# nade/mode_a.py
from collections import deque
from typing import Deque, List, Tuple, Optional
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.dh.x25519.public import PublicKey

from .noise_wrapper import NoiseXKWrapper


class NadeByteLink:
    HANDSHAKE_TAG = 0x20
    REKEY_PERIOD_MS = 60_000
    RX_WINDOW_SIZE = 128

    def __init__(
        self,
        side: str,
        local_kp: Optional[KeyPair] = None,
        peer_pub: Optional[PublicKey] = None,
        debug=None,
    ):
        self.side = side
        self.debug = debug or (lambda _m: None)

        dh = X25519DH()
        self.local_kp = local_kp or dh.generate_keypair()
        self.peer_pub = peer_pub

        self.is_initiator = (self.side == "left")
        self.noise = NoiseXKWrapper(self.local_kp, self.peer_pub, self.debug)

        # handshake state
        self.handshake_started = False
        self.handshake_done = False

        # TX/RX state
        self.tx_seq = 0
        self.rx_seen = set()
        self.rx_order: Deque[int] = deque(maxlen=self.RX_WINDOW_SIZE)
        self.txq: Deque[Tuple[bytes, int]] = deque()
        self.t_ms_now = 0
        self.last_rekey_ms = 0
        self.rx_plain: Deque[bytes] = deque()

    # ---------------- DryBox API ----------------

    def on_timer(self, t_ms: int) -> None:
        self.t_ms_now = t_ms

        if not self.handshake_started:
            self._start_initial_handshake()
            self._flush_pending_hs()

        if self.handshake_done and (t_ms - self.last_rekey_ms >= self.REKEY_PERIOD_MS):
            self.noise.begin_rekey()
            self.handshake_done = False
            self._flush_pending_hs()
            self.last_rekey_ms = t_ms

    def poll_link_tx(self, budget: int) -> List[Tuple[bytes, int]]:
        out: List[Tuple[bytes, int]] = []
        self._drain_pending_hs(out, budget)

        while self.txq and len(out) < budget:
            out.append(self.txq.popleft())

        return out

    def on_link_rx(self, sdu: bytes, t_ms: int) -> None:
        if sdu and sdu[0] == self.HANDSHAKE_TAG:
            self.noise.process_handshake_message(sdu[1:])
            self._flush_pending_hs()
            if self.noise.handshake_complete and not self.handshake_done:
                self.handshake_done = True
                self.last_rekey_ms = t_ms
            return

        if not self.handshake_done or len(sdu) < 4:
            return

        seq = int.from_bytes(sdu[:4], "big")
        ct = sdu[4:]

        if seq in self.rx_seen:
            return
        self.rx_seen.add(seq)
        self.rx_order.append(seq)
        if len(self.rx_order) == self.rx_order.maxlen:
            self.rx_seen.intersection_update(self.rx_order)

        ad = seq.to_bytes(4, "big")
        try:
            pt = self.noise.decrypt_sdu(ad, ct)
            self.rx_plain.append(pt)
        except Exception:
            return

    # ------------- Application helpers -------------

    def app_send_plain_sdu(self, plaintext: bytes) -> None:
        if not self.handshake_done:
            raise RuntimeError("Handshake not complete")
        seq = self.tx_seq
        ad = seq.to_bytes(4, "big")
        ct = self.noise.encrypt_sdu(ad, plaintext)
        self.txq.append((ad + ct, self.t_ms_now))
        self.tx_seq += 1

    def app_recv_plain_all(self) -> List[bytes]:
        out = list(self.rx_plain)
        self.rx_plain.clear()
        return out

    # ------------- Internal -------------

    def _start_initial_handshake(self):
        self.handshake_started = True
        self.noise.start_handshake(self.is_initiator)

    def _flush_pending_hs(self):
        while True:
            msg = self.noise.get_next_handshake_message()
            if not msg:
                break
            self.txq.append((bytes([self.HANDSHAKE_TAG]) + msg, self.t_ms_now))

    def _drain_pending_hs(self, out_list: List[Tuple[bytes, int]], budget: int):
        # Messages already queued in txq; rely on ordering
        pass
