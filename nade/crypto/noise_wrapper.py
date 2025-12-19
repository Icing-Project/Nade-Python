from typing import Optional, List, Callable, Tuple
from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.processing.impl.symmetricstate import SymmetricState
from dissononce.processing.impl.cipherstate import CipherState
from dissononce.processing.handshakepatterns.interactive.XK import XKHandshakePattern
from dissononce.cipher.chachapoly import ChaChaPolyCipher
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.hash.sha256 import SHA256Hash

# Typing alias
from dissononce.processing.impl.cipherstate import CipherState as _CipherStateType


def _hex(b: Optional[bytes]) -> str:
    if b is None:
        return "<none>"
    return b.hex()


class NoiseXKWrapper:
    """Noise XK wrapper using dissononce with fresh handshake per session."""

    def __init__(self, keypair: KeyPair, peer_pubkey: Optional[PublicKey] = None,
                 debug_callback: Optional[Callable[[str], None]] = None):
        self.keypair = keypair
        self.peer_pubkey = peer_pubkey
        self.debug = debug_callback or (lambda *a, **k: None)

        self._hs: Optional[HandshakeState] = None
        self._send_cs: Optional[_CipherStateType] = None
        self._recv_cs: Optional[_CipherStateType] = None
        self.handshake_complete = False
        self.is_initiator: Optional[bool] = None

        # queued handshake messages
        self.outgoing_messages: List[bytes] = []

    # ---------------- helpers ----------------
    def _new_handshakestate(self) -> HandshakeState:
        cipher = ChaChaPolyCipher()
        dh = X25519DH()
        hshash = SHA256Hash()
        symmetric = SymmetricState(CipherState(cipher), hshash)
        return HandshakeState(symmetric, dh)

    # ---------------- handshake control ----------------
    def start_handshake(self, initiator: bool):
        self.debug(f"[NoiseXK] start_handshake(initiator={initiator})")
        self.is_initiator = initiator
        self.handshake_complete = False
        self._send_cs = None
        self._recv_cs = None
        self.outgoing_messages.clear()

        self._hs = self._new_handshakestate()

        # Debug
        local_pub_b = self.keypair.public.data
        peer_pub_b = self.peer_pubkey.data if self.peer_pubkey else None
        self.debug(f"[NoiseXK] local_pub={_hex(local_pub_b)} remote_pub={_hex(peer_pub_b)}")

        if initiator:
            if self.peer_pubkey is None:
                raise ValueError("Initiator requires peer static public key")
            self._hs.initialize(XKHandshakePattern(), True, b'', s=self.keypair, rs=self.peer_pubkey)

            out = bytearray()
            cs_pair = self._hs.write_message(b'', out)
            if out:
                self.outgoing_messages.append(bytes(out))
                self.debug(f"[NoiseXK] queued initial M1 ({len(out)} bytes): {bytes(out).hex()}")
            if cs_pair:
                self._complete_handshake(cs_pair)

        else:
            self._hs.initialize(XKHandshakePattern(), False, b'', s=self.keypair)
            self.debug("[NoiseXK] responder initialized (waiting for M1)")

    def process_handshake_message(self, data: bytes):
        if self.handshake_complete:
            self.debug(f"[NoiseXK] ignoring handshake msg (already complete)")
            return

        if self._hs is None:
            # Responder initializes lazily
            self._hs = self._new_handshakestate()
            self._hs.initialize(XKHandshakePattern(), False, b'', s=self.keypair)
            self.debug("[NoiseXK] lazily initialized handshake state for responder")

        self.debug(f"[NoiseXK] processing HS msg ({len(data)} bytes): {data.hex()}")
        payload = bytearray()
        cs_pair_read = self._hs.read_message(data, payload)
        if cs_pair_read:
            self._complete_handshake(cs_pair_read)
            return

        # If a response is required, queue it
        out = bytearray()
        cs_pair_write = self._hs.write_message(b'', out)
        if out:
            self.outgoing_messages.append(bytes(out))
            self.debug(f"[NoiseXK] queued HS response ({len(out)} bytes): {bytes(out).hex()}")
        if cs_pair_write:
            self._complete_handshake(cs_pair_write)


    def get_next_handshake_message(self) -> Optional[bytes]:
        return self.outgoing_messages.pop(0) if self.outgoing_messages else None

    # ---------------- AEAD ----------------
    def encrypt_sdu(self, ad: bytes, plaintext: bytes) -> bytes:
        if not (self.handshake_complete and self._send_cs):
            raise RuntimeError("Noise not ready")
        return self._send_cs.encrypt_with_ad(ad, plaintext)

    def decrypt_sdu(self, ad: bytes, ciphertext: bytes) -> bytes:
        if not (self.handshake_complete and self._recv_cs):
            raise RuntimeError("Noise not ready")
        return self._recv_cs.decrypt_with_ad(ad, ciphertext)

    def _complete_handshake(self, cs_pair: Tuple[_CipherStateType, _CipherStateType]):
        cs0, cs1 = cs_pair
        if self.is_initiator:
            self._send_cs, self._recv_cs = cs0, cs1
        else:
            self._send_cs, self._recv_cs = cs1, cs0
        print(f"[NoiseXK] handshake complete; secure channel ready")
        self.handshake_complete = True
        self.debug("[NoiseXK] handshake complete; secure channel ready")

    # ---------------- rekey ----------------
    def begin_rekey(self):
        self._hs = None
        self._send_cs = None
        self._recv_cs = None
        self.handshake_complete = False
        self.start_handshake(bool(self.is_initiator))
