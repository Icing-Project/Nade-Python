# mode_a/noise_wrapper.py
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

# typing alias for readability
from dissononce.processing.impl.cipherstate import CipherState as _CipherStateType


def _hex(b: Optional[bytes]) -> str:
    return b.hex() if b else "<none>"


class NoiseXKWrapper:
    """Dissononce-only message-passing Noise XK wrapper with fresh HS per handshake."""

    def __init__(self, keypair: KeyPair, peer_pubkey: Optional[PublicKey] = None,
                 debug_callback: Optional[Callable[[str], None]] = None):
        """
        keypair: dissononce.dh.keypair.KeyPair (local static)
        peer_pubkey: dissononce.dh.x25519.public.PublicKey (peer static) - required for initiator
        debug_callback: optional callable(str) for logging (pass print)
        """
        self.keypair = keypair
        self.peer_pubkey = peer_pubkey
        self.debug = debug_callback or (lambda *a, **k: None)

        # We'll create the HandshakeState fresh when a handshake starts.
        self._hs: Optional[HandshakeState] = None

        self._send_cs: Optional[_CipherStateType] = None
        self._recv_cs: Optional[_CipherStateType] = None
        self.handshake_complete = False
        self.is_initiator: Optional[bool] = None

        # Queued handshake messages to be sent by the runner
        self.outgoing_messages: List[bytes] = []

    # ---------- helpers ----------
    def _new_handshakestate(self) -> HandshakeState:
        """Create a fresh HandshakeState with new primitives."""
        cipher = ChaChaPolyCipher()
        dh = X25519DH()
        hshash = SHA256Hash()
        symmetric = SymmetricState(CipherState(cipher), hshash)
        return HandshakeState(symmetric, dh)

    # ---------- handshake control ----------
    def start_handshake(self, initiator: bool):
        """Begin a fresh XK handshake. Always create new HandshakeState to avoid reused state."""
        self.debug(f"[NoiseXK] start_handshake(initiator={initiator})")
        self.is_initiator = initiator
        self.handshake_complete = False
        self._send_cs = None
        self._recv_cs = None
        self.outgoing_messages.clear()

        # build fresh HS
        self._hs = self._new_handshakestate()

        # For debug: show static pubs (best-effort access)
        local_pub_b = None
        peer_pub_b = None
        try:
            local_pub_b = self.keypair.public.data
        except Exception:
            try:
                local_pub_b = bytes(self.keypair.public)
            except Exception:
                local_pub_b = None
        if self.peer_pubkey is not None:
            try:
                peer_pub_b = self.peer_pubkey.data
            except Exception:
                try:
                    peer_pub_b = bytes(self.peer_pubkey)
                except Exception:
                    peer_pub_b = None

        self.debug(f"[NoiseXK] local_pub={_hex(local_pub_b)} remote_pub={_hex(peer_pub_b)}")

        if initiator:
            if self.peer_pubkey is None:
                raise ValueError("Initiator requires peer static public key")
            self._hs.initialize(XKHandshakePattern(), True, b'', s=self.keypair, rs=self.peer_pubkey)
            # Write M1
            buf = bytearray()
            self._hs.write_message(b'', buf)
            self.outgoing_messages.append(bytes(buf))
            self.debug(f"[NoiseXK] queued M1 ({len(buf)} bytes): {bytes(buf).hex()}")
        else:
            self._hs.initialize(XKHandshakePattern(), False, b'', s=self.keypair)
            self.debug("[NoiseXK] responder initialized (waiting for M1)")

    def process_handshake_message(self, data: bytes):
        """Process an incoming handshake message and generate response if needed."""
        # Defensive: ignore HS messages if we already have a complete handshake
        if self.handshake_complete:
            self.debug(f"[NoiseXK] received HS message but handshake already complete; ignoring ({len(data)} B)")
            return

        if self._hs is None:
            # Defensive fallback (shouldn't normally happen): initialize as responder
            self._hs = self._new_handshakestate()
            self._hs.initialize(XKHandshakePattern(), False, b'', s=self.keypair)
            self.debug("[NoiseXK] lazily initialized handshake state for responder")

        self.debug(f"[NoiseXK] processing incoming HS ({len(data)} bytes): {data.hex()}")
        try:
            # First, attempt to read the incoming token.
            payload = bytearray()
            cs_pair_read = self._hs.read_message(data, payload)
        except Exception as e:
            self.debug(f"[NoiseXK] read_message failed; input={data.hex()}; err={e}")
            raise

        # If read completed the handshake (responder final read), install keys and return.
        if cs_pair_read:
            self.debug("[NoiseXK] read_message indicates handshake completion (read path)")
            self._complete_handshake(cs_pair_read)
            return

        # Otherwise, we may need to write a response (initiator or intermediate).
        try:
            out = bytearray()
            cs_pair_write = self._hs.write_message(b'', out)
            if out:
                self.outgoing_messages.append(bytes(out))
                self.debug(f"[NoiseXK] queued HS response ({len(out)} bytes): {bytes(out).hex()}")
            if cs_pair_write:
                # Handshake completed as a result of our write (initiator path)
                self.debug("[NoiseXK] write_message indicates handshake completion (write path)")
                self._complete_handshake(cs_pair_write)
        except IndexError as ie:
            # HandshakeState had no message patterns left; treat as benign and log
            self.debug(f"[NoiseXK] write_message produced IndexError (no message patterns left); ignoring. err={ie}")
        except Exception as e:
            # Other unexpected errors should propagate after logging
            self.debug(f"[NoiseXK] write_message failed; err={e}")
            raise

    def get_next_handshake_message(self) -> Optional[bytes]:
        """Return next queued handshake message (or None)."""
        return self.outgoing_messages.pop(0) if self.outgoing_messages else None

    # ---------- AEAD interfaces ----------
    def encrypt_sdu(self, ad: bytes, plaintext: bytes) -> bytes:
        if not (self.handshake_complete and self._send_cs):
            raise RuntimeError("Noise not ready")
        return self._send_cs.encrypt_with_ad(ad, plaintext)

    def decrypt_sdu(self, ad: bytes, ciphertext: bytes) -> bytes:
        if not (self.handshake_complete and self._recv_cs):
            raise RuntimeError("Noise not ready")
        return self._recv_cs.decrypt_with_ad(ad, ciphertext)

    def _complete_handshake(self, cs_pair: Tuple[_CipherStateType, _CipherStateType]) -> None:
        """Install cipher states returned by HandshakeState (pair)."""
        self.debug("[NoiseXK] completing handshake")
        cs0, cs1 = cs_pair
        if self.is_initiator:
            self._send_cs, self._recv_cs = cs0, cs1
            self.debug("[NoiseXK] set cipherstates as initiator")
        else:
            self._send_cs, self._recv_cs = cs1, cs0
            self.debug("[NoiseXK] set cipherstates as responder")
        self.handshake_complete = True
        self.debug("[NoiseXK] handshake complete; secure channel ready")

    # ---------- rekey ----------
    def begin_rekey(self) -> None:
        """Start a fresh XK handshake with same static keys."""
        self.debug("[NoiseXK] begin_rekey")
        self._hs = None
        self._send_cs = None
        self._recv_cs = None
        self.handshake_complete = False
        # reuse previous role if set, else default to initiator
        initiator = bool(self.is_initiator)
        self.start_handshake(initiator)
