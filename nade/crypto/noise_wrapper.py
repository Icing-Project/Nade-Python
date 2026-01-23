import traceback
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
        """
        Processes an incoming handshake message with robust error handling
        and state inspection.
        """
        # 1. Check Pre-conditions
        if self.handshake_complete:
            self.debug(f"[NoiseXK] Ignoring handshake msg (already complete)")
            return

        if not data:
            self.debug("warning", "[NoiseXK] Received empty handshake data frame.")
            return

        # 2. Lazy Initialization
        if self._hs is None:
            try:
                # Responder initializes lazily
                self._hs = self._new_handshakestate()
                self._hs.initialize(XKHandshakePattern(), False, b'', s=self.keypair)
                self.debug("[NoiseXK] Lazily initialized handshake state for responder")
            except Exception as e:
                # If init fails, we cannot proceed.
                self._log_crash("Handshake State Initialization", e)
                raise RuntimeError(f"Noise Initialization Failed: {repr(e)}") from e

        # 3. Process Message with granular error handling
        try:
            self.debug(f"[NoiseXK] Processing HS msg ({len(data)} bytes): {data.hex()}")
            
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
                self.debug(f"[NoiseXK] Queued HS response ({len(out)} bytes): {bytes(out).hex()}")
            
            if cs_pair_write:
                self._complete_handshake(cs_pair_write)

        except Exception as e:
            # 4. Contextual Logging
            # Log the full details here so you see them even if the caller just prints {e}
            self._log_crash("Handshake Processing", e, data)
            
            # Re-raise with a descriptive message so the caller's log isn't empty
            raise RuntimeError(f"Noise Protocol Error: {type(e).__name__} - {args_to_str(e)}") from e

    def _log_crash(self, context: str, e: Exception, data: bytes = None):
        """Helper to log detailed crash info including tracebacks."""
        tb_str = traceback.format_exc()
        error_details = (
            f"\n[NoiseXK] === CRITICAL ERROR: {context} ===\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Repr: {repr(e)}\n"
            f"Error Args: {e.args}\n"
        )
        if data:
            error_details += f"Bad Data Hex: {data.hex()}\n"
        
        error_details += f"Stack Trace:\n{tb_str}"
        error_details += "========================================="
        
        self.debug("error", error_details)


    def get_next_handshake_message(self) -> Optional[bytes]:
        return self.outgoing_messages.pop(0) if self.outgoing_messages else None

    # ---------------- AEAD ----------------
    def encrypt_sdu(self, ad: bytes, plaintext: bytes) -> bytes:
        if not (self.handshake_complete and self._send_cs):
            self.log("RuntimeError", f"Noise not ready")
            raise RuntimeError("Noise not ready")
        return self._send_cs.encrypt_with_ad(ad, plaintext)

    def decrypt_sdu(self, ad: bytes, ciphertext: bytes) -> bytes:
        if not (self.handshake_complete and self._recv_cs):
            self.log("RuntimeError", f"Noise not ready")
            raise RuntimeError("Noise not ready")
        return self._recv_cs.decrypt_with_ad(ad, ciphertext)

    def _complete_handshake(self, cs_pair: Tuple[_CipherStateType, _CipherStateType]):
        cs0, cs1 = cs_pair
        if self.is_initiator:
            self._send_cs, self._recv_cs = cs0, cs1
        else:
            self._send_cs, self._recv_cs = cs1, cs0
        self.handshake_complete = True

    # ---------------- rekey ----------------
    def begin_rekey(self):
        self._hs = None
        self._send_cs = None
        self._recv_cs = None
        self.handshake_complete = False
        self.start_handshake(bool(self.is_initiator))

def args_to_str(e):
    if hasattr(e, 'args') and e.args:
        return str(e.args)
    return str(e)