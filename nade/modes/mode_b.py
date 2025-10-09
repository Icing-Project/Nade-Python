import numpy as np
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from nade.crypto.noise_wrapper import NoiseXKWrapper

# ------------------------------------------------------------------
# Audio parameters (DryBox defaults)
# ------------------------------------------------------------------
SAMPLE_RATE = 8000       # Hz
BLOCK_SIZE = 160         # samples (20 ms @ 8kHz)

# ------------------------------------------------------------------
# Modem parameters (4-FSK)
# ------------------------------------------------------------------
SYMBOL_LEN = 8
SYMBOLS_PER_BLOCK = BLOCK_SIZE // SYMBOL_LEN  # 160/8 = 20 symbols
BITS_PER_SYMBOL = 2
BITS_PER_BLOCK = SYMBOLS_PER_BLOCK * BITS_PER_SYMBOL


class NadeAudioPort:
    SAMPLE_RATE = SAMPLE_RATE
    BLOCK_SAMPLES = BLOCK_SIZE

    def __init__(self, side: str, local_kp: KeyPair, peer_pub: PublicKey, debug=print):
        self.side = side
        self.local_kp = local_kp
        self.peer_pub = peer_pub
        self.debug = debug

        # Handshake state
        # NoiseXKWrapper doesn’t accept `debug`, so drop it
        self.noise = NoiseXKWrapper(local_kp, peer_pub)

        # Buffers
        self.tx_bitstream = []
        self.rx_bitbuffer = []

    # ------------------------------------------------------------------
    # DryBox API
    # ------------------------------------------------------------------
    def pull_tx_block(self, t_ms: int) -> np.ndarray:
        if not self.tx_bitstream:
            return np.zeros(BLOCK_SIZE, dtype=np.int16)

        bits = self.tx_bitstream[:BITS_PER_BLOCK]
        self.tx_bitstream = self.tx_bitstream[BITS_PER_BLOCK:]
        return self._modulate(bits)

    def push_rx_block(self, pcm: np.ndarray, t_ms: int) -> None:
        bits = self._demodulate(pcm)
        self.rx_bitbuffer.extend(bits)
        self._try_parse_frames()

    def on_timer(self, t_ms: int) -> None:
        # For future AGC / FEC / reconfig
        pass

    def poll_link_tx(self, budget: int):
        return self.noise.poll_link_tx(budget)

    def on_link_rx(self, sdu: bytes, t_ms: int):
        self.noise.on_link_rx(sdu, t_ms)
