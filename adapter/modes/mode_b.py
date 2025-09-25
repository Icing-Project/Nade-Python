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
FSK_TONES = [600, 1200, 1800, 2400]
SYMBOL_LEN = 8
SYMBOLS_PER_BLOCK = BLOCK_SIZE // SYMBOL_LEN  # 160/8 = 20 symbols
BITS_PER_SYMBOL = 2
BITS_PER_BLOCK = SYMBOLS_PER_BLOCK * BITS_PER_SYMBOL


def bits_to_symbols(bits):
    out = []
    for i in range(0, len(bits), BITS_PER_SYMBOL):
        pair = bits[i:i+BITS_PER_SYMBOL]
        if len(pair) < 2:
            pair += [0] * (2 - len(pair))
        val = (pair[0] << 1) | pair[1]
        out.append(val)
    return out


def symbols_to_bits(symbols):
    out = []
    for s in symbols:
        out.append((s >> 1) & 1)
        out.append(s & 1)
    return out


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

    # ------------------------------------------------------------------
    # Modem
    # ------------------------------------------------------------------
    def _modulate(self, bits) -> np.ndarray:
        symbols = bits_to_symbols(bits)
        pcm = np.zeros(BLOCK_SIZE, dtype=np.float32)

        for i, sym in enumerate(symbols):
            f = FSK_TONES[sym]
            n0 = i * SYMBOL_LEN
            n1 = n0 + SYMBOL_LEN
            t = np.arange(SYMBOL_LEN) / SAMPLE_RATE
            tone = np.sin(2 * np.pi * f * t)
            pcm[n0:n1] = tone

        pcm = 0.25 * pcm
        return (pcm * 32767).astype(np.int16)

    def _demodulate(self, block: np.ndarray):
        block = block.astype(np.float32) / 32767.0
        symbols = []
        for i in range(SYMBOLS_PER_BLOCK):
            n0 = i * SYMBOL_LEN
            n1 = n0 + SYMBOL_LEN
            seg = block[n0:n1]
            scores = []
            t = np.arange(SYMBOL_LEN) / SAMPLE_RATE
            for f in FSK_TONES:
                ref = np.sin(2 * np.pi * f * t)
                score = np.dot(seg, ref)
                scores.append(score)
            sym = int(np.argmax(scores))
            symbols.append(sym)
        return symbols_to_bits(symbols)

    # ------------------------------------------------------------------
    # Frame parsing placeholder
    # ------------------------------------------------------------------
    def _try_parse_frames(self):
        # TODO: integrate FEC / Noise frame parsing
        pass
