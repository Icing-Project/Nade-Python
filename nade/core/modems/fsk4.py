# nade/core/modems/fsk4.py
from __future__ import annotations
from typing import List, Deque, Optional, Tuple
from collections import deque
import math
import numpy as np

class FourFSKModem:
    """
    4-FSK modem minimal:
      - 2 bits/sym → 4 tons.
      - sps = 8 (8 échantillons par symbole) @ 8 kHz → 1000 sym/s.
      - Fréquences: [1000, 1500, 2000, 2500] Hz (non-cohérent, Goertzel).
      - Framing simple: PREAMBLE (0x55 * 16) + SYNC(0xD3,0x91) + LEN(1) + PAYLOAD + CHECKSUM(1, sum mod 256)
    API:
      - queue_text(str)
      - pull_tx_block(t_ms) -> np.int16[160]
      - push_rx_block(pcm:int16[160], t_ms)
      - on_timer(t_ms)
      - pop_received_texts() -> List[str]
    """

    SR = 8000
    BLK = 160
    SPS = 8  # samples per symbol
    SYMS_PER_BLK = BLK // SPS  # 20
    TONES = [1000.0, 1500.0, 2000.0, 2500.0]  # Hz
    AMP = 9000  # amplitude (int16)

    PREAMBLE = bytes([0x55] * 16)    # 0b01010101 * 16
    SYNC = bytes([0xD3, 0x91])

    def __init__(self, logger=None):
        self.log = logger or (lambda lvl, msg: None)

        # TX state
        self._tx_bits: Deque[int] = deque()
        self._tx_sym_cursor = 0  # position dans le bloc courant (0..19)
        self._tx_wave_cache = self._build_symbol_waves()  # dict{sym: np.int16[SPS]}
        self._idle_symbol = 0  # ton par défaut quand rien à émettre

        # RX state
        self._rx_bitbuf: Deque[int] = deque()
        self._rx_bytebuf: Deque[int] = deque()
        self._rx_texts: Deque[str] = deque()
        self._rx_searching = True

        # Goertzel precompute
        self._goertzel_coeffs = [2.0 * math.cos(2.0 * math.pi * f / self.SR * self.SPS) for f in self.TONES]

    # ---------- Public API ----------
    def queue_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        if len(payload) > 255:
            payload = payload[:255]
        frame = self._build_frame(payload)
        bits = self._bytes_to_2bit_symbols(frame)  # renvoie liste de symbols (0..3) mais sous forme de 2 bits
        # On stocke directement des symboles (0..3) en les poussant 2 bits à la fois
        for sym in bits:
            self._tx_bits.append(sym)

        self.log("info", f"[4FSK] TX enqueued text '{text}' ({len(payload)} bytes)")

    def pull_tx_block(self, t_ms: int) -> np.ndarray:
        # Produit 20 symboles (ou idle) → 160 échantillons
        out = np.empty(self.BLK, dtype=np.int16)
        p = 0
        for _ in range(self.SYMS_PER_BLK):
            sym = self._idle_symbol
            if self._tx_bits:
                sym = self._tx_bits.popleft()
            wave = self._tx_wave_cache[sym]
            out[p:p+self.SPS] = wave
            p += self.SPS
        return out

    def push_rx_block(self, pcm: np.ndarray, t_ms: int) -> None:
        # Découpe 20 symboles de 8 échantillons → détecte le ton dominant par Goertzel
        if pcm is None or len(pcm) < self.BLK:
            return
        # Normalise en float
        x = pcm.astype(np.float32)

        # Pour chaque symbole:
        idx = 0
        for _ in range(self.SYMS_PER_BLK):
            seg = x[idx:idx+self.SPS]
            idx += self.SPS
            sym = self._detect_symbol_goertzel(seg)
            self._rx_push_symbol(sym)

        # Essaye d’extraire des trames & textes
        self._rx_drain_frames()

    def on_timer(self, t_ms: int) -> None:
        pass

    def pop_received_texts(self) -> List[str]:
        items = list(self._rx_texts)
        self._rx_texts.clear()
        return items

    # ---------- TX helpers ----------
    def _build_symbol_waves(self) -> List[np.ndarray]:
        # Génère 4 ondes (int16) de SPS échantillons chacune
        waves: List[np.ndarray] = []
        t = np.arange(self.SPS) / self.SR
        for f in self.TONES:
            w = (self.AMP * np.sin(2.0 * math.pi * f * t)).astype(np.int16)
            waves.append(w)
        return waves

    def _build_frame(self, payload: bytes) -> bytes:
        ln = len(payload)
        cksum = (sum(payload) + ln) & 0xFF
        return self.PREAMBLE + self.SYNC + bytes([ln]) + payload + bytes([cksum])

    def _bytes_to_2bit_symbols(self, data: bytes) -> List[int]:
        # Transforme chaque octet en 4 symboles (2 bits chacun, MSB→LSB)
        out: List[int] = []
        for b in data:
            out.append((b >> 6) & 0x03)
            out.append((b >> 4) & 0x03)
            out.append((b >> 2) & 0x03)
            out.append((b >> 0) & 0x03)
        return out

    # ---------- RX helpers ----------
    def _detect_symbol_goertzel(self, seg: np.ndarray) -> int:
        # Non-cohérent: Goertzel sur SPS=8 échantillons
        # Accumulateur Goertzel classique
        max_pow = -1.0
        best_k = 0
        for k, coeff in enumerate(self._goertzel_coeffs):
            s_prev = 0.0
            s_prev2 = 0.0
            for x in seg:
                s = x + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
            if power > max_pow:
                max_pow = power
                best_k = k
        return best_k  # 0..3

    def _rx_push_symbol(self, sym: int) -> None:
        # Accumule 4 symboles → 1 octet (MSB..LSB)
        self._rx_bitbuf.append(sym)
        if len(self._rx_bitbuf) >= 4:
            b = ((self._rx_bitbuf.popleft() & 3) << 6) \
              | ((self._rx_bitbuf.popleft() & 3) << 4) \
              | ((self._rx_bitbuf.popleft() & 3) << 2) \
              | ((self._rx_bitbuf.popleft() & 3) << 0)
            self._rx_bytebuf.append(b)

    def _rx_drain_frames(self) -> None:
        # Recherche PREAMBLE + SYNC, puis extrait LEN, PAYLOAD, CHECKSUM
        buf = self._rx_bytebuf
        while True:
            # cherche SYNC (après avoir vu du 0x55)
            if len(buf) < 32:  # seuil pour éviter la recherche coûteuse si trop court
                return

            # sync search
            found = False
            i = 0
            # Tolère n'importe quel nombre de 0x55 (≥ 8), puis 0xD3 0x91
            while i <= len(buf) - 2:
                if i >= 8 and buf[i] == self.SYNC[0] and buf[i+1] == self.SYNC[1]:
                    found = True
                    break
                i += 1
            if not found:
                # drop quelques octets pour ne pas gonfler sans fin
                if len(buf) > 256:
                    for _ in range(64):
                        buf.popleft()
                return

            # On droppe tout jusqu’au SYNC (exclu)
            for _ in range(i):
                buf.popleft()

            # On a [SYNC0, SYNC1, ...]
            if len(buf) < 3:
                return
            sync0 = buf.popleft()
            sync1 = buf.popleft()
            if sync0 != self.SYNC[0] or sync1 != self.SYNC[1]:
                continue

            if not buf:
                return
            ln = buf.popleft()
            need = ln + 1  # payload + checksum
            if len(buf) < need:
                # pas assez de données, on attend le prochain bloc
                # (on remet ce qu’on a consommé ? Simple: on garde l’état, on ne remet pas)
                return

            payload = bytes([buf.popleft() for _ in range(ln)])
            cksum = buf.popleft()
            if ((sum(payload) + ln) & 0xFF) == cksum:
                try:
                    text = payload.decode("utf-8", errors="replace")
                except Exception:
                    text = payload.decode("latin-1", errors="replace")
                self._rx_texts.append(text)
                self.log("info", f"[4FSK] RX text: '{text}'")
            else:
                self.log("warn", "[4FSK] checksum mismatch")
            # Continue à chercher une autre trame dans le tampon
