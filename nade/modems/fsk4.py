from __future__ import annotations
from typing import Deque, List, Optional, Tuple, Callable
from collections import deque
import math
import numpy as np

from .imodem import IModem, ModemConfig, BackpressurePolicy, Int16Block

# ---- small helpers ----------------------------------------------------------
def _ensure_c_contig_i16(x: np.ndarray, n: int) -> Int16Block:
    if x.dtype != np.int16 or x.ndim != 1 or x.size != n:
        raise ValueError("pcm must be int16[block_size]")
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x  # type: ignore[return-value]

# ---- 4-FSK modem ------------------------------------------------------------
class FourFSKModem(IModem):
    """
    4-FSK modem implementing the enhanced IModem interface.
    - 2 bits/symbol, 4 tones
    - Simple non-coherent Goertzel per symbol
    - Framing: PREAMBLE(0x55*N) + SYNC(2) + LEN(1) + PAYLOAD + SUM(1)
    """

    # Defaults for modem-specific params (can be overridden by configure())
    _DEF_TONES = (1000.0, 1500.0, 2000.0, 2500.0)
    _DEF_SPS   = 8
    _DEF_AMP   = 9000
    _DEF_PREAMBLE = 16
    _DEF_SYNC  = (0xD3, 0x91)

    def __init__(self, cfg: Optional[ModemConfig] = None,
                 logger: Optional[Callable[[str, object], None]] = None,
                 **modem_params):
        self.log = logger or (lambda lvl, payload: None)
        # queues
        self._tx_syms: Deque[int] = deque()
        self._rx_syms: Deque[int] = deque()
        self._rx_bytes: Deque[int] = deque()
        self._rx_frames: Deque[bytes] = deque()
        # runtime state
        self._idle_symbol = 0
        # modem-specific params
        self._tones = list(modem_params.get("tones", self._DEF_TONES))
        self._sps   = int(modem_params.get("sps", self._DEF_SPS))
        self._amp   = int(modem_params.get("amp", self._DEF_AMP))
        self._preamble_len = int(modem_params.get("preamble_len", self._DEF_PREAMBLE))
        self._sync = tuple(int(x) & 0xFF for x in modem_params.get("sync", self._DEF_SYNC))
        # core cfg
        if cfg is None:
            cfg = ModemConfig(sample_rate_hz=8000, block_size=160)
        self.configure(cfg)

    # ---- lifecycle / capability --------------------------------------------
    def configure(self, cfg: ModemConfig) -> None:
        # immutable copy
        self.cfg = ModemConfig(**vars(cfg))
        # derived
        self.SR = int(self.cfg.sample_rate_hz)
        self.BLK = int(self.cfg.block_size)
        self.SPS = self._sps
        self.SYMS_PER_BLK = self.BLK // self.SPS
        self.PREAMBLE = bytes([0x55] * self._preamble_len)
        self.SYNC = bytes([self._sync[0], self._sync[1]])
        # synthesis & detection tables
        self._waves = self._build_symbol_waves()
        self._coeffs = [2.0 * math.cos(2.0 * math.pi * (f / self.SR)) for f in self._tones]
        # enforce queue sizes (truncate if needed)
        self._truncate_rx_queue()
        # leave TX symbol queue intact (safe across reconfig)
        self.log("info", f"[4FSK] configured sr={self.SR} blk={self.BLK} sps={self.SPS} tones={self._tones}")

    def reset(self) -> None:
        self._tx_syms.clear()
        self._rx_syms.clear()
        self._rx_bytes.clear()
        self._rx_frames.clear()

    def close(self) -> None:
        # No resources to release right now (kept for symmetry).
        pass

    # ---- timer --------------------------------------------------------------
    def on_timer(self, t_ms: int) -> None:
        # hook for future AGC/rate adaptation
        return

    # ---- app frames ---------------------------------------------------------
    def tx_enqueue(self, frame: bytes) -> bool:
        # backpressure on *frames*, not symbols
        if len(self._pending_frames()) >= self.cfg.max_tx_frames:
            pol = self.cfg.backpressure
            if pol is BackpressurePolicy.BLOCK_NEVER:
                return False
            elif pol is BackpressurePolicy.DROP_NEWEST:
                return False
            elif pol is BackpressurePolicy.DROP_OLDEST:
                # drop one oldest frame from the symbol queue boundary
                self._drain_one_tx_frame_symbols(drop_only=True)
            else:
                return False

        payload = frame[:255]
        fr_bytes = self._build_frame(payload)
        # expand to 2-bit symbols and enqueue
        for sym in self._bytes_to_2bit_symbols(fr_bytes):
            self._tx_syms.append(sym)
        return True

    def rx_dequeue(self, limit: Optional[int] = None) -> list[bytes]:
        out: List[bytes] = []
        n = len(self._rx_frames) if limit is None else min(limit, len(self._rx_frames))
        for _ in range(n):
            out.append(self._rx_frames.popleft())
        return out

    # ---- audio block I/O ----------------------------------------------------
    def pull_tx_block(self, t_ms: int) -> Int16Block:
        out = np.empty(self.BLK, dtype=np.int16)
        p = 0
        for _ in range(self.SYMS_PER_BLK):
            sym = self._tx_syms.popleft() if self._tx_syms else self._idle_symbol
            out[p:p+self.SPS] = self._waves[sym]
            p += self.SPS
        return out  # C-contiguous by construction

    def push_rx_block(self, pcm: Int16Block, t_ms: int) -> None:
        x = _ensure_c_contig_i16(pcm, self.BLK).astype(np.float32, copy=False)
        idx = 0
        for _ in range(self.SYMS_PER_BLK):
            seg = x[idx:idx+self.SPS]
            idx += self.SPS
            sym, _ = self._detect_symbol(seg)
            self._rx_syms.append(sym)

            if len(self._rx_syms) >= 4:
                b = ((self._rx_syms.popleft() & 3) << 6) \
                  | ((self._rx_syms.popleft() & 3) << 4) \
                  | ((self._rx_syms.popleft() & 3) << 2) \
                  | ((self._rx_syms.popleft() & 3) << 0)
                self._rx_bytes.append(b)

        self._drain_frames()

    # ---- internals ----------------------------------------------------------
    def _build_symbol_waves(self) -> List[np.ndarray]:
        t = np.arange(self.SPS, dtype=np.float32) / float(self.SR)
        waves = []
        for f in self._tones:
            w = (self._amp * np.sin(2.0 * math.pi * f * t)).astype(np.int16)
            waves.append(w)
        return waves

    def _detect_symbol(self, seg: np.ndarray) -> Tuple[int, List[float]]:
        best_k = 0
        best_p = -1.0
        powers: List[float] = []
        for k, coeff in enumerate(self._coeffs):
            s_prev = 0.0
            s_prev2 = 0.0
            for x in seg:
                s = x + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
            powers.append(power)
            if power > best_p:
                best_p = power
                best_k = k
        return best_k, powers

    def _bytes_to_2bit_symbols(self, data: bytes) -> List[int]:
        out: List[int] = []
        for b in data:
            out.extend([(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, (b >> 0) & 3])
        return out

    def _build_frame(self, payload: bytes) -> bytes:
        ln = len(payload)
        cksum = (sum(payload) + ln) & 0xFF
        return self.PREAMBLE + self.SYNC + bytes([ln]) + payload + bytes([cksum])

    def _drain_frames(self) -> None:
        """Parse frames out of _rx_bytes into _rx_frames with RX backpressure."""
        # honor RX backpressure: keep at most max_rx_frames
        while len(self._rx_frames) > self.cfg.max_rx_frames:
            # DROP_OLDEST semantics: discard oldest
            self._rx_frames.popleft()

        buf = self._rx_bytes
        while True:
            if len(self._rx_frames) >= self.cfg.max_rx_frames:
                return  # stop parsing; keep bytes for later

            # need minimum header
            min_hdr = self._preamble_len + 2 + 1
            if len(buf) < min_hdr:
                return

            # skip preamble (>=8 x 0x55)
            pre = 0
            while pre < len(buf) and buf[pre] == 0x55:
                pre += 1
            if pre < 8:
                buf.popleft()
                continue

            if pre + 1 >= len(buf):
                return
            if buf[pre] != self.SYNC[0] or buf[pre+1] != self.SYNC[1]:
                buf.popleft()
                continue

            # consume preamble + sync
            for _ in range(pre):
                buf.popleft()
            if len(buf) < 2:
                return
            if buf.popleft() != self.SYNC[0]: return
            if buf.popleft() != self.SYNC[1]: return

            if not buf: return
            ln = buf.popleft()
            need = ln + 1
            if len(buf) < need:
                return

            payload = bytes([buf.popleft() for _ in range(ln)])
            cksum = buf.popleft()
            ok = (((sum(payload) + ln) & 0xFF) == cksum)
            if ok:
                if len(self._rx_frames) < self.cfg.max_rx_frames:
                    self._rx_frames.append(payload)
                else:
                    # overflow: apply RX policy (we only implement DROP_OLDEST)
                    self._rx_frames.popleft()
                    self._rx_frames.append(payload)

    # bookkeeping for backpressure on TX (frame granularity)
    def _pending_frames(self) -> List[int]:
        # heuristic: count frames by scanning symbol queue for preamble/sync boundaries is too heavy.
        # Instead, we track at the enqueue time only; simple approximation:
        # 1 enqueued frame == we added some symbols; we can't count precisely without markers,
        # so this returns 0/1 to allow a single-frame window unless more elaborate tagging is added.
        return [1] if len(self._tx_syms) > 0 else []

    def _drain_one_tx_frame_symbols(self, drop_only: bool = False) -> None:
        # Aggressively drop current TX buffer (best-effort) — coarse but effective.
        self._tx_syms.clear()
        # if we had per-frame markers, we'd only drop up to next marker.
