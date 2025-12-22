from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Deque, Iterable, List, Optional
import ctypes
import importlib
import math
from collections import deque
from pathlib import Path

import numpy as np

from .imodem import BackpressurePolicy, IModem, Int16Block, ModemConfig


class _LiquidFSKLibrary:
    """ctypes bridge for the subset of liquid-dsp FSK routines we need."""

    _instance: Optional["_LiquidFSKLibrary"] = None

    def __init__(self) -> None:
        try:
            module = importlib.import_module("liquid")
        except ImportError as exc:  # pragma: no cover
            self.log("RuntimeError", f"liquid-dsp is required for the audio modem backend; install the 'liquid-dsp' wheel")
            raise RuntimeError(
                "liquid-dsp is required for the audio modem backend; install the 'liquid-dsp' wheel"
            ) from exc

        lib_path = self._discover_shared_library(Path(module.__file__).resolve())
        if lib_path is None:
            self.log("RuntimeError", f"Unable to locate the libliquid shared library next to the python extension")
            raise RuntimeError("Unable to locate the libliquid shared library next to the python extension")

        self.lib = ctypes.CDLL(str(lib_path))

        class _CFComplex(ctypes.Structure):
            _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]

        self._c_complex = _CFComplex

        c_uint = ctypes.c_uint
        c_float = ctypes.c_float
        c_void_p = ctypes.c_void_p

        self.lib.fskmod_create.restype = c_void_p
        self.lib.fskmod_create.argtypes = [c_uint, c_uint, c_float]
        self.lib.fskmod_destroy.restype = ctypes.c_int
        self.lib.fskmod_destroy.argtypes = [c_void_p]
        self.lib.fskmod_modulate.restype = ctypes.c_int
        self.lib.fskmod_modulate.argtypes = [c_void_p, c_uint, ctypes.POINTER(_CFComplex)]

        self.lib.fskdem_create.restype = c_void_p
        self.lib.fskdem_create.argtypes = [c_uint, c_uint, c_float]
        self.lib.fskdem_destroy.restype = ctypes.c_int
        self.lib.fskdem_destroy.argtypes = [c_void_p]
        self.lib.fskdem_demodulate.restype = c_uint
        self.lib.fskdem_demodulate.argtypes = [c_void_p, ctypes.POINTER(_CFComplex)]

        self.lib.firhilbf_create.restype = c_void_p
        self.lib.firhilbf_create.argtypes = [c_uint, c_float]
        self.lib.firhilbf_destroy.restype = ctypes.c_int
        self.lib.firhilbf_destroy.argtypes = [c_void_p]
        self.lib.firhilbf_reset.restype = ctypes.c_int
        self.lib.firhilbf_reset.argtypes = [c_void_p]
        self.lib.firhilbf_r2c_execute.restype = ctypes.c_int
        self.lib.firhilbf_r2c_execute.argtypes = [c_void_p, c_float, ctypes.POINTER(_CFComplex)]

    # ------------------------------------------------------------------ utils
    @classmethod
    def instance(cls) -> "_LiquidFSKLibrary":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_mod(self, bits_per_symbol: int, sps: int, bandwidth: float) -> ctypes.c_void_p:
        handle = self.lib.fskmod_create(bits_per_symbol, sps, ctypes.c_float(bandwidth))
        if not handle:
            self.log("RuntimeError", f"fskmod_create returned NULL")
            raise RuntimeError("fskmod_create returned NULL")
        return handle

    def destroy_mod(self, handle: Optional[ctypes.c_void_p]) -> None:
        if handle:
            self.lib.fskmod_destroy(handle)

    def modulate(self, handle: ctypes.c_void_p, symbol: int, buf: np.ndarray) -> None:
        ptr = buf.ctypes.data_as(ctypes.POINTER(self._c_complex))
        rc = self.lib.fskmod_modulate(handle, symbol, ptr)
        if rc != 0:
            self.log("RuntimeError", f"fskmod_modulate failed with code {rc}")
            raise RuntimeError(f"fskmod_modulate failed with code {rc}")

    def create_dem(self, bits_per_symbol: int, sps: int, bandwidth: float) -> ctypes.c_void_p:
        handle = self.lib.fskdem_create(bits_per_symbol, sps, ctypes.c_float(bandwidth))
        if not handle:
            self.log("RuntimeError", f"fskdem_create returned NULL")
            raise RuntimeError("fskdem_create returned NULL")
        return handle

    def destroy_dem(self, handle: Optional[ctypes.c_void_p]) -> None:
        if handle:
            self.lib.fskdem_destroy(handle)

    def demodulate(self, handle: ctypes.c_void_p, buf: np.ndarray) -> int:
        ptr = buf.ctypes.data_as(ctypes.POINTER(self._c_complex))
        return int(self.lib.fskdem_demodulate(handle, ptr))

    def create_hilbert(self, semi_length: int, attenuation_db: float) -> ctypes.c_void_p:
        handle = self.lib.firhilbf_create(semi_length, ctypes.c_float(attenuation_db))
        if not handle:
            self.log("RuntimeError", f"firhilbf_create returned NULL")
            raise RuntimeError("firhilbf_create returned NULL")
        return handle

    def destroy_hilbert(self, handle: Optional[ctypes.c_void_p]) -> None:
        if handle:
            self.lib.firhilbf_destroy(handle)

    def reset_hilbert(self, handle: Optional[ctypes.c_void_p]) -> None:
        if handle:
            self.lib.firhilbf_reset(handle)

    def hilbert_r2c(self, handle: ctypes.c_void_p, samples: np.ndarray, out: np.ndarray) -> None:
        c_value = self._c_complex()
        for idx, sample in enumerate(samples.astype(np.float32, copy=False)):
            self.lib.firhilbf_r2c_execute(handle, ctypes.c_float(float(sample)), ctypes.byref(c_value))
            out[idx] = complex(c_value.real, c_value.imag)

    @staticmethod
    def _discover_shared_library(root: Path) -> Optional[Path]:
        candidates: List[Path] = []

        if root.is_file() and root.suffix in {".so", ".pyd", ".dll"}:
            candidates.append(root)

        parent = root.parent if root.is_file() else root
        candidates.extend(parent.glob("libliquid*.so"))
        candidates.extend(parent.glob("libliquid*.dylib"))
        candidates.extend(parent.glob("liquid*.dll"))

        lib_dir = parent / "liquid_dsp.libs"
        if lib_dir.exists():
            candidates.extend(lib_dir.glob("libliquid*.so"))
            candidates.extend(lib_dir.glob("libliquid*.dylib"))
            candidates.extend(lib_dir.glob("liquid*.dll"))

        return next((p for p in candidates if p.exists()), None)


@dataclass
class _BitBucket:
    bits: int = 0
    count: int = 0

    def push(self, value: int, width: int) -> Iterable[int]:
        self.bits = (self.bits << width) | (value & ((1 << width) - 1))
        self.count += width
        out: List[int] = []
        while self.count >= 8:
            shift = self.count - 8
            out.append((self.bits >> shift) & 0xFF)
            self.bits &= (1 << shift) - 1 if shift else 0
            self.count -= 8
        return out

    def clear(self) -> None:
        self.bits = 0
        self.count = 0


class LiquidFSKModem(IModem):
    """
    Liquid-dsp-backed M-FSK modem (supports B/4-FSK via runtime parameters).

    Single IModem implementation of Liquid FSK, configurable for 2-FSK (BFSK) or 4-FSK.
    """

    DEFAULT_SPS: int = 40  # samples per symbol, == SR * symbol_duration, e.g. 40 @ 8kHz for 5ms symbols
    DEFAULT_BANDWIDTH: float = 0.18  # normalized (0, 0.5)
    DEFAULT_CARRIER_HZ: float = 1200.0
    DEFAULT_AMPLITUDE: int = 9000
    DEFAULT_IDLE_SYMBOL: int = 0

    def __init__(self,
                 cfg: Optional[ModemConfig] = None,
                 logger: Optional[Callable[[str, object], None]] = None,
                 **params: object) -> None:
        self.log = logger or (lambda level, payload: None)

        self.bits_per_symbol = int(params.get("bits_per_symbol", 2))
        if self.bits_per_symbol <= 0:
            raise ValueError("bits_per_symbol must be > 0")
        if self.bits_per_symbol > 4:
            raise ValueError("bits_per_symbol > 4 is not currently supported")

        self._amp = int(params.get("amp", self.DEFAULT_AMPLITUDE))
        self._samples_per_symbol = int(params.get("samples_per_symbol", self.DEFAULT_SPS))
        self._bandwidth = float(params.get("bandwidth", self.DEFAULT_BANDWIDTH))
        self._carrier_hz = float(params.get("carrier_hz", self.DEFAULT_CARRIER_HZ))
        self._idle_symbol = int(params.get("idle_symbol", self.DEFAULT_IDLE_SYMBOL))
        # Optional RX mixer sign override for debug/tuning: -1 (default) uses exp(-j 2π f_c), +1 uses exp(+j 2π f_c)
        self._rx_mix_sign = int(params.get("rx_mix_sign", -1))

        # Optional CFO tracking (EMA + per-block rotator)
        self._cfo_track: bool = bool(params.get("cfo_track", False))
        self._cfo_alpha: float = float(params.get("cfo_alpha", 0.1))
        self._cfo_max_hz: float = float(params.get("cfo_max_hz", 1500.0))
        self._cfo_rps_est: float = 0.0  # radians per sample

        if self._amp <= 0:
            raise ValueError("amp must be positive")
        if self._samples_per_symbol < (1 << self.bits_per_symbol):
            raise ValueError("samples_per_symbol must be >= 2^bits_per_symbol for liquid fskmod")
        if not (0.0 < self._bandwidth < 0.5):
            raise ValueError("bandwidth must be in (0, 0.5)")

        self._backend = _LiquidFSKLibrary.instance()

        self._mod_handle: Optional[ctypes.c_void_p] = None
        self._dem_handle: Optional[ctypes.c_void_p] = None
        self._hilbert_handle: Optional[ctypes.c_void_p] = None

        self._tx_syms: Deque[int] = deque()
        self._rx_frames: Deque[bytes] = deque()
        self._rx_bytes: Deque[int] = deque()
        self._rx_buffer = np.zeros(0, dtype=np.complex64)
        self._analytic_tmp = np.zeros(0, dtype=np.complex64)

        self._tx_phase = 0.0
        self._rx_phase = 0.0

        self._bit_bucket = _BitBucket()

        self.cfg = ModemConfig(sample_rate_hz=8000, block_size=160) if cfg is None else cfg
        self.configure(self.cfg)

        # --- instrumentation (demod progress) ---
        self._metric_rx_symbols: int = 0
        self._metric_rx_bytes_total: int = 0
        self._metric_preamble_hits: int = 0
        self._metric_frames_decoded: int = 0

        # --- symbol-level reassembly (bps==2) ---
        self._rx_symbols: Deque[int] = deque()
        self._sym_state: str = "search"
        self._sym_needed: int = 0
        self._sym_expect: List[int] = []
        self._sym_tmp_symbols: List[int] = []
        # Preamble detection window state
        self._sym_pre_win: Deque[int] = deque()
        self._sym_pre_count: int = 0

    # ---------------------------------------------------------------- public
    def configure(self, cfg: ModemConfig) -> None:
        if cfg.block_size % self._samples_per_symbol != 0:
            raise ValueError("block_size must be a multiple of samples_per_symbol")

        self.cfg = ModemConfig(**vars(cfg))

        self.SR = int(self.cfg.sample_rate_hz)
        self.BLK = int(self.cfg.block_size)
        self.SPS = self._samples_per_symbol
        self.SYMS_PER_BLK = self.BLK // self.SPS

        self._backend.destroy_mod(self._mod_handle)
        self._backend.destroy_dem(self._dem_handle)
        self._backend.destroy_hilbert(getattr(self, "_hilbert_handle", None))

        self._mod_handle = self._backend.create_mod(self.bits_per_symbol, self.SPS, self._bandwidth)
        self._dem_handle = self._backend.create_dem(self.bits_per_symbol, self.SPS, self._bandwidth)

        self._mod_tmp = np.zeros(self.SPS, dtype=np.complex64)
        phase = 2.0 * math.pi * self._carrier_hz * np.arange(self.SPS) / self.SR
        self._carrier_symbol = np.exp(1j * phase).astype(np.complex64)
        self._carrier_symbol_phase = float(2.0 * math.pi * self._carrier_hz * self.SPS / self.SR)
        block_phase = 2.0 * math.pi * self._carrier_hz * np.arange(self.BLK) / self.SR
        self._carrier_block = np.exp(-1j * block_phase).astype(np.complex64)
        self._carrier_block_phase = float(2.0 * math.pi * self._carrier_hz * self.BLK / self.SR)
        self._hilbert_handle = self._backend.create_hilbert(8, 60.0)
        self._analytic_tmp = np.zeros(self.BLK, dtype=np.complex64)

        self._rx_buffer = np.zeros(0, dtype=np.complex64)
        self._bit_bucket.clear()
        self._tx_phase = 0.0
        self._rx_phase = 0.0
        self._backend.reset_hilbert(getattr(self, "_hilbert_handle", None))

        # Reset runtime instrumentation at (re)configure
        self._metric_rx_symbols = 0
        self._metric_rx_bytes_total = 0
        self._metric_preamble_hits = 0
        self._metric_frames_decoded = 0
        # Reset symbol-level state at (re)configure
        try:
            self._rx_symbols.clear()
            self._sym_state = "search"
            self._sym_needed = 0
            self._sym_expect = []
            self._sym_tmp_symbols = []
            self._sym_pre_win.clear()
            self._sym_pre_count = 0
        except Exception:
            pass

        self.log("info", {
            "event": "cfg",
            "modem": "liquid-fsk",
            "bits_per_symbol": self.bits_per_symbol,
            "sps": self.SPS,
            "bandwidth": self._bandwidth,
            "carrier_hz": self._carrier_hz,
        })

    def reset(self) -> None:
        self._tx_syms.clear()
        self._rx_frames.clear()
        self._rx_bytes.clear()
        self._rx_buffer = np.zeros(0, dtype=np.complex64)
        self._bit_bucket.clear()
        self._tx_phase = 0.0
        self._rx_phase = 0.0
        self._backend.reset_hilbert(self._hilbert_handle)

    def close(self) -> None:
        self._backend.destroy_mod(self._mod_handle)
        self._backend.destroy_dem(self._dem_handle)
        self._backend.destroy_hilbert(getattr(self, "_hilbert_handle", None))
        self._mod_handle = None
        self._dem_handle = None

    def tx_enqueue(self, frame: bytes) -> bool:
        if len(self._pending_frames()) >= self.cfg.max_tx_frames:
            pol = self.cfg.backpressure
            if pol is BackpressurePolicy.BLOCK_NEVER:
                return False
            if pol is BackpressurePolicy.DROP_NEWEST:
                return False
            if pol is BackpressurePolicy.DROP_OLDEST:
                self._drop_tx_symbols()
            else:
                return False

        payload = frame[:255]
        frame_bytes = self._build_frame(payload)
        for sym in self._bytes_to_symbols(frame_bytes):
            self._tx_syms.append(sym)
        return True

    def rx_dequeue(self, limit: Optional[int] = None) -> List[bytes]:
        out: List[bytes] = []
        n = len(self._rx_frames) if limit is None else min(limit, len(self._rx_frames))
        for _ in range(n):
            out.append(self._rx_frames.popleft())
        return out

    def push_tx_block(self, t_ms: int) -> Int16Block:
        out = np.empty(self.BLK, dtype=np.int16)
        amp = float(self._amp)

        for idx in range(self.SYMS_PER_BLK):
            if self._tx_syms:
                sym = self._tx_syms.popleft()
            else:
                self.log("debug", f"push_tx_block: _tx_syms is NULL")
                sym = self._idle_symbol
            self._backend.modulate(self._mod_handle, sym, self._mod_tmp)

            phase_rot = np.complex64(math.cos(self._tx_phase) + 1j * math.sin(self._tx_phase))
            mixed = self._mod_tmp * (self._carrier_symbol * phase_rot)
            real_wave = mixed.real

            start = idx * self.SPS
            scaled = np.clip(np.round(real_wave * amp), -32767, 32767).astype(np.int16)
            out[start:start + self.SPS] = scaled

            self._tx_phase = (self._tx_phase + self._carrier_symbol_phase) % (2.0 * math.pi)

        return out

    def pull_rx_block(self, pcm: Int16Block, t_ms: int) -> None:
        norm = pcm.astype(np.float32) / float(self._amp)
        self._backend.hilbert_r2c(self._hilbert_handle, norm, self._analytic_tmp)
        phase_rot = np.complex64(math.cos(self._rx_phase) - 1j * math.sin(self._rx_phase))
        mix_block = self._carrier_block if self._rx_mix_sign < 0 else np.conj(self._carrier_block)
        if self._cfo_track and self._cfo_rps_est != 0.0:
            n = np.arange(self.BLK, dtype=np.float32)
            cfo_vec = np.exp(-1j * (self._cfo_rps_est * n)).astype(np.complex64)
            baseband = self._analytic_tmp * (mix_block * cfo_vec * phase_rot)
        else:
            baseband = self._analytic_tmp * (mix_block * phase_rot)
        self._rx_phase = (self._rx_phase + self._carrier_block_phase) % (2.0 * math.pi)

        if self._rx_buffer.size == 0:
            self._rx_buffer = baseband
        else:
            self._rx_buffer = np.concatenate((self._rx_buffer, baseband))

        block_syms: List[int] = []
        while self._rx_buffer.size >= self.SPS:
            chunk = np.ascontiguousarray(self._rx_buffer[:self.SPS])
            self._rx_buffer = self._rx_buffer[self.SPS:]
            sym = self._backend.demodulate(self._dem_handle, chunk)
            self._handle_symbol(sym)
            self._metric_rx_symbols += 1
            block_syms.append(int(sym))

        self._drain_frames()
        self._drain_frames_symbol()

        # Emit lightweight demod progress metric
        try:
            # CFO estimate (rad/sample -> Hz), robust angle of sum(conj(x[n]) * x[n+1])
            cfo_hz = 0.0
            try:
                v = np.vdot(baseband[:-1], baseband[1:])
                rad_per_sample = float(np.angle(v))
                cfo_hz = rad_per_sample * (self.SR / (2.0 * math.pi))
            except Exception:
                cfo_hz = 0.0
            self.log("metric", {
                "event": "demod",
                "bits_per_symbol": self.bits_per_symbol,
                "sps": self.SPS,
                "rx_syms": int(self._metric_rx_symbols),
                "rx_bytes": int(self._metric_rx_bytes_total),
                "preamble_hits": int(self._metric_preamble_hits),
                "frames": int(self._metric_frames_decoded),
                "rx_queue_frames": int(len(self._rx_frames)),
                "cfo_hz_est": float(cfo_hz),
                "sym_head": block_syms[:16],
                "sym_count": len(block_syms),
            })
        except Exception:
            pass

        # Update CFO estimate for next block
        if self._cfo_track:
            try:
                target_rps = float(cfo_hz) * (2.0 * math.pi) / float(self.SR)
                max_rps = float(self._cfo_max_hz) * (2.0 * math.pi) / float(self.SR)
                # clamp
                if target_rps > max_rps:
                    target_rps = max_rps
                elif target_rps < -max_rps:
                    target_rps = -max_rps
                a = self._cfo_alpha
                if a < 0.0:
                    a = 0.0
                if a > 1.0:
                    a = 1.0
                self._cfo_rps_est = (1.0 - a) * self._cfo_rps_est + a * target_rps
            except Exception:
                pass

    # ---------------------------------------------------------------- helpers
    def _handle_symbol(self, sym: int) -> None:
        # Always feed byte-oriented path (BFSK and generic fallback)
        bytes_out = list(self._bit_bucket.push(sym, self.bits_per_symbol))
        if bytes_out:
            self.log("debug", f"_handle_symbol: sym={sym} bytes_out={bytes_out}")
        for byte in bytes_out:
            self._rx_bytes.append(byte)
            self._metric_rx_bytes_total += 1
        # Additionally keep symbols for 4FSK symbol-domain parsing
        if self.bits_per_symbol == 2:
            self._rx_symbols.append(sym & 0x3)

    def _bytes_to_symbols(self, data: bytes) -> Iterable[int]:
        mask = (1 << self.bits_per_symbol) - 1
        out: List[int] = []
        accum = 0
        bits = 0
        for b in data:
            accum = (accum << 8) | b
            bits += 8
            while bits >= self.bits_per_symbol:
                shift = bits - self.bits_per_symbol
                out.append((accum >> shift) & mask)
                accum &= (1 << shift) - 1 if shift else 0
                bits -= self.bits_per_symbol
        if bits:
            out.append((accum << (self.bits_per_symbol - bits)) & mask)
        return out

    def _build_frame(self, payload: bytes) -> bytes:
        preamble = bytes([0x55] * 16)
        sync = bytes([0xD3, 0x91])
        ln = len(payload)
        checksum = (sum(payload) + ln) & 0xFF
        return preamble + sync + bytes([ln]) + payload + bytes([checksum])

    def _drain_frames(self) -> None:
        while len(self._rx_frames) > self.cfg.max_rx_frames:
            self._rx_frames.popleft()

        buf = self._rx_bytes
        self.log("debug", f"_drain_frames: rx_bytes length={len(buf)}")
        while True:
            if len(self._rx_frames) >= self.cfg.max_rx_frames:
                return
            if len(buf) < 18:
                return

            preamble_len = 0
            while preamble_len < len(buf) and buf[preamble_len] == 0x55:
                preamble_len += 1
            if preamble_len < 8:
                buf.popleft()
                continue

            # Ensure sync bytes are present after preamble
            if preamble_len + 2 > len(buf) - 1:
                return
            if buf[preamble_len] != 0xD3 or buf[preamble_len + 1] != 0x91:
                buf.popleft()
                continue

            # Peek the length without consuming; only pop once the full frame is present
            if preamble_len + 3 > len(buf):
                return
            ln = buf[preamble_len + 2]
            total_needed = preamble_len + 2 + 1 + ln + 1
            if len(buf) < total_needed:
                return

            # We have a plausible preamble + sync and full frame buffered
            self._metric_preamble_hits += 1

            # Consume preamble
            for _ in range(preamble_len):
                buf.popleft()

            # Consume header
            if buf.popleft() != 0xD3:
                continue
            if buf.popleft() != 0x91:
                continue

            ln2 = buf.popleft()
            if ln2 != ln:
                # Length changed while peeking; resync
                continue

            payload = bytes(buf.popleft() for _ in range(ln))
            checksum = buf.popleft()
            if ((sum(payload) + ln) & 0xFF) == checksum:
                if len(self._rx_frames) >= self.cfg.max_rx_frames:
                    self._rx_frames.popleft()
                self._rx_frames.append(payload)
                self._metric_frames_decoded += 1
                try:
                    self.log("metric", {"event": "frame_rx", "len": len(payload), "hex_head": payload[:16].hex(), "path": "byte"})
                except Exception:
                    pass

    def _pending_frames(self) -> List[int]:
        return [1] if self._tx_syms else []

    def _drop_tx_symbols(self) -> None:
        self._tx_syms.clear()

    # ---------------------------------------------------------------- symbol-level helpers (bps==2)
    def _symbols_to_byte(self, syms: List[int]) -> int:
        # Expect exactly 4 symbols (MSB-first groups)
        if len(syms) != 4:
            return -1
        return ((syms[0] & 0x3) << 6) | ((syms[1] & 0x3) << 4) | ((syms[2] & 0x3) << 2) | (syms[3] & 0x3)

    def _drain_frames_symbol(self) -> None:
        if self.bits_per_symbol != 2:
            return
        PRE_SYM = 1
        PRE_WIN = 32
        THRESH = 0.75

        while self._rx_symbols:
            s = self._rx_symbols.popleft()
            # update sliding window for preamble detection
            self._sym_pre_win.append(s)
            if s == PRE_SYM:
                self._sym_pre_count += 1
            if len(self._sym_pre_win) > PRE_WIN:
                old = self._sym_pre_win.popleft()
                if old == PRE_SYM:
                    self._sym_pre_count -= 1

            if self._sym_state == "search":
                if len(self._sym_pre_win) == PRE_WIN and (self._sym_pre_count / PRE_WIN) >= THRESH:
                    self._metric_preamble_hits += 1
                    self._sym_state = "sync"
                    self._sym_tmp_symbols.clear()
                else:
                    continue

            if self._sym_state == "sync":
                if len(self._rx_symbols) + len(self._sym_tmp_symbols) < 8:
                    return
                while len(self._sym_tmp_symbols) < 8 and self._rx_symbols:
                    self._sym_tmp_symbols.append(self._rx_symbols.popleft())
                b0 = self._symbols_to_byte(self._sym_tmp_symbols[:4])
                b1 = self._symbols_to_byte(self._sym_tmp_symbols[4:8])
                # Tolerant sync: accept if Hamming distance across both bytes <= threshold
                exp0, exp1 = 0xD3, 0x91
                hd = ((b0 ^ exp0) & 0xFF).bit_count() + ((b1 ^ exp1) & 0xFF).bit_count()
                SYNC_HD_MAX = 3
                if hd <= SYNC_HD_MAX:
                    self._sym_state = "len"
                    self._sym_tmp_symbols.clear()
                else:
                    last = self._sym_tmp_symbols[-1]
                    self._sym_tmp_symbols.clear()
                    self._sym_state = "search"
                    self._sym_pre_win.clear()
                    self._sym_pre_count = 1 if last == PRE_SYM else 0
                    if last == PRE_SYM:
                        self._sym_pre_win.append(last)
                continue

            if self._sym_state == "len":
                if len(self._rx_symbols) < 4:
                    return
                syms = [self._rx_symbols.popleft() for _ in range(4)]
                ln = self._symbols_to_byte(syms)
                if ln < 0:
                    self._sym_state = "search"
                    self._sym_pre_win.clear()
                    self._sym_pre_count = 0
                    continue
                self._sym_needed = ln
                self._sym_state = "payload"
                self._sym_expect = []
                continue

            if self._sym_state == "payload":
                need_syms = 4 * (self._sym_needed - len(self._sym_expect))
                if len(self._rx_symbols) < need_syms + 4:
                    return
                while len(self._sym_expect) < self._sym_needed:
                    syms = [self._rx_symbols.popleft() for _ in range(4)]
                    byte = self._symbols_to_byte(syms)
                    self._sym_expect.append(byte)
                c_syms = [self._rx_symbols.popleft() for _ in range(4)]
                cval = self._symbols_to_byte(c_syms)
                checksum = (sum(self._sym_expect) + len(self._sym_expect)) & 0xFF
                if checksum == cval:
                    if len(self._rx_frames) >= self.cfg.max_rx_frames:
                        self._rx_frames.popleft()
                    self._rx_frames.append(bytes(self._sym_expect))
                    self._metric_frames_decoded += 1
                    try:
                        self.log("metric", {"event": "frame_rx", "len": len(self._sym_expect), "hex_head": bytes(self._sym_expect[:16]).hex(), "path": "symbol"})
                    except Exception:
                        pass
                self._sym_state = "search"
                self._sym_pre_win.clear()
                self._sym_pre_count = 0
                self._sym_expect = []
                self._sym_tmp_symbols.clear()
                continue




