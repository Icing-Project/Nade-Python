# nade/core/modems/fsk4.py
from __future__ import annotations
from typing import List, Deque, Optional, Tuple, Callable, Dict, Any
from collections import deque
import math
import numpy as np

class FourFSKModem:
    """
    4-FSK paramétrable et neutre (pas d'accès DryBox ici).
    - 2 bits/symbole → 4 tons (indices 0..3).
    - Détection non-cohérente par Goertzel (par symbole).
    - Framing très simple: PREAMBLE (0x55 * N) + SYNC (2 octets) + LEN(1) + PAYLOAD + CHECKSUM(1)
      (checksum = somme(payload)+LEN mod 256)

    Config attendue (toutes ont des valeurs par défaut):
      {
        "sr": 8000,               # sample rate
        "block": 160,             # taille bloc audio DryBox
        "sps": 8,                 # samples per symbol
        "tones": [1000,1500,2000,2500],
        "amp": 9000,              # amplitude sortie int16
        "preamble_len": 16,       # nb d'octets 0x55
        "sync": [0xD3, 0x91],     # 2 octets
        "checksum": "sum",        # "sum" (mod256) — on pourra mettre "crc16" plus tard
        "metrics": True           # publier des métriques via logger("metric", {...})
      }

    Logger: callable(level: str, payload_or_str)
      - logger("info", "message texte")
      - logger("metric", { ... })
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, logger: Optional[Callable[[str, Any], None]] = None):
        self.log = logger or (lambda lvl, payload: None)
        self.cfg: Dict[str, Any] = {}
        self.reconfigure(cfg or {})

        # TX
        self._tx_syms: Deque[int] = deque()
        self._idle_symbol = 0

        # RX
        self._rx_syms: Deque[int] = deque()    # symboles 0..3
        self._rx_bytes: Deque[int] = deque()   # octets reconstruits
        self._rx_texts: Deque[str] = deque()

        # stats courtes pour métriques
        self._pow_hist: Deque[Tuple[float, float]] = deque(maxlen=64)  # (max_pow, mean_others)
        self._sync_hits: Deque[int] = deque(maxlen=128)                # 1 si sync trouvé dans la fenêtre, sinon 0

    # ---------- Configuration ----------
    def reconfigure(self, new_cfg: Dict[str, Any]) -> None:
        # merge avec défauts
        cfg = {
            "sr": 8000,
            "block": 160,
            "sps": 8,
            "tones": [1000.0, 1500.0, 2000.0, 2500.0],
            "amp": 9000,
            "preamble_len": 16,
            "sync": [0xD3, 0x91],
            "checksum": "sum",
            "metrics": True,
        }
        cfg.update(new_cfg or {})
        self.cfg = cfg

        self.SR: int = int(cfg["sr"])
        self.BLK: int = int(cfg["block"])
        self.SPS: int = int(cfg["sps"])
        self.SYMS_PER_BLK: int = self.BLK // self.SPS
        self.TONES: List[float] = [float(f) for f in cfg["tones"]]
        self.AMP: int = int(cfg["amp"])
        self.PREAMBLE: bytes = bytes([0x55] * int(cfg["preamble_len"]))
        sync = cfg["sync"]
        if not (isinstance(sync, (list, tuple)) and len(sync) == 2):
            raise ValueError("sync must be two bytes")
        self.SYNC: bytes = bytes([int(sync[0]) & 0xFF, int(sync[1]) & 0xFF])
        self.CHK: str = str(cfg["checksum"]).lower()
        self.METRICS: bool = bool(cfg.get("metrics", True))

        # pré-calculs
        self._waves = self._build_symbol_waves()
        self._coeffs = [2.0 * math.cos(2.0 * math.pi * f * (self.SPS / self.SR)) for f in self.TONES]

        self.log("info", f"[4FSK] reconfigured: sr={self.SR} blk={self.BLK} sps={self.SPS} tones={self.TONES} amp={self.AMP} preamble={len(self.PREAMBLE)}")

    # ---------- API pile audio ----------
    def queue_text(self, text: str) -> None:
        payload = text.encode("utf-8")[:255]
        frame = self._build_frame(payload)
        for sym in self._bytes_to_2bit_symbols(frame):
            self._tx_syms.append(sym)
        self.log("info", f"[4FSK] TX enqueue '{text}' ({len(payload)}B)")

    def pull_tx_block(self, t_ms: int) -> np.ndarray:
        out = np.empty(self.BLK, dtype=np.int16)
        p = 0
        for _ in range(self.SYMS_PER_BLK):
            sym = self._tx_syms.popleft() if self._tx_syms else self._idle_symbol
            out[p:p+self.SPS] = self._waves[sym]
            p += self.SPS
        if self.METRICS:
            # goodput estimé (2 bits/sym * SYMS_PER_BLK * SR/BLK par seconde) mais on publie par bloc
            used = min(self.SYMS_PER_BLK, len(self._tx_syms) + (0 if self._tx_syms else 0))
            gbps = (used * 2) / (self.BLK / self.SR)  # bits/s sur ce bloc
            self.log("metric", {"layer": "modem", "event": "tx_block", "goodput_bps": gbps})
        return out

    def push_rx_block(self, pcm: np.ndarray, t_ms: int) -> None:
        if pcm is None or len(pcm) < self.BLK:
            return
        x = pcm.astype(np.float32)

        max_pows = []
        mean_others = []

        idx = 0
        for _ in range(self.SYMS_PER_BLK):
            seg = x[idx:idx+self.SPS]
            idx += self.SPS
            sym, powers = self._detect_symbol(seg)
            self._rx_syms.append(sym)
            # stats
            m = float(max(powers))
            o = float(sum(powers) - m) / max(1.0, float(len(powers) - 1))
            max_pows.append(m)
            mean_others.append(o)

            # pack 4 symboles -> 1 octet
            if len(self._rx_syms) >= 4:
                b = ((self._rx_syms.popleft() & 3) << 6) \
                  | ((self._rx_syms.popleft() & 3) << 4) \
                  | ((self._rx_syms.popleft() & 3) << 2) \
                  | ((self._rx_syms.popleft() & 3) << 0)
                self._rx_bytes.append(b)

        # SNR & lock approximatifs
        if self.METRICS and max_pows:
            max_mean = sum(max_pows) / len(max_pows)
            oth_mean = sum(mean_others) / len(mean_others)
            snr_db = 10.0 * math.log10((max(1e-9, max_mean)) / (max(1e-9, oth_mean)))
            self._pow_hist.append((max_mean, oth_mean))

            lock = self._estimate_lock_ratio()
            self.log("metric", {"layer": "modem", "event": "rx_block", "snr_db_est": snr_db, "lock_ratio": lock})

        self._drain_frames()

    def on_timer(self, t_ms: int) -> None:
        pass

    def pop_received_texts(self) -> List[str]:
        items = list(self._rx_texts)
        self._rx_texts.clear()
        return items

    # ---------- Internes ----------
    def _build_symbol_waves(self) -> List[np.ndarray]:
        waves: List[np.ndarray] = []
        t = np.arange(self.SPS) / self.SR
        for f in self.TONES:
            w = (self.AMP * np.sin(2.0 * math.pi * f * t)).astype(np.int16)
            waves.append(w)
        return waves

    def _build_frame(self, payload: bytes) -> bytes:
        ln = len(payload)
        if self.CHK == "sum":
            cksum = (sum(payload) + ln) & 0xFF
        else:
            cksum = (sum(payload) + ln) & 0xFF  # placeholder (CRC16 plus tard)
        return self.PREAMBLE + self.SYNC + bytes([ln]) + payload + bytes([cksum])

    def _bytes_to_2bit_symbols(self, data: bytes) -> List[int]:
        out: List[int] = []
        for b in data:
            out.extend([(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, (b >> 0) & 3])
        return out

    def _detect_symbol(self, seg: np.ndarray) -> Tuple[int, List[float]]:
        powers: List[float] = []
        best_k = 0
        best_p = -1.0
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

    def _estimate_lock_ratio(self) -> float:
        """Heuristique: regarde si SYNC est observé régulièrement dans le tampon."""
        # Convertit le tampon d'octets en recherche rapide de SYNC
        buf = bytes(self._rx_bytes)
        hits = 0
        window = 0
        i = 0
        # on ne consomme pas, on scanne juste
        while i + 1 < len(buf):
            if buf[i] == self.SYNC[0] and buf[i+1] == self.SYNC[1]:
                hits += 1
                i += 2
            else:
                i += 1
            window += 1
            if window > 256:
                break
        self._sync_hits.append(1 if hits > 0 else 0)
        if not self._sync_hits:
            return 0.0
        return sum(self._sync_hits) / len(self._sync_hits)

    def _drain_frames(self) -> None:
        buf = self._rx_bytes
        while True:
            if len(buf) < (len(self.PREAMBLE) + 2 + 1):
                return

            # cherche SYNC avec tolérance de préambule variable (>= 8 octets 0x55)
            # 1) sauter les 0x55 initiaux (au moins 8 si dispo)
            pre = 0
            while pre < len(buf) and buf[pre] == 0x55:
                pre += 1
            if pre < 8:
                # pas assez de préambule: drop 1 octet et réessaie
                buf.popleft()
                continue

            # 2) SYNC présent ?
            if pre + 1 >= len(buf):
                return
            if buf[pre] != self.SYNC[0] or buf[pre+1] != self.SYNC[1]:
                # pattern non reconnu → drop 1 octet et réessaie
                buf.popleft()
                continue

            # 3) Consommer jusqu'au SYNC
            for _ in range(pre):
                buf.popleft()
            # consomme SYNC
            if len(buf) < 2:
                return
            if buf.popleft() != self.SYNC[0]: return
            if buf.popleft() != self.SYNC[1]: return

            if not buf:
                return
            ln = buf.popleft()
            need = ln + 1  # payload + checksum
            if len(buf) < need:
                # on attend plus de données
                return
            payload = bytes([buf.popleft() for _ in range(ln)])
            cksum = buf.popleft()

            ok = (((sum(payload) + ln) & 0xFF) == cksum)
            if ok:
                try:
                    text = payload.decode("utf-8", errors="replace")
                except Exception:
                    text = payload.decode("latin-1", errors="replace")
                self._rx_texts.append(text)
                self.log("info", f"[4FSK] RX text: '{text}'")
                if self.METRICS:
                    self.log("metric", {"layer": "modem", "event": "rx_text", "len": len(payload)})
            else:
                self.log("info", "[4FSK] checksum mismatch")
