# nade/core/audio.py
from __future__ import annotations
from typing import Optional, Dict, Any, Type, Callable, List
import numpy as np

from .modems.imodem import IModem, ModemConfig, BackpressurePolicy
from .modems.fsk4 import FourFSKModem

# registry
_MODEMS: dict[str, Type[IModem]] = {
    "4fsk": FourFSKModem,
}

class AudioStack:
    """
    Modem-agnostic façade (DBX-ABI v1) with bytes-first API + text helpers.
    """

    def __init__(self,
                 modem: str = "4fsk",
                 modem_cfg: Optional[Dict[str, Any]] = None,
                 logger: Optional[Callable[[str, object], None]] = None):
        self.logger = logger or (lambda lvl, payload: None)
        self.set_modem(modem, modem_cfg or {})

    # ---- modem selection / reconfiguration ----------------------------------
    def set_modem(self, name: str, cfg_dict: Dict[str, Any]) -> None:
        cls = _MODEMS.get(name.lower())
        if not cls:
            raise ValueError(f"Unsupported modem '{name}'. Available: {list(_MODEMS)}")

        mc = self._mk_modem_config(cfg_dict)
        # pass through modem-specific params (tones/sps/amp/…)
        modem_specific = {k: v for k, v in cfg_dict.items()
                          if k not in {"sample_rate_hz", "block_size", "max_tx_frames",
                                       "max_rx_frames", "backpressure", "abi_version"}}
        self.modem: IModem = cls(cfg=mc, logger=self.logger, **modem_specific)  # type: ignore[arg-type]
        self.modem_name = name.lower()

    def reconfigure(self, modem: Optional[str] = None, modem_cfg: Optional[Dict[str, Any]] = None) -> None:
        if modem is None or modem.lower() == self.modem_name:
            if modem_cfg:
                self.modem.configure(self._mk_modem_config(modem_cfg))
            return
        self.set_modem(modem, modem_cfg or {})

    # ---- DBX-ABI v1 ---------------------------------------------------------
    def pull_tx_block(self, t_ms: int) -> np.ndarray:
        return self.modem.pull_tx_block(t_ms)

    def push_rx_block(self, pcm: np.ndarray, t_ms: int) -> None:
        self.modem.push_rx_block(pcm, t_ms)

    def on_timer(self, t_ms: int) -> None:
        self.modem.on_timer(t_ms)

    # ---- byte API -----------------------------------------------------------
    def tx_enqueue(self, frame: bytes) -> bool:
        return self.modem.tx_enqueue(frame)

    def pop_rx_frames(self, limit: Optional[int] = None) -> List[bytes]:
        return self.modem.rx_dequeue(limit)

    # ---- convenience for text-only tests -----------------------------------
    def queue_text(self, text: str) -> bool:
        return self.modem.tx_enqueue(text.encode("utf-8"))

    def pop_received_texts(self, limit: Optional[int] = None) -> list[str]:
        out: list[str] = []
        for fr in self.modem.rx_dequeue(limit):
            try:
                out.append(fr.decode("utf-8"))
            except Exception:
                pass
        return out

    # ---- helpers ------------------------------------------------------------
    def _mk_modem_config(self, d: Dict[str, Any]) -> ModemConfig:
        # sensible defaults matching DryBox 8k/20ms blocks
        sr = int(d.get("sample_rate_hz", 8000))
        bs = int(d.get("block_size", 160))
        max_tx = int(d.get("max_tx_frames", 64))
        max_rx = int(d.get("max_rx_frames", 64))
        bp = d.get("backpressure", BackpressurePolicy.DROP_OLDEST)
        if isinstance(bp, str):
            bp = BackpressurePolicy[bp]
        return ModemConfig(
            sample_rate_hz=sr,
            block_size=bs,
            max_tx_frames=max_tx,
            max_rx_frames=max_rx,
            backpressure=bp,  # type: ignore[arg-type]
            abi_version=int(d.get("abi_version", 1)),
        )
