# nade/core/audio.py
from __future__ import annotations
from typing import Optional, List, Callable, Dict, Any
import numpy as np
from .modems.fsk4 import FourFSKModem


class AudioStack:
    """
    Façade Audio de Nade (configurable).
    """

    def __init__(self, modem: str = "4fsk", modem_cfg: Optional[Dict[str, Any]] = None,
                 logger: Optional[Callable[[str, Any], None]] = None):
        self.logger = logger or (lambda lvl, payload: None)
        self.modem_name = modem.lower()
        if self.modem_name == "4fsk":
            self.modem = FourFSKModem(cfg=modem_cfg or {}, logger=self.logger)
        else:
            raise ValueError(f"Unsupported modem '{modem}'")

    # ---- API consommée par l'adapter ----
    def pull_tx_block(self, t_ms: int) -> np.ndarray:
        return self.modem.pull_tx_block(t_ms)

    def push_rx_block(self, pcm, t_ms: int) -> None:
        self.modem.push_rx_block(pcm, t_ms)

    def on_timer(self, t_ms: int) -> None:
        self.modem.on_timer(t_ms)

    # ---- Contrôle (state machine, tests, etc.) ----
    def queue_text(self, text: str) -> None:
        self.modem.queue_text(text)

    def pop_received_texts(self) -> List[str]:
        return self.modem.pop_received_texts()

    def reconfigure(self, modem: Optional[str] = None, modem_cfg: Optional[Dict[str, Any]] = None) -> None:
        if modem is not None and modem.lower() != self.modem_name:
            raise ValueError("Changing modem kind at runtime is not supported yet")
        if modem_cfg is not None:
            self.modem.reconfigure(modem_cfg)
