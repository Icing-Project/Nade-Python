# nade/core/audio.py
from __future__ import annotations
from typing import Optional, List, Callable
import numpy as np
from .modems.fsk4 import FourFSKModem

class AudioStack:
    """
    Façade Audio de Nade: l’adapter parle à cette classe,
    et elle délègue au modem 4-FSK (ou autres plus tard).
    """

    def __init__(self, modem: str = "4fsk", logger: Optional[Callable[[str,str],None]] = None):
        self.logger = logger or (lambda lvl, msg: None)
        if modem.lower() == "4fsk":
            self.modem = FourFSKModem(logger=self.logger)
        else:
            raise ValueError(f"Unsupported modem '{modem}'")

    # ---- API consommée par l’adapter ----
    def pull_tx_block(self, t_ms: int) -> np.ndarray:
        return self.modem.pull_tx_block(t_ms)

    def push_rx_block(self, pcm, t_ms: int) -> None:
        self.modem.push_rx_block(pcm, t_ms)

    def on_timer(self, t_ms: int) -> None:
        self.modem.on_timer(t_ms)

    # ---- Convenience pour les tests démo ----
    def queue_text(self, text: str) -> None:
        self.modem.queue_text(text)

    def pop_received_texts(self) -> List[str]:
        return self.modem.pop_received_texts()
