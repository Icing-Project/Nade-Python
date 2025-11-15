from typing import Callable, Optional

from .imodem import ModemConfig
from .cpfsk import LiquidFSKModem


class LiquidFourFSKModem(LiquidFSKModem):
    def __init__(self, cfg: Optional[ModemConfig] = None,
                 logger: Optional[Callable[[str, object], None]] = None,
                 **params: object) -> None:
        params.setdefault("bits_per_symbol", 2)
        params.setdefault("samples_per_symbol", 40)
        params.setdefault("bandwidth", 0.18)
        params.setdefault("carrier_hz", 1300.0)
        super().__init__(cfg=cfg, logger=logger, **params)


__all__ = ["LiquidFourFSKModem"]
