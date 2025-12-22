from __future__ import annotations

import pytest

from nade.modems import LiquidBFSKModem, LiquidFourFSKModem
from nade.modems.imodem import ModemConfig


def _loopback_once(modem, payload: bytes) -> list[bytes]:
    assert modem.tx_enqueue(payload)
    received: list[bytes] = []

    for tick in range(120):
        block = modem.push_tx_block(tick * 20)
        modem.pull_rx_block(block, tick * 20)
        received.extend(modem.rx_dequeue())
        if received:
            break

    modem.close()
    return received


@pytest.mark.parametrize("modem_cls,payload", [
    (LiquidFourFSKModem, b"nade-four-fsk"),
    (LiquidBFSKModem, b"nade-bfsk"),
])
def test_cpfsk_loopback(modem_cls, payload: bytes) -> None:
    modem = modem_cls(ModemConfig(sample_rate_hz=8000, block_size=160))
    received = _loopback_once(modem, payload)
    assert payload in received
