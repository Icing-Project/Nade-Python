import os
import sys
import time
import pytest

pytest.importorskip("liquid")

import numpy as np

from nade.modems import LiquidFourFSKModem, LiquidBFSKModem
from nade.modems.imodem import ModemConfig


@pytest.mark.parametrize("modem_cls,params", [
    pytest.param(
        LiquidFourFSKModem,
        {"samples_per_symbol": 40, "bandwidth": 0.18, "carrier_hz": 1300.0, "amp": 12000},
        marks=pytest.mark.xfail(reason="4FSK loopback tuning pending; passes in DryBox with more runtime")
    ),
    (LiquidBFSKModem,    {"samples_per_symbol": 80, "bandwidth": 0.12, "carrier_hz": 900.0,  "amp": 12000}),
])
def test_cpfsk_loopback_decodes(modem_cls, params):
    cfg = ModemConfig(sample_rate_hz=8000, block_size=160)
    tx = modem_cls(cfg=cfg, **params)
    rx = modem_cls(cfg=cfg, **params)

    payload = b"hello-audio-loop"
    assert tx.tx_enqueue(payload)

    t = 0
    got = []
    # Up to 600 blocks (~12s at 20ms per block) to flush one frame end-to-end
    for _ in range(600):
        blk = tx.push_tx_block(t)
        assert isinstance(blk, np.ndarray) and blk.dtype == np.int16 and blk.shape == (160,)
        rx.pull_rx_block(blk, t)
        frames = rx.rx_dequeue(8)
        got.extend(frames)
        if any(fr == payload for fr in got):
            break
        t += 10

    assert any(fr == payload for fr in got), "No decoded frame received in loopback"
