"""
Audio transport implementation.

Wraps the existing AudioStack/modem to provide ITransport interface.
"""
from __future__ import annotations

from typing import Callable, Any
import numpy as np

from .interface import ITransport, Int16Block
from ..audio import AudioStack


class AudioTransport(ITransport):
    """
    Audio transport using FSK modems.

    This is a thin wrapper around the existing AudioStack that
    conforms to the ITransport interface.
    """

    def __init__(
        self,
        modem: str = "bfsk",
        modem_cfg: dict[str, Any] | None = None,
        logger: Callable[[str, object], None] | None = None,
    ):
        self._logger = logger or (lambda lvl, msg: None)
        self._audio_stack = AudioStack(
            modem=modem,
            modem_cfg=modem_cfg or {},
            logger=self._logger,
        )

    # === RX Path ===

    def feed_rx_samples(self, pcm: Int16Block, t_ms: int) -> list[bytes]:
        """Feed PCM samples, return any decoded SDUs."""
        self._audio_stack.pull_rx_block(pcm, t_ms)
        return self._audio_stack.pop_rx_frames()

    # === TX Path ===

    def queue_tx_sdu(self, sdu: bytes) -> bool:
        """Queue SDU for transmission."""
        return self._audio_stack.tx_enqueue(sdu)

    def get_tx_samples(self, count: int, t_ms: int) -> Int16Block:
        """Get modulated PCM samples."""
        # Note: AudioStack.push_tx_block returns block_size samples
        # which matches DryBox's expected 160 samples at 8kHz
        pcm = self._audio_stack.push_tx_block(t_ms)

        # Normalize PCM to avoid clipping
        if pcm is not None and pcm.size > 0:
            max_abs = np.max(np.abs(pcm))
            if max_abs > 0:
                pcm = (pcm.astype(np.float32) / max_abs) * 32767
                pcm = pcm.astype(np.int16)

        return pcm

    # === Status ===

    def has_pending_tx(self) -> bool:
        """Check if transport has pending TX data."""
        # Access internal modem state to check TX queue
        # This is a simplified check; the modem may have symbols in flight
        return False  # AudioStack doesn't expose this directly

    # === Lifecycle ===

    def reset(self) -> None:
        """Reset transport state."""
        self._audio_stack.modem.reset()
