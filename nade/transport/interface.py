"""
Transport interface - abstracts the FEC + Modem layers.

The transport layer handles:
- Framing and deframing (preamble, sync, checksum)
- Modulation and demodulation (FSK)
- FEC encoding/decoding (future)

It presents a simple SDU-in / SDU-out interface to the protocol layer.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np
import numpy.typing as npt


Int16Block = npt.NDArray[np.int16]


@runtime_checkable
class ITransport(Protocol):
    """
    Transport layer interface.

    Hides FEC + Modem complexity from the protocol layer.
    All methods are non-blocking.
    """

    # === RX Path ===

    def feed_rx_samples(self, pcm: Int16Block, t_ms: int) -> list[bytes]:
        """
        Feed raw PCM samples to the transport.

        Returns list of decoded SDUs (may be empty if no complete frame yet).
        The transport handles demodulation and frame extraction internally.
        """
        ...

    # === TX Path ===

    def queue_tx_sdu(self, sdu: bytes) -> bool:
        """
        Queue an SDU for transmission.

        Returns True if queued, False if transport TX buffer is full.
        The transport handles framing and modulation internally.
        """
        ...

    def get_tx_samples(self, count: int, t_ms: int) -> Int16Block:
        """
        Get next `count` PCM samples for transmission.

        Returns int16 array of length `count`.
        If no data to send, returns silence (zeros or idle pattern).
        """
        ...

    # === Status ===

    def has_pending_tx(self) -> bool:
        """Check if there are SDUs or samples pending transmission."""
        ...

    # === Lifecycle ===

    def reset(self) -> None:
        """Reset transport state (clear queues, reset modem)."""
        ...
