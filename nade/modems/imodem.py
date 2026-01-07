from __future__ import annotations
from typing import Protocol, Optional, runtime_checkable
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import numpy.typing as npt

Int16Block = npt.NDArray[np.int16]  # Must be C-contiguous, length==block_size

class BackpressurePolicy(Enum):
    DROP_OLDEST = auto()
    DROP_NEWEST = auto()
    BLOCK_NEVER = auto()  # non-blocking, return False

@dataclass(frozen=True)
class ModemConfig:
    sample_rate_hz: int
    block_size: int                 # BLOCK
    max_tx_frames: int = 64
    max_rx_frames: int = 64
    backpressure: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST
    abi_version: int = 1            # DBX-ABI v1
    # add modem-specific fields as needed (e.g., tones, sps, amp, sync…)

@runtime_checkable
class IModem(Protocol):
    """
    Minimal, side-effect-free modem interface for DryBox.
    Time values MUST be monotonic (same clock as DryBox), in milliseconds since an arbitrary start.
    Audio cadence is driven by DryBox via push_tx_block()/pull_rx_block().
    All methods MUST be non-blocking.
    """

    # ---- Capability / lifecycle ----
    def configure(self, cfg: ModemConfig) -> None: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...

    # ---- App frames (bytes) ----
    # Returns True if enqueued, False if dropped due to backpressure.
    def tx_enqueue(self, frame: bytes) -> bool: ...
    # Dequeues up to `limit` frames (or all if None).
    def rx_dequeue(self, limit: Optional[int] = None) -> list[bytes]: ...

    # ---- AudioBlock API (DBX-ABI v1) ----
    # Must return a C-contiguous int16 array of length == cfg.block_size.
    def push_tx_block(self, t_ms: int) -> Int16Block: ...
    # Accepts a C-contiguous int16 array of length == cfg.block_size.
    def pull_rx_block(self, pcm: Int16Block, t_ms: int) -> None: ...
