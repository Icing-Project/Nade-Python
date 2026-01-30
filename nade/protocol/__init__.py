"""
Nade Protocol - Pure functional state machine.

This module contains the protocol logic separated from I/O concerns.
The NadeProtocol.step() function is the core: it takes state and event,
returns new state and actions to execute.
"""
from .state import NadeState, Phase
from .events import (
    Event,
    StartSession,
    StopSession,
    TransportRxReady,
    TransportTxCapacity,
    AppSendData,
    TimerExpired,
    LinkQualityUpdate,
)
from .actions import (
    Action,
    CryptoStartHandshake,
    CryptoProcessMessage,
    CryptoEncrypt,
    CryptoDecrypt,
    TransportSend,
    TransportFlushHandshake,
    TimerStart,
    TimerCancel,
    AppDeliver,
    AppNotify,
    Log,
)
from .machine import NadeProtocol

__all__ = [
    # State
    "NadeState",
    "Phase",
    # Events
    "Event",
    "StartSession",
    "StopSession",
    "TransportRxReady",
    "TransportTxCapacity",
    "AppSendData",
    "TimerExpired",
    "LinkQualityUpdate",
    # Actions
    "Action",
    "CryptoStartHandshake",
    "CryptoProcessMessage",
    "CryptoEncrypt",
    "CryptoDecrypt",
    "TransportSend",
    "TransportFlushHandshake",
    "TimerStart",
    "TimerCancel",
    "AppDeliver",
    "AppNotify",
    "Log",
    # Protocol
    "NadeProtocol",
]
