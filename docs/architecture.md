# Nade Architecture

This document describes the Nade protocol architecture, including both the current implementation and the planned future extensions.

## Overview

**Nade** (Noise-Authenticated Duplex Encryption) is a secure communication protocol designed for real-time audio channels. It combines:

- **Noise XK pattern** for mutual authentication and key exchange
- **FSK modulation** for audio-based transmission
- **Future: FEC** for error correction over lossy channels
- **Future: Codec** for audio compression

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              ADAPTERS                                    │
│                                                                          │
│  ┌─────────────────────┐              ┌─────────────────────┐           │
│  │   DryBox Adapter    │              │  Desktop App Adapter │           │
│  │  (clock-driven)     │              │   (event-driven)     │           │
│  │                     │              │   [FUTURE]           │           │
│  │  push_tx_block()    │              │                      │           │
│  │  pull_rx_block()    │              │  on_mic_samples()    │           │
│  └──────────┬──────────┘              └──────────┬───────────┘           │
│             │                                    │                       │
│             └──────────────┬─────────────────────┘                       │
│                            ↓                                             │
└────────────────────────────┼────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────┐
│                     NADE CORE (nade/)                                    │
│                            │                                             │
│  ┌─────────────────────────▼────────────────────────────────────────┐   │
│  │                      NadeEngine                                   │   │
│  │  - Executes actions from protocol                                │   │
│  │  - Bridges to crypto and transport                               │   │
│  │  - Manages timer requests                                        │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│  ┌──────────────────────────▼───────────────────────────────────────┐   │
│  │                   NadeProtocol (state machine)                    │   │
│  │  - Pure functional: step(state, event) → (state, actions)        │   │
│  │  - Clock-independent                                             │   │
│  │  - No I/O, no side effects                                       │   │
│  │                                                                   │   │
│  │  States: IDLE → HS_* → ESTABLISHED → [PAUSED] → [EXPIRED]        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                             │                                            │
│              ┌──────────────┴──────────────┐                            │
│              ↓                             ↓                            │
│  ┌─────────────────────┐       ┌───────────────────────────────────┐   │
│  │  NoiseXKWrapper     │       │      AudioTransport               │   │
│  │  (crypto)           │       │      (transport)                  │   │
│  │                     │       │                                   │   │
│  │  - Handshake        │       │  ┌─────────────────────────────┐  │   │
│  │  - Encrypt/Decrypt  │       │  │  AudioStack                 │  │   │
│  └─────────────────────┘       │  │  └─ IModem (BFSK/4-FSK)     │  │   │
│                                │  └─────────────────────────────┘  │   │
│                                └───────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### Protocol Layer (`nade/protocol/`)

The protocol layer implements a **pure functional state machine**:

```python
new_state, actions = NadeProtocol.step(state, event)
```

**Key files:**
- `state.py` - `NadeState` dataclass and `Phase` enum
- `events.py` - Event types (inputs to state machine)
- `actions.py` - Action types (outputs from state machine)
- `machine.py` - `NadeProtocol.step()` implementation

**Design principles:**
- Immutable state (frozen dataclasses)
- No I/O or side effects
- Clock-independent (reacts to events, not time)
- Testable without mocks

### Engine (`nade/engine.py`)

The engine **executes actions** produced by the protocol:

```python
engine = NadeEngine(crypto=noise, transport=audio_transport)
engine.feed_event(StartSession(role="initiator"))
```

**Responsibilities:**
- Feed events to the protocol
- Execute crypto actions (start handshake, encrypt, decrypt)
- Execute transport actions (send SDU, flush handshake)
- Manage timer requests (adapters handle actual timing)
- Invoke application callbacks

### Transport Layer (`nade/transport/`)

Abstracts the physical layer (modem + FEC):

```python
class ITransport(Protocol):
    def feed_rx_samples(pcm, t_ms) -> list[bytes]: ...
    def queue_tx_sdu(sdu) -> bool: ...
    def get_tx_samples(count, t_ms) -> ndarray: ...
```

**Current implementation:**
- `AudioTransport` wraps `AudioStack` with BFSK/4-FSK modems

### Crypto Layer (`nade/crypto/`)

Handles Noise XK handshake and AEAD encryption:

- `NoiseXKWrapper` - Current implementation using `dissononce`

### Adapters (`adapter/`)

Convert adapter-specific I/O model to engine events:

- **DryBox Adapter** - Clock-driven (`push_tx_block`/`pull_rx_block`)
- **Desktop App Adapter** [FUTURE] - Event-driven (`on_mic_samples`)

---

## Protocol State Machine

### Phases

```
                    ┌──────────────────────────────────────┐
                    │                IDLE                   │
                    └──────────────────┬───────────────────┘
                                       │ StartSession
              ┌────────────────────────┴────────────────────────┐
              ↓                                                 ↓
┌─────────────────────────┐                     ┌─────────────────────────┐
│  HS_INITIATOR_STARTING  │                     │  HS_RESPONDER_STARTING  │
│  → HS_INITIATOR_AWAIT_M2│                     │                         │
│  → HS_INITIATOR_AWAIT_CF│                     │  → HS_RESPONDER_AWAIT_M3│
└───────────┬─────────────┘                     └───────────┬─────────────┘
            │                                               │
            └───────────────────┬───────────────────────────┘
                                ↓
                    ┌───────────────────────────┐
                    │       ESTABLISHED         │
                    └─────┬───────────┬─────────┘
                          │           │
         ┌────────────────┘           └────────────────┐
         ↓                                             ↓
┌─────────────────┐                         ┌─────────────────┐
│     PAUSED      │ [FUTURE]                │    CLOSING      │ [FUTURE]
└────────┬────────┘                         └────────┬────────┘
         │                                           │
         └───────────────────┬───────────────────────┘
                             ↓
                    ┌─────────────────┐
                    │    EXPIRED      │
                    └─────────────────┘
```

### Events

| Event | Description |
|-------|-------------|
| `StartSession(role)` | Begin handshake as initiator or responder |
| `StopSession(reason)` | End session |
| `TransportRxReady(sdus)` | Transport decoded SDUs from physical layer |
| `TransportTxCapacity(budget)` | Transport has TX capacity |
| `AppSendData(payload)` | Application wants to send data |
| `TimerExpired(timer_id)` | A timer fired |
| `LinkQualityUpdate(...)` | [FUTURE] Link metrics for adaptive control |

### Actions

| Action | Description |
|--------|-------------|
| `CryptoStartHandshake(is_initiator)` | Initialize crypto handshake |
| `CryptoProcessMessage(data)` | Process handshake message |
| `CryptoEncrypt(plaintext)` | Encrypt and queue for TX |
| `CryptoDecrypt(ciphertext)` | Decrypt and deliver to app |
| `TransportSend(sdu)` | Send SDU via transport |
| `TransportFlushHandshake()` | Flush handshake messages to transport |
| `TimerStart(id, duration_ms)` | Request timer |
| `TimerCancel(id)` | Cancel timer |
| `AppDeliver(payload)` | Deliver data to application |
| `AppNotify(event_type, details)` | Notify app of protocol event |
| `Log(level, message)` | Emit log message |

---

## Future Architecture

### Full Stack Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application                               │
│                    (voice samples, text, files)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Codec Layer (ICodec)                         │
│                  Opus, Speex, PCM passthrough                    │
│                                                                  │
│  Compress audio before encryption, decompress after decryption  │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Crypto Layer (ICrypto)                       │
│                  Noise XK, [future alternatives]                 │
│                                                                  │
│  Handshake, AEAD encryption/decryption                          │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Transport Layer (ITransport)                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    FEC Layer (IFEC)                        │ │
│  │              Reed-Solomon, Convolutional, LDPC             │ │
│  │                                                            │ │
│  │  Add redundancy for error correction over lossy channels   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Modem Layer (IModem)                     │ │
│  │                BFSK, 4-FSK, 8-FSK, OFDM                    │ │
│  │                                                            │ │
│  │  Framing, modulation, demodulation                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
                    Physical medium (audio)
```

### Planned Interfaces

#### ICrypto

```python
class ICrypto(Protocol):
    """Crypto layer interface for future hot-swappable implementations."""

    def start_handshake(self, initiator: bool) -> None: ...
    def process_handshake_message(self, data: bytes) -> None: ...
    def get_next_handshake_message(self) -> bytes | None: ...

    @property
    def handshake_complete(self) -> bool: ...

    def encrypt(self, ad: bytes, plaintext: bytes) -> bytes: ...
    def decrypt(self, ad: bytes, ciphertext: bytes) -> bytes: ...

    def rekey(self) -> None: ...
```

#### ICodec

```python
class ICodec(Protocol):
    """Codec layer interface for audio compression."""

    def encode(self, pcm: ndarray) -> bytes: ...
    def decode(self, data: bytes) -> ndarray: ...

    @property
    def sample_rate(self) -> int: ...
    @property
    def frame_size(self) -> int: ...
    @property
    def bitrate(self) -> int: ...

    def set_bitrate(self, bitrate: int) -> None: ...
```

#### IFEC

```python
class IFEC(Protocol):
    """FEC layer interface for error correction."""

    def encode(self, data: bytes) -> bytes: ...
    def decode(self, data: bytes) -> tuple[bytes, int]: ...  # (data, errors_corrected)

    @property
    def redundancy_ratio(self) -> float: ...

    def set_redundancy(self, level: int) -> None: ...
```

### Adaptive Control

The state machine will control transport parameters based on link quality:

```python
# In NadeProtocol.step():
case (Phase.ESTABLISHED, LinkQualityUpdate(packet_loss_rate=plr)):
    if plr > 0.1:
        # High loss: increase FEC
        return (state, [TransportSetFEC(level=2)])
    elif plr < 0.01:
        # Low loss: reduce FEC for efficiency
        return (state, [TransportSetFEC(level=0)])
```

### Desktop App Adapter

Event-driven adapter for real-time audio I/O:

```python
class DesktopAdapter:
    """Event-driven adapter for desktop application."""

    async def on_mic_samples(self, pcm: ndarray) -> None:
        """Called when microphone provides audio samples."""
        # Feed to transport RX (mic = remote's TX)
        sdus = self._engine.feed_rx_samples(pcm, self._now_ms())
        if sdus:
            await self._engine.feed_event(TransportRxReady(tuple(sdus)))

    async def get_speaker_samples(self, count: int) -> ndarray:
        """Called when speaker needs audio samples."""
        # Get from transport TX (speaker = our TX to remote)
        return self._engine.get_tx_samples(count, self._now_ms())

    async def _timer_loop(self):
        """Background timer management."""
        while self._running:
            await asyncio.sleep(0.1)
            for timer in self._check_expired_timers():
                await self._engine.feed_event(TimerExpired(timer.id))
```

### Session Management

Future session lifecycle features:

- **PAUSED state**: No data received, session alive, can resume on first valid frame
- **Session expiry**: Configurable timeout after which session terminates
- **Keepalive**: Periodic empty frames to maintain session
- **Graceful shutdown**: CLOSING state for clean termination

---

## Data Flow

### TX Path (Application → Physical)

```
1. Application calls send_sdu(data)
2. Adapter feeds AppSendData event to engine
3. Protocol emits CryptoEncrypt action
4. Engine encrypts via NoiseXKWrapper
5. Engine queues ciphertext to transport
6. Adapter calls get_tx_samples()
7. Transport frames and modulates → PCM
8. Adapter sends PCM to physical layer (DryBox/speaker)
```

### RX Path (Physical → Application)

```
1. Adapter receives PCM from physical layer (DryBox/mic)
2. Adapter calls feed_rx_samples()
3. Transport demodulates and deframes → SDUs
4. Adapter feeds TransportRxReady event to engine
5. Protocol emits CryptoDecrypt action
6. Engine decrypts via NoiseXKWrapper
7. Engine invokes on_app_data callback
8. Application receives plaintext
```

---

## Testing Strategy

### Unit Testing (Protocol)

The pure functional protocol can be tested without any I/O:

```python
def test_initiator_handshake():
    state = NadeState()

    # Start session
    state, actions = NadeProtocol.step(state, StartSession(role="initiator"))
    assert state.phase == Phase.HS_INITIATOR_STARTING
    assert any(isinstance(a, CryptoStartHandshake) for a in actions)

    # Receive M2
    state, actions = NadeProtocol.step(state, TransportTxCapacity(255))
    state, actions = NadeProtocol.step(state, TransportRxReady((b"M2",)))
    assert state.phase == Phase.HS_INITIATOR_AWAITING_CONF
```

### Integration Testing (Engine)

Test with mock crypto/transport:

```python
def test_engine_handshake():
    crypto = MockCrypto()
    transport = MockTransport()
    engine = NadeEngine(crypto=crypto, transport=transport)

    engine.feed_event(StartSession(role="initiator"))
    assert crypto.handshake_started
    assert transport.has_pending_tx()
```

### End-to-End Testing (DryBox)

Use DryBox scenarios for full system testing:

```bash
uv run drybox-run \
  --scenario audio_isolated_loop.yaml \
  --left adapter.drybox_adapter:Adapter \
  --right adapter.drybox_adapter:Adapter
```

---

## Migration Notes

### From Old Architecture

The old adapter embedded protocol logic directly:

```python
# OLD: Logic scattered in adapter
if self._pending_handshake:
    self._pending_handshake = False
    self._noise.start_handshake(...)
```

The new architecture separates concerns:

```python
# NEW: Protocol logic in state machine
state, actions = NadeProtocol.step(state, StartSession(role="initiator"))
# Engine executes actions
for action in actions:
    engine._execute(action)
```

### Compatibility

- Byte mode: Unchanged (mock handshake for testing)
- Audio mode: Refactored to use NadeEngine
- External API: `send_sdu()`, `is_handshake_complete()` unchanged
