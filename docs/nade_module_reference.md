# Nade Module Reference

This document explains, in plain terms, every module, class, and function in the `nade/` package, including key parameters and what they do. It avoids signal‑processing jargon where possible and focuses on how to use the APIs.

## Contents
- Overview
- `nade/__init__.py`
- `nade/audio.py`
- `nade/modems/imodem.py`
- `nade/modems/cpfsk.py`
- `nade/modems/fsk4.py`
- `nade/modems/__init__.py`
- `nade/crypto/noise_wrapper.py`
- Practical Parameter Notes
- Quick Start Examples

---

## Overview

- Audio blocks: fixed-size arrays of 16-bit audio samples (`int16`) that represent a small slice of time (e.g., 20 ms). DryBox pulls a TX block to play out and pushes an RX block to decode.
- Frames: application payloads (bytes) you want to send/receive. The modem converts frames ↔ audio blocks.

---

## `nade/__init__.py`

Public exports and version.

- Exports:
  - `AudioStack` — high-level façade for audio modem usage.
  - `NoiseXKWrapper` — helper for Noise XK encryption.
  - `LiquidBFSKModem`, `LiquidFourFSKModem` — concrete FSK modem implementations.
  - Alias `FourFSKModem = LiquidFourFSKModem` for backward compatibility.

---

## `nade/audio.py`

High-level, modem-agnostic façade matching DryBox’s Audio ABI.

### Class: `AudioStack`

Purpose: Manage a selected modem, provide bytes/text helpers, and expose the DryBox ABI calls.

- `__init__(modem: str = "4fsk", modem_cfg: Optional[Dict[str, Any]] = None, logger: Optional[Callable[[str, object], None]] = None)`
  - `modem`: Choose between `"bfsk"` or `"4fsk"`.
  - `modem_cfg`: Common modem config (sample rate, block size, queues, backpressure) plus modem-specific keys.
  - `logger(level, payload)`: Optional callback to receive structured logs/metrics.

- `set_modem(name: str, cfg_dict: Dict[str, Any]) -> None`
  - Switch to a new modem by name. Converts `cfg_dict` into `ModemConfig`, forwarding modem-specific keys (e.g., `samples_per_symbol`, `bandwidth`, `amp`, etc.). Raises `ValueError` for unknown names.

- `reconfigure(modem: Optional[str] = None, modem_cfg: Optional[Dict[str, Any]] = None) -> None`
  - Reconfigure the current modem with new settings or change to a different modem.

- `pull_tx_block(t_ms: int) -> np.ndarray`
  - Produce an `int16` audio block to transmit for the current time `t_ms` (milliseconds, monotonic).

- `push_rx_block(pcm: np.ndarray, t_ms: int) -> None`
  - Provide one received `int16` audio block to the modem for demodulation.

- `tx_enqueue(frame: bytes) -> bool`
  - Queue a payload to transmit. Returns `False` if backpressure rejects it.

- `pop_rx_frames(limit: Optional[int] = None) -> List[bytes]`
  - Fetch up to `limit` received frames (or all if `None`).

- `queue_text(text: str) -> bool`
  - Convenience: UTF‑8 encode `text` and queue as a frame.

- `pop_received_texts(limit: Optional[int] = None) -> list[str]`
  - Convenience: Pop frames and return those that decode as UTF‑8 strings.

- `_mk_modem_config(d: Dict[str, Any]) -> ModemConfig`
  - Builds `ModemConfig` with defaults tailored for DryBox 8 kHz / 160‑sample blocks.
  - Keys:
    - `sample_rate_hz` (default `8000`)
    - `block_size` (default `160`)
    - `max_tx_frames`, `max_rx_frames` (default `64`)
    - `backpressure` (`DROP_OLDEST` | `DROP_NEWEST` | `BLOCK_NEVER`, default `DROP_OLDEST`)
    - `abi_version` (default `1` — DryBox Audio ABI v1)

---

## `nade/modems/imodem.py`

Interface definitions and common configuration types.

- `Int16Block`
  - Numpy type alias: `np.ndarray[np.int16]` representing an audio block.

- `BackpressurePolicy`
  - `DROP_OLDEST`: Drop oldest queued TX data to make room.
  - `DROP_NEWEST`: Reject the new item when full.
  - `BLOCK_NEVER`: Non-blocking reject (functionally similar to dropping newest here).

- `@dataclass ModemConfig`
  - `sample_rate_hz: int` — Audio sample rate (e.g., 8000).
  - `block_size: int` — Samples per block (e.g., 160 for 20 ms at 8 kHz).
  - `max_tx_frames: int = 64`, `max_rx_frames: int = 64` — Queue capacities.
  - `backpressure: BackpressurePolicy = DROP_OLDEST` — Behavior when TX queue is full.
  - `abi_version: int = 1` — DryBox Audio ABI version.

- `Protocol IModem`
  - `configure(cfg: ModemConfig) -> None` — Apply runtime config; non-blocking.
  - `reset() -> None` — Clear internal state and queues.
  - `close() -> None` — Free resources.
  - `tx_enqueue(frame: bytes) -> bool` — Enqueue a frame; respect backpressure.
  - `rx_dequeue(limit: Optional[int]) -> list[bytes]` — Pop received frames.
  - `pull_tx_block(t_ms: int) -> Int16Block` — Produce the next TX audio block.
  - `push_rx_block(pcm: Int16Block, t_ms: int) -> None` — Consume an RX audio block.

---

## `nade/modems/cpfsk.py`

Concrete Continuous‑Phase FSK modem backed by `liquid-dsp` through `ctypes`.

### `_LiquidFSKLibrary`
Bridges to the `libliquid` shared library for a minimal subset of APIs.

- Requirements: Python `liquid` extension plus `libliquid` shared library discoverable next to it. If missing, construction raises `RuntimeError`.
- Singleton access: `instance()`.
- Modulator:
  - `create_mod(bits_per_symbol, sps, bandwidth)` → handle
  - `destroy_mod(handle)`
  - `modulate(handle, symbol, buf)` — Write one complex symbol waveform into `buf`.
- Demodulator:
  - `create_dem(bits_per_symbol, sps, bandwidth)` → handle
  - `destroy_dem(handle)`
  - `demodulate(handle, buf)` → `int` symbol index from one symbol’s worth of samples.
- Hilbert (analytic signal):
  - `create_hilbert(semi_length, attenuation_db)` → handle
  - `destroy_hilbert(handle)`, `reset_hilbert(handle)`
  - `hilbert_r2c(handle, samples, out)` — Convert real float samples to complex analytic samples.

### `_BitBucket`
Packs/unpacks streams of bits into bytes.

- `push(value: int, width: int) -> Iterable[int]` — Feed `width` bits; yields complete bytes when available.
- `clear()` — Reset buffer.

### `LiquidFSKModem`
A general M‑FSK modem (BFSK and 4FSK via parameters). Uses `_LiquidFSKLibrary` internally.

- Constructor: `__init__(cfg: Optional[ModemConfig] = None, logger: Optional[Callable[[str, object], None]] = None, **params)`
  - General parameters:
    - `bits_per_symbol` (int): 1 for BFSK, 2 for 4FSK (must be > 0, ≤ 4).
    - `samples_per_symbol` (int): Samples per symbol; larger is slower but more tolerant.
    - `bandwidth` (float in (0, 0.5)): Liquid‑DSP filter/separation knob.
    - `carrier_hz` (float): Center frequency (pitch) of the signal.
    - `amp` (int): Output amplitude; audio is clipped to int16 range.
    - `idle_symbol` (int): Symbol sent when no data is queued.
    - `rx_mix_sign` (int): Debug; -1 uses e^-j2πf (default), +1 uses e^+j2πf.
    - CFO tracking (optional):
      - `cfo_track` (bool): Enable coarse frequency‑offset tracking.
      - `cfo_alpha` (float 0..1): Smoothing factor.
      - `cfo_max_hz` (float): Clamp for correction.
  - `cfg: ModemConfig`: Sample rate, block size, queues, backpressure.
  - `logger(level, payload)`: Optional callback; emits `event=cfg`, `event=demod`, `event=frame_rx`.

- `configure(cfg: ModemConfig) -> None`
  - Validates `block_size % samples_per_symbol == 0`.
  - Rebuilds mod/dem handles, precomputes phase tables, resets buffers and counters.

- `reset() -> None`
  - Clears queues/buffers, resets phases, resets Hilbert state.

- `close() -> None`
  - Destroys native handles and frees resources.

- `tx_enqueue(frame: bytes) -> bool`
  - Enqueues a frame to transmit. Frame format:
    - Preamble: 16 bytes `0x55`
    - Sync: `0xD3, 0x91`
    - Length: 1 byte (0..255); payload is capped at 255 bytes
    - Payload: `frame[:255]`
    - Checksum: `(sum(payload) + length) & 0xFF`
  - Honors `BackpressurePolicy` when TX is “full”.

- `rx_dequeue(limit: Optional[int] = None) -> List[bytes]`
  - Pop up to `limit` received frames (FIFO).

- `pull_tx_block(t_ms: int) -> Int16Block`
  - Generates one `block_size` of `int16` audio by:
    - Taking one symbol at a time (or an idle symbol when queue is empty),
    - Getting one complex symbol waveform from `liquid-dsp`,
    - Mixing with a carrier, scaling by `amp`, writing into the block.

- `push_rx_block(pcm: Int16Block, t_ms: int) -> None`
  - Demodulates one RX block by:
    - Converting `pcm` to float and computing the analytic signal (Hilbert),
    - Mixing down by the carrier (with optional frequency‑offset correction),
    - Feeding exact `samples_per_symbol` chunks to the demodulator to recover symbol indices,
    - Assembling frames both in a byte‑domain parser and a 4FSK symbol‑domain parser,
    - Emitting demod metrics (symbols/bytes seen, preamble hits, decoded frames, queue size, CFO estimate).

- Helpers
  - `_handle_symbol(sym: int)` — Feeds the byte assembler and, when 4FSK, also the symbol‑domain assembler.
  - `_bytes_to_symbols(data: bytes)` — Packs bytes into M‑ary symbols (MSB‑first) per `bits_per_symbol`.
  - `_build_frame(payload: bytes)` — Creates the over‑the‑air frame (preamble + sync + length + payload + checksum).
  - `_drain_frames()` — Byte‑domain parser scanning a rolling byte buffer for preamble+sync, then validates and emits frames.
  - `_pending_frames()` — Indicates if any TX symbols remain.
  - `_drop_tx_symbols()` — Clears TX symbols (used for backpressure `DROP_OLDEST`).
  - 4FSK symbol‑domain (only when `bits_per_symbol == 2`):
    - `_symbols_to_byte(syms: List[int]) -> int` — Convert four 2‑bit symbols into one byte.
    - `_drain_frames_symbol()` — Sliding‑window preamble detection and tolerant sync; emits frames when checksum matches.

### Presets

- `LiquidBFSKModem`
  - Defaults: `bits_per_symbol=1`, `samples_per_symbol=80`, `bandwidth=0.12`, `carrier_hz=900.0`.

- `LiquidFourFSKModem`
  - Defaults: `bits_per_symbol=2`, `samples_per_symbol=40`, `bandwidth=0.18`, `carrier_hz=1300.0`.

---

## `nade/modems/fsk4.py`

Alias shim for backward compatibility.

- `FourFSKModem = LiquidFourFSKModem`.

---

## `nade/modems/__init__.py`

Re‑exports `LiquidBFSKModem` and `LiquidFourFSKModem` for convenience.

---

## `nade/crypto/noise_wrapper.py`

Noise XK handshake and AEAD via `dissononce`.

### Class: `NoiseXKWrapper`
- `__init__(keypair: KeyPair, peer_pubkey: Optional[PublicKey] = None, debug_callback: Optional[Callable[[str, None]], None] = None)`
  - `keypair`: Your X25519 keypair.
  - `peer_pubkey`: Peer’s X25519 public key (required if you are the initiator).
  - `debug_callback`: Optional logger for handshake progress.

- `start_handshake(initiator: bool)`
  - Begin a fresh XK handshake. As initiator, requires `peer_pubkey`.

- `process_handshake_message(data: bytes)`
  - Feed an incoming handshake message; may queue an outgoing message retrievable via `get_next_handshake_message()`.
  - On completion, creates send/receive cipher states.

- `get_next_handshake_message() -> Optional[bytes]`
  - Pop the next handshake message to send.

- `encrypt_sdu(ad: bytes, plaintext: bytes) -> bytes`
  - Encrypt a payload with associated data `ad` (after handshake completes).

- `decrypt_sdu(ad: bytes, ciphertext: bytes) -> bytes`
  - Decrypt a payload with associated data `ad` (after handshake completes).

- `begin_rekey()`
  - Reset and restart a new handshake with the same role.

---

## Practical Parameter Notes

- `bits_per_symbol`: How many bits each symbol carries. BFSK = 1 bit, 4FSK = 2 bits. Higher can be faster but more sensitive to channel impairments.
- `samples_per_symbol`: Duration of a symbol in samples. Larger values decrease rate but generally improve robustness.
- `bandwidth`: Liquid‑DSP tuning knob (normalized 0–0.5). Start within 0.1–0.2 for these presets.
- `carrier_hz`: Center pitch of the audio signal (e.g., 900–1300 Hz in presets).
- `amp`: Audio amplitude scaling; values too high clip at 16‑bit limits. Recommended to stay below full‑scale.
- `idle_symbol`: Symbol sent when no data is queued; keeps tone present on channel.
- `backpressure`: Behavior when TX is full: drop oldest, drop newest, or non‑blocking reject.
- CFO (`cfo_*`): Optional coarse frequency‑offset handling. Useful for slightly off‑tuned audio paths; otherwise ignore.

---

## Quick Start Examples

Create a stack, choose a modem, and send/receive text.

```python
from nade import AudioStack

# 4FSK defaults (8 kHz, 160-sample blocks)
stack = AudioStack(modem="4fsk", modem_cfg={
    "sample_rate_hz": 8000,
    "block_size": 160,
    # modem-specific overrides:
    "samples_per_symbol": 40,
    "bandwidth": 0.18,
    "carrier_hz": 1300.0,
})

# Queue text for transmission
stack.queue_text("hello world")

# Produce a TX audio block (call periodically)
tx_block = stack.pull_tx_block(t_ms=0)

# Feed an RX audio block (from capture path)
stack.push_rx_block(tx_block, t_ms=20)

# Retrieve decoded text (if looped back)
print(stack.pop_received_texts())
```

Recommended smoke config known to decode on an isolated loop:

```bash
# BFSK loopback-friendly settings
NADE_ADAPTER_CFG='{"modem":"bfsk","modem_cfg":{"samples_per_symbol":80,"bandwidth":0.12,"amp":12000}}'
```

If you’d like deeper tuning or scenario guidance, see the DryBox testing procedure in the repository guidelines.

