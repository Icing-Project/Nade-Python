# CPFSK Newbie Guide

This document explains how the CPFSK audio modem works in this repo, using a concrete DryBox example and plain language. It is written for developers with little or no signal processing background.

Relevant files:
- `nade/modems/cpfsk.py`
- `nade/modems/fsk2.py`
- `nade/modems/fsk4.py`
- `nade/audio.py`
- `adapter/drybox_adapter.py`

## Quick Start (Mental Model)

You can think of the audio modem as a pipeline:

1) Text -> bytes (UTF-8)
2) Bytes -> frame (preamble + sync + length + payload + checksum)
3) Frame bytes -> symbols (1 or 2 bits per symbol)
4) Symbols -> audio samples (PCM int16)
5) On receive: audio samples -> symbols -> bytes -> frame -> text

That is all the modem does: it moves bytes over audio by encoding them as changes in frequency.

## What CPFSK Means

CPFSK = continuous-phase frequency shift keying.

- "FSK" means you represent data using different tones (frequencies).
- "Continuous-phase" means the waveform does not jump or reset phase between symbols.
  That makes the signal smoother and easier to demodulate.

Liquid-dsp handles the math for the CPFSK waveform. This file orchestrates the flow.

## Core Concepts (No Math Needed)

### PCM audio
- PCM is just a stream of numbers that represent sound over time.
- Here it uses 16-bit signed integers (range -32768..32767).
- The default audio rate is 8000 samples per second.

### Symbols vs bits
- A symbol is the smallest unit sent over audio.
- Each symbol carries `bits_per_symbol` bits.
  - 1 bit per symbol -> 2-FSK (BFSK): two tones.
  - 2 bits per symbol -> 4-FSK: four tones.
- `samples_per_symbol` controls how long a symbol lasts.

### Complex signals and carrier
- The modem uses complex numbers internally for modulation.
- It then shifts the signal to an audible carrier frequency (e.g. 900 or 1300 Hz).
- Only the real part is sent as PCM.

### Hilbert transform
- Incoming audio is real numbers.
- A Hilbert transform turns that into a complex signal that is easier to demodulate.

### Framing
Data is not sent raw. It is framed like this:

- Preamble: 16 bytes of 0x55
- Sync: 0xD3 0x91
- Length: 1 byte
- Payload: up to 255 bytes
- Checksum: 1 byte = (sum(payload) + length) & 0xFF

The preamble and sync help the receiver find the start of a frame.

## Where the Message Comes From (DryBox Audio Mode)

In `adapter/drybox_adapter.py`, the adapter auto-queues a demo text in audio mode:

- Default text: `Hello from Nade-Python Audio mode!`
- It is sent as UTF-8 bytes via `AudioStack.queue_text()`.

`AudioStack` is in `nade/audio.py`. It forwards the bytes to the selected modem.

## Default Modems and Parameters

`AudioStack` selects a modem by name:

- `bfsk` -> `LiquidBFSKModem` (`nade/modems/fsk2.py`)
  - bits_per_symbol = 1
  - samples_per_symbol = 80
  - carrier_hz = 900
  - bandwidth = 0.12

- `4fsk` -> `LiquidFourFSKModem` (`nade/modems/fsk4.py`)
  - bits_per_symbol = 2
  - samples_per_symbol = 40
  - carrier_hz = 1300
  - bandwidth = 0.18

Both use the same implementation in `nade/modems/cpfsk.py`.

## Concrete Example (BFSK)

Message:

- `Hello from Nade-Python Audio mode!`
- UTF-8 bytes length = 34

Frame bytes are:

- Preamble: 16 * 0x55
- Sync: 0xD3 0x91
- Length: 0x22 (34)
- Payload: ASCII bytes for the string
- Checksum: 0x29

Full frame hex (54 bytes total):

```
55555555555555555555555555555555d39122
48656c6c6f2066726f6d204e6164652d507974686f6e20417564696f206d6f646521
29
```

How symbols are built for BFSK:

- BFSK uses 1 bit per symbol.
- Each byte becomes 8 symbols (MSB-first).
- 54 bytes * 8 = 432 symbols total.

Timing with BFSK defaults:

- samples_per_symbol = 80
- sample_rate = 8000
- 80/8000 = 10 ms per symbol
- 432 symbols = 4.32 seconds of audio
- block_size = 160 samples -> 2 symbols per block
- 432 symbols / 2 = 216 blocks

TX path (inside `LiquidFSKModem.pull_tx_block`):

1) For each symbol, liquid-dsp creates CPFSK baseband samples.
2) They are mixed to a carrier (900 Hz).
3) The real part is scaled to int16 PCM.

RX path (inside `LiquidFSKModem.push_rx_block`):

1) PCM -> float -> Hilbert -> complex signal.
2) Mix down from carrier to baseband.
3) Demodulate symbols.
4) Rebuild bytes and parse frames.
5) Emit `text_rx` once a frame checks out.

## Concrete Example (4-FSK)

Same message and same frame bytes. The difference is symbolization and timing.

4-FSK uses 2 bits per symbol:

- Each byte becomes 4 symbols (MSB-first, 2-bit groups).
- 54 bytes * 4 = 216 symbols total.

Examples:

- 0x55 = 01010101 -> 01 01 01 01 -> symbols [1, 1, 1, 1]
- 0xD3 = 11010011 -> 11 01 00 11 -> [3, 1, 0, 3]
- 0x91 = 10010001 -> 10 01 00 01 -> [2, 1, 0, 1]
- 0x22 = 00100010 -> 00 10 00 10 -> [0, 2, 0, 2]
- 'H' = 0x48 = 01001000 -> 01 00 10 00 -> [1, 0, 2, 0]

Timing with 4-FSK defaults:

- samples_per_symbol = 40
- sample_rate = 8000
- 40/8000 = 5 ms per symbol
- 216 symbols = 1.08 seconds of audio
- block_size = 160 samples -> 4 symbols per block
- 216 symbols / 4 = 54 blocks

Why there is a special 4-FSK parser:

`cpfsk.py` runs a byte parser for all modes and an extra symbol parser for 2 bits per symbol.

- The symbol parser detects the preamble by looking for symbol `1` repeated.
- It tolerates some errors in the sync bytes (Hamming distance).
- This makes 4-FSK more robust in noisy conditions.

## What Happens During TX and RX (Functions to Know)

In `nade/modems/cpfsk.py`:

- `tx_enqueue(frame)`:
  - Builds a framed packet and pushes symbols into the TX queue.

- `pull_tx_block(t_ms)`:
  - Produces one PCM block (int16 array) for DryBox.

- `push_rx_block(pcm, t_ms)`:
  - Demodulates PCM into symbols, rebuilds bytes, and extracts frames.

- `_drain_frames()`:
  - Byte-based frame parser.

- `_drain_frames_symbol()`:
  - Symbol-based frame parser for 4-FSK.

## Useful Configuration Knobs

You can override the modem without code changes:

```
NADE_ADAPTER_CFG='{"modem":"4fsk","modem_cfg":{"samples_per_symbol":40,"bandwidth":0.18,"carrier_hz":1300,"amp":9000}}'
```

Common values:
- `modem`: `bfsk` or `4fsk`
- `samples_per_symbol`: larger = slower but more robust
- `bandwidth`: lower = narrower signal, can reduce noise
- `carrier_hz`: tone center frequency (audible)
- `amp`: output volume (int16 scaling)

## How to Read Metrics During a Run

`cpfsk.py` emits metrics via `self.log("metric", ...)`, which the adapter forwards as DryBox events:

- `event=demod`
- `rx_syms`: number of symbols demodulated
- `rx_bytes`: number of bytes reconstructed
- `preamble_hits`: preambles detected
- `frames`: frames decoded

If you see `preamble_hits` increasing but `frames` stuck at 0, the sync or checksum is likely failing.

## Common Failure Modes (Simple Checks)

- No frames decoded:
  - Ensure modem and config match on both ends.
  - Try increasing `samples_per_symbol` or lowering `bandwidth`.
  - Check `amp` and make sure PCM is not clipping.

- Garbled text:
  - Check `carrier_hz` mismatch.
  - Inspect `sym_head` in metrics to see if symbols are random.

- Very slow transmission:
  - BFSK is slower than 4-FSK for the same samples per symbol.
  - Increase bits per symbol (use 4-FSK) or reduce samples per symbol.

## Summary

The CPFSK modem is a simple byte-to-audio pipeline:

- Frames and preambles make synchronization possible.
- Symbols encode bits as tones.
- CPFSK keeps the waveform smooth.
- The receiver reverses the process and validates with a checksum.

Once you understand the framing and symbolization, the rest is just orchestration around liquid-dsp.
