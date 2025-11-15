**Audio Smoke Test Plan**

Goal
- Validate DryBox Mode B (AudioBlock) with the Nade adapter and demodulate a simple text end-to-end.

Environment
- Single Python 3.11 environment for both projects or DryBox’s venv with `liquid-dsp` installed.
- DryBox repo as sibling `../DryBox`.

Setup
- Install both projects into one env (recommended):
  - `uv run pip install -e .`
  - `uv run pip install -e ../DryBox`
  - `uv run pip install liquid-dsp`
- Alternatively, use DryBox’s venv:
  - `../DryBox/.venv/bin/python -m ensurepip --upgrade`
  - `../DryBox/.venv/bin/pip install -e .`
  - `../DryBox/.venv/bin/pip install liquid-dsp`

Baseline Runs
- ByteLink: `uv run drybox-run --scenario ../DryBox/drybox/scenarios/bytelink_volte_smoke.yaml --left adapter.drybox_adapter:Adapter --right adapter.drybox_adapter:Adapter --out runs/drybox_nade_bytelink_smoke --tick-ms 10 --seed 123 --no-ui`
- Audio (isolated loop): `uv run drybox-run --scenario ../DryBox/drybox/scenarios/audio_isolated_loop.yaml --left adapter.drybox_adapter:Adapter --right adapter.drybox_adapter:Adapter --out runs/drybox_nade_audio_isolated --tick-ms 10 --seed 123 --no-ui`

Expected Outputs
- `metrics.csv`: regular `audioblock` tx/rx records per tick.
- `events.jsonl`:
  - `log` with modem config (`cfg`)
  - `text_tx` (adapter injects a hello message at start)
  - `text_rx` once demod decodes a frame into text
  - You can compare TX vs RX quickly with: `python tools/compare_text_run.py <runs/.../events.jsonl>`

If Audio Fails to Decode
- Likely parameter mismatch or insufficient SNR; the DSP backend is present if `liquid` imports.
- Tune modem parameters and add instrumentation.

Instrumentation
- Emit metrics from the modem/logger: symbol count, rx byte count, sync/preamble detections, frames decoded.
- Inspect `events.jsonl` for progress toward frame assembly.

Parameter Sweep (Quick)
- Try conservative BFSK and 4FSK points:
  - BFSK: `samples_per_symbol` ∈ {64, 80}, `bandwidth` ∈ {0.12}
  - 4FSK: `samples_per_symbol` ∈ {40, 64}, `bandwidth` ∈ {0.18, 0.12}
- Use environment override to pass adapter config:
  - `NADE_ADAPTER_CFG='{"modem": "bfsk", "modem_cfg": {"samples_per_symbol": 80, "bandwidth": 0.12}}'`
  - Then run the isolated audio scenario.
 - Optional knobs for difficult cases:
   - `amp`: output amplitude scaling (e.g., 12000–16000)
   - `rx_mix_sign`: `-1` or `+1` to flip RX mixing sign
   - `cfo_track`: `true` to enable simple CFO tracking (with `cfo_alpha`, `cfo_max_hz`)

Unit Test (Loopback)
- Add `tests/modems/test_cpfsk_loop.py` to modulate a payload with Liquid(B|4)FSK at 8 kHz / 160-sample blocks and demodulate on a separate instance; assert payload received. Skip if `import liquid` fails.

Action Plan
1) Add loopback test to validate modem encode/decode in isolation.
2) Enable adapter config passthrough from env to tune without code changes.
3) Run quick parameter sweep; select the first config that yields a `text_rx`.
4) If still failing, add demod instrumentation (events/metrics) and revisit baseband/hilbert mixing and preamble detection logic.

Success Criteria
- `events.jsonl` contains at least one `text_rx` with the injected hello message in the isolated audio loop scenario.

Working Findings (current)
- Demod progress is visible via `metric` events emitted by the modem logger (`event: demod`). Key fields: `rx_syms`, `rx_bytes`, `preamble_hits`, `frames`.
- A conservative configuration that produces `text_rx` on the isolated loop:
  - `modem: bfsk`
  - `modem_cfg: { samples_per_symbol: 80, bandwidth: 0.12, amp: 12000 }`
  - Set via env override: `NADE_ADAPTER_CFG='{"modem":"bfsk","modem_cfg":{"samples_per_symbol":80,"bandwidth":0.12,"amp":12000}}'`
- Notes:
  - Reassembly logic now waits until the full frame (preamble+sync+len+payload+checksum) is buffered before consuming header bytes, preventing partial consumption mid-frame.
  - For 4FSK, initial sweeps (sps=40, bw=0.12/0.18, amp up to 16000, rx_mix_sign toggled, optional `cfo_track`) did not yet produce `text_rx` within 5 s. Demod metrics show rising `rx_bytes` but `preamble_hits=0`. Use longer runs (10–15 s) and adjust carrier or enable `cfo_track`.
