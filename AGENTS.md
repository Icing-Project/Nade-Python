# Repository Guidelines

## Project Structure & Module Organization
- `nade/` hosts the protocol stack: `audio.py` orchestrates modem selection, `crypto/` wraps Noise building blocks, `fec/` and `modems/` handle signal processing, and `preprocessing/` stores reusable helpers.
- `adapter/` exposes the DryBox integration; `drybox_adapter.py` binds byte and audio modes while `modes/` provides runnable presets.
- `runs/` captures local experiments; prune noisy artifacts before committing.

## Build, Test, and Development Commands
- `uv sync` installs the locked Python 3.11 dependency set.
- `uv run pip install -e .` publishes the package locally so DryBox discovers `nade-python`.
- `uv run python -m adapter.drybox_adapter` performs a quick import and smoke check of the adapter entry points.

## DryBox Testing Procedure
- Prereqs:
  - Working dir at `Nade-Python`.
  - Sibling checkout of `../DryBox` (has its own `.venv`).
  - For Audio mode only: `liquid-dsp` wheel must be installed in the Python environment used to run DryBox.

- Option A — Use DryBox’s virtualenv (recommended, avoids extra installs):
  - ByteLink (low-deformation smoke):
    - `PYTHONPATH=../DryBox:. ../DryBox/.venv/bin/python -m drybox.core.runner --scenario ../DryBox/drybox/scenarios/bytelink_volte_smoke.yaml --left $(pwd)/adapter/drybox_adapter.py:Adapter --right $(pwd)/adapter/drybox_adapter.py:Adapter --out runs/drybox_nade_bytelink_smoke --tick-ms 10 --seed 123 --no-ui`
    - Success criteria: exit code 0, files written under `runs/drybox_nade_bytelink_smoke` (check `metrics.csv`, `pubkeys.txt`).
  - Audio (isolated loop; requires `liquid-dsp`):
  - `PYTHONPATH=../DryBox:. ../DryBox/.venv/bin/python -m drybox.core.runner --scenario ../DryBox/drybox/scenarios/audio_isolated_loop.yaml --left $(pwd)/adapter/drybox_adapter.py:Adapter --right $(pwd)/adapter/drybox_adapter.py:Adapter --out runs/drybox_nade_audio_isolated --tick-ms 10 --seed 123 --no-ui`
  - The adapter will auto-queue a demo text on start; inspect `events.jsonl` and `metrics.csv`.
  - Recommended smoke configuration that yields `text_rx` on isolated loop:
    - `NADE_ADAPTER_CFG='{"modem":"bfsk","modem_cfg":{"samples_per_symbol":80,"bandwidth":0.12,"amp":12000}}'`
    - Look for `text_rx` events and `metric` events with `event=demod` for progress (rx_syms, rx_bytes, preamble_hits, frames).
  - To compare TX vs RX text after a run: `python tools/compare_text_run.py runs/<out>/events.jsonl`

- Option B — Install both projects into one env with `uv`:
  - `uv run pip install -e .`
  - `uv run pip install -e ../DryBox`
  - ByteLink run: `uv run drybox-run --scenario ../DryBox/drybox/scenarios/bytelink_volte_smoke.yaml --left $(pwd)/adapter/drybox_adapter.py:Adapter --right $(pwd)/adapter/drybox_adapter.py:Adapter --out runs/drybox_nade_bytelink_smoke --tick-ms 10 --seed 123 --no-ui`
  - Audio run (requires `liquid-dsp` installed in the same env): `uv run drybox-run --scenario ../DryBox/drybox/scenarios/audio_isolated_loop.yaml --left $(pwd)/adapter/drybox_adapter.py:Adapter --right $(pwd)/adapter/drybox_adapter.py:Adapter --out runs/drybox_nade_audio_isolated --tick-ms 10 --seed 123 --no-ui`

- Notes:
  - DryBox v1 loads adapters by explicit module path spec `path/to/adapter.py:Class`. The adapter class here is `Adapter` in `adapter/drybox_adapter.py`.
  - You can override adapter audio settings without code changes via:
    - `NADE_ADAPTER_CFG='{"modem": "bfsk", "modem_cfg": {"samples_per_symbol": 80, "bandwidth": 0.12}}'`
    - The adapter merges this JSON into its internal `nade_cfg` before starting.
  - Outputs land in `--out` (e.g., `runs/...`):
    - `metrics.csv`: per-tick metrics (goodput, jitter, loss, etc.).
    - `events.jsonl`: adapter-emitted events (text_rx, metric, logs).
    - `pubkeys.txt`: Ed25519 public keys used and adapter specs.

## Coding Style & Naming Conventions
- Use 4-space indentation, maintain explicit type annotations, and keep docstrings focused on protocol intent.
- Modules, functions, and variables stay `snake_case`; classes remain `UpperCamelCase`; constants such as `ABI_VERSION` are all caps.
- Prefer small helper functions (see `_ensure_c_contig_i16`) over deeply nested logic when manipulating numpy buffers.
- Strip trailing whitespace and ensure files end with a newline before opening a PR.

## Testing Guidelines
- Add unit tests under `tests/`, mirroring the source layout (e.g., `tests/modems/test_fsk4.py`).
- Adopt `pytest` and cover handshake paths, queue limits, and adapter error handling; skip gracefully when optional numpy dependencies are absent.
- Run `uv run pytest` locally; attach essential log snippets from `runs/` when manual DryBox sessions validate behavior.

## Commit & Pull Request Guidelines
- Commit subjects follow the existing short, Title-Case imperative style (e.g., `Add Interface`); include rationale or design notes in the body when needed.
- Reference related tickets, DryBox scenario IDs, or spec links so future readers can trace context.
- Pull requests must state the change, test evidence (`pytest` output or manual scenario), and any configuration updates; attach screenshots or logs when audio payloads shift.
- Request review from both protocol and adapter maintainers when touching `nade/audio.py` or `adapter/drybox_adapter.py`.

## Security & Configuration Tips
- Keep secrets out of source; pass keys through DryBox configuration before handing them to `_MockByteLink`.
- Document new events emitted through `emit_event` so downstream telemetry stays in sync.
- Update `pyproject.toml` and regenerate `uv.lock` whenever crypto dependencies shift to avoid desynchronised installs.
