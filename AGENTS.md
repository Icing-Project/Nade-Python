# Repository Guidelines

## Project Structure & Module Organization
- `nade/` hosts the protocol stack: `audio.py` orchestrates modem selection, `crypto/` wraps Noise building blocks, `fec/` and `modems/` handle signal processing, and `preprocessing/` stores reusable helpers.
- `adapter/` exposes the DryBox integration; `drybox_adapter.py` binds byte and audio modes while `modes/` provides runnable presets.
- `runs/` captures local experiments; prune noisy artifacts before committing.

## Build, Test, and Development Commands
- `uv sync` installs the locked Python 3.11 dependency set.
- `uv run pip install -e .` publishes the package locally so DryBox discovers `nade-python`.
- `uv run python -m adapter.drybox_adapter` performs a quick import and smoke check of the adapter entry points.

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
