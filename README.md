# Nade-Python
Nade's Python test-oriented implementation, design to fit the DryBox.

For contribution expectations and workflows, read the [Repository Guidelines](AGENTS.md).

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | ✅ Supported | Install with `uv sync` |
| macOS | ✅ Supported | Install with `uv sync` |
| Windows | ✅ Supported | Bundled liquid-dsp included |
| WSL2 | ✅ Supported | Uses Linux wheels |

### Windows Notes

Audio features (BFSK/4FSK modems) require liquid-dsp, which is bundled with this package for Windows users. No additional setup required - just:

```bash
cd DryBox
uv add --dev ../Nade-Python
```

The bundled liquid-dsp DLLs are located in `nade/_vendor/liquid/` and are automatically used on Windows.

**Building liquid-dsp on Windows (optional)**:

If you need to rebuild the Windows DLLs (e.g., for a different Python version):

1. Install MSYS2 from https://www.msys2.org/
2. Open MSYS2 MinGW64 terminal
3. Navigate to Nade-Python directory
4. Run: `./scripts/build_liquid_windows.sh`
5. Verify files in `nade/_vendor/liquid/`

## HOW TO USE / TEST

**We utilize [UV](https://docs.astral.sh/uv/getting-started/installation/), this is not absolutely required but best to have.**

All these steps can also be done with manual venv use.

### DryBox use

**First, make sure you have the DryBox in a different directory.**
```bash
cd .. && git clone git@github.com:Icing-Project/DryBox.git
```

Install the local requirements
```bash
uv sync
```

Not required, but **recommended**:
Install Nade as a package
```bash
uv pip install -e .
```

Install the DryBox as a package
```bash
uv pip install -e ../DryBox
```

Run in the DryBox CLI with the Nade adapter:
```bash
uv drybox-run --scenario [Your Scenario] --left adapters.drybox_adapter:Adapter --right adapters.drybox_adapter:Adapter --out runs/[Your run name]
```

Or, preferred, run the GUI:
```bash
uv run -m drybox.gui.app
```

Then select the Nade package as left and right adapters.
