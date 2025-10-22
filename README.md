# Nade-Python
Nade's Python test-oriented implementation, design to fit the DryBox.

For contribution expectations and workflows, read the [Repository Guidelines](AGENTS.md).

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
uv drybox-run --scenario [Your Scenario] --left adapter.drybox_adapter:Adapter --right adapter.drybox_adapter:Adapter --out runs/[Your run name]
```

Or, preferred, run the GUI:
```bash
uv run -m drybox.gui.app
```

Then select the Nade package as left and right adapters.
