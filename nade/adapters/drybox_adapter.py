"""
DryBox Adapter - Re-export shim.

This module allows imports like:
    from nade.adapters.drybox_adapter import Adapter

by re-exporting from the top-level adapters.drybox_adapter module.
"""
import sys
from pathlib import Path
import importlib.util

# Ensure we import from the project root's adapters package
_project_root = Path(__file__).parent.parent.parent
_adapters_path = _project_root / "adapters"

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import the actual module
_spec = importlib.util.spec_from_file_location(
    "adapters.drybox_adapter",
    _adapters_path / "drybox_adapter.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export
Adapter = _module.Adapter

__all__ = ["Adapter"]
