"""
NDA Adapter - Re-export shim.

This module allows imports like:
    from nade.adapters.nda_adapter import NDAAdapter

by re-exporting from the top-level adapters.nda_adapter module.
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
    "adapters.nda_adapter",
    _adapters_path / "nda_adapter.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export
NDAAdapter = _module.NDAAdapter

__all__ = ["NDAAdapter"]
