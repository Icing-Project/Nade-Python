"""
Nade Adapters - Re-exports from top-level adapters package.

This subpackage allows imports like:
    from nade.adapters.nda_adapter import NDAAdapter

which is the expected import path for NDA C++ integration via pybind11.
"""
import sys
from pathlib import Path

# Ensure we import from the project root's adapters package, not tests/adapters
_project_root = Path(__file__).parent.parent.parent
_adapters_path = _project_root / "adapters"

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Now import from the actual adapters package
import importlib.util

_nda_spec = importlib.util.spec_from_file_location(
    "adapters.nda_adapter",
    _adapters_path / "nda_adapter.py"
)
_nda_module = importlib.util.module_from_spec(_nda_spec)
_nda_spec.loader.exec_module(_nda_module)
NDAAdapter = _nda_module.NDAAdapter

_drybox_spec = importlib.util.spec_from_file_location(
    "adapters.drybox_adapter",
    _adapters_path / "drybox_adapter.py"
)
_drybox_module = importlib.util.module_from_spec(_drybox_spec)
_drybox_spec.loader.exec_module(_drybox_module)
Adapter = _drybox_module.Adapter

__all__ = [
    "NDAAdapter",
    "Adapter",
]
