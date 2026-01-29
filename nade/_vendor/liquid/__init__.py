"""
liquid-dsp minimal Python wrapper for Windows
This module serves as an import target. The actual library is loaded via ctypes.
"""
import os
from pathlib import Path

__version__ = "1.7.0"
__file__ = str(Path(__file__).parent / "libliquid.dll")

# Ensure DLL dependencies can be found
_dll_dir = Path(__file__).parent
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(str(_dll_dir))
