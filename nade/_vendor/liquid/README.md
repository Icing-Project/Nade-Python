# Bundled liquid-dsp for Windows

This directory contains pre-compiled liquid-dsp libraries for Windows compatibility.

## Contents

After building with `scripts/build_liquid_windows.sh`, this directory should contain:

- `liquid.cp3XX-win_amd64.pyd` - Python extension module (XX = Python version, e.g., cp311 for Python 3.11)
- `libliquid.dll` or `liquid.dll` - liquid-dsp shared library
- Additional dependency DLLs (FFTW, etc.)

## Building

To build the Windows DLLs:

1. Install MSYS2 from https://www.msys2.org/
2. Open MSYS2 MinGW64 terminal
3. Navigate to Nade-Python root directory
4. Run: `./scripts/build_liquid_windows.sh`
5. Files will be copied here automatically

## Usage

These bundled libraries are automatically used on Windows when liquid-dsp is not available from PyPI.

The import mechanism in `nade/__init__.py` adds this directory to `sys.path`, allowing the bundled `liquid` module to be imported.

For full documentation, see `docs/WINDOWS_SUPPORT.md`.
