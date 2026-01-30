# Windows Support for liquid-dsp

## Overview

Nade-Python bundles pre-compiled liquid-dsp libraries for Windows to enable zero-setup installation. The liquid-dsp C library and all dependencies are included in `nade/_vendor/liquid/`.

## What's Bundled

Located in `nade/_vendor/liquid/`:
- **libliquid.dll** (4.0 MB) - The liquid-dsp C library
- **libfftw3f-3.dll** (4.4 MB) - FFTW single-precision library
- **libfftw3-3.dll** (4.1 MB) - FFTW double-precision library
- **MinGW runtime DLLs** - libgcc, libstdc++, libwinpthread, libgomp
- **__init__.py** - Python wrapper module for import

## How It Works

### Import Mechanism

When you import `from nade.modems.cpfsk import LiquidFSKModem`:

1. The `nade/__init__.py` adds `nade/_vendor/` to `sys.path`
2. The modem code tries `import liquid`
3. Python finds the `nade/_vendor/liquid/` package
4. The `__init__.py` sets up DLL discovery via `os.add_dll_directory()`
5. The modem loads `libliquid.dll` via ctypes and accesses C functions directly

### No Python Extension Needed

Unlike typical Python packages, liquid-dsp for Windows uses **pure ctypes** - no `.pyd` files. The modem code directly loads the C library DLL and calls functions using ctypes FFI.

## Installation (Users)

**Zero setup required!** Just install the package:

```bash
pip install git+https://github.com/your-org/Nade-Python.git
# or
uv add path/to/Nade-Python
```

All DLLs are included and automatically discovered on Windows.

## Rebuilding (Developers)

If you need to rebuild the DLLs (e.g., to update liquid-dsp version or add optimizations):

### Prerequisites

1. Install MSYS2 from https://www.msys2.org/
2. Install Windows Python (the version you want to support)

### Build Steps

1. Open **MSYS2 MinGW64** terminal (not MSYS2 MSYS or UCRT64)

2. Navigate to the project directory:
   ```bash
   cd /c/Users/YourName/path/to/Nade-Python
   ```

3. Run the build script:
   ```bash
   ./scripts/build_liquid_windows.sh
   ```

4. The script will:
   - Install MSYS2 dependencies (gcc, autoconf, fftw)
   - Clone liquid-dsp from GitHub
   - Run configure with workarounds for MinGW issues
   - Build the static library
   - Manually link the DLL (makefile linking fails due to `-lc` flag)
   - Copy DLL and dependencies to `nade/_vendor/liquid/`
   - Create the Python `__init__.py` wrapper

5. Verify the build:
   ```bash
   ls -lh nade/_vendor/liquid/
   # Should show libliquid.dll and dependencies
   ```

### Testing

From **Windows PowerShell** (not WSL):

```powershell
cd C:\Users\YourName\path\to\Nade-Python
python -c "import sys; sys.path.insert(0, 'nade/_vendor'); import liquid; print(liquid.__version__)"
```

Expected output: `1.7.0` (or current version)

### Build Challenges & Solutions

The build process overcomes several MinGW-specific issues:

1. **Configure tests fail with `-lc`**: Use autoconf cache variables to skip:
   ```bash
   ac_cv_lib_c_main=yes ac_cv_lib_m_main=yes ac_cv_func_malloc=yes ...
   ```

2. **Makefile links with `-lc` and `-lm`**: These cause linker errors on MinGW. Solution: Let the static library build succeed, then manually link the DLL:
   ```bash
   gcc -shared -o libliquid.dll -Wl,--whole-archive libliquid.a -Wl,--no-whole-archive -lfftw3f
   ```

3. **MinGW libraries are implicit**: `-lc` and `-lm` don't exist as separate libraries on MinGW - they're part of the C runtime. The linker automatically includes them.

## Architecture Notes

### Why ctypes instead of Cython/pybind11?

The original liquid-dsp doesn't provide Python bindings. Options were:

1. **Create Python bindings** (Cython/SWIG/pybind11) - Complex, maintenance burden
2. **Use ctypes** (current approach) - Simple, no compilation needed for Python side
3. **Vendor a Python wrapper** - Adds external dependency

We chose **ctypes** for simplicity. The `_LiquidFSKLibrary` class in `nade/modems/cpfsk.py` wraps the C API.

### Cross-Platform Strategy

| Platform | liquid-dsp Source | How Installed |
|----------|-------------------|---------------|
| Linux    | PyPI wheel (if available) or compile | pip install liquid-dsp |
| macOS    | PyPI wheel (if available) or compile | pip install liquid-dsp |
| Windows  | Bundled DLLs | Included in nade/_vendor/ |
| WSL2     | Linux PyPI wheel | pip install liquid-dsp |

The code tries system installation first, falls back to bundled version on import error.

## Troubleshooting

### "invalid ELF header" error

**Cause**: You're running from WSL (Linux Python) trying to load a Windows DLL.

**Solution**: Run from Windows PowerShell/CMD, or use:
```bash
cmd.exe /c "python your_script.py"
```

### DLL load failed

**Symptoms**: `OSError: [WinError 126] The specified module could not be found`

**Causes**:
1. Missing dependency DLLs
2. Wrong Python version (32-bit vs 64-bit)
3. Missing Visual C++ Redistributable

**Solutions**:
- Check all DLLs present: `ls nade/_vendor/liquid/`
- Use 64-bit Python (DLLs are x86_64)
- Install VC++ Redist: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Function not found

**Symptoms**: `AttributeError: function 'fskmod_create' not found`

**Cause**: DLL is outdated or corrupted

**Solution**: Rebuild DLL with latest liquid-dsp

## Future Improvements

- **GitHub Actions CI**: Automate Windows DLL builds
- **Multiple Python versions**: Build for Python 3.10, 3.11, 3.12, 3.13
- **Publish to PyPI**: Provide Windows wheels so users don't need bundled DLLs
- **ARM64 support**: Build for ARM64 Windows when ARM64 MSYS2 is stable

## References

- liquid-dsp: https://github.com/jgaeddert/liquid-dsp
- MSYS2: https://www.msys2.org/
- MinGW-w64: https://www.mingw-w64.org/
- FFTW: http://www.fftw.org/
