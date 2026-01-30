"""
Pytest configuration for Nade-Python tests.

This file provides fixtures and utilities for testing.
"""
import pytest
import sys
from pathlib import Path

# Ensure nade package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_keypair():
    """Generate X25519 keypair for testing."""
    from dissononce.dh.x25519.x25519 import X25519DH
    dh = X25519DH()
    keypair = dh.generate_keypair()
    return keypair.private.data, keypair.public.data


@pytest.fixture
def alice_bob_keypairs():
    """Generate two keypairs for loopback testing."""
    from dissononce.dh.x25519.x25519 import X25519DH
    dh = X25519DH()

    alice = dh.generate_keypair()
    bob = dh.generate_keypair()

    return {
        "alice": {
            "private": alice.private.data,
            "public": alice.public.data,
        },
        "bob": {
            "private": bob.private.data,
            "public": bob.public.data,
        },
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_modem: mark test as requiring liquid-dsp modem"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require modem if not available."""
    # Check if modem is available
    modem_available = False
    try:
        from nade.modems.cpfsk import _LiquidFSKLibrary
        _LiquidFSKLibrary.instance()
        modem_available = True
    except Exception:
        pass

    if not modem_available:
        skip_modem = pytest.mark.skip(reason="liquid-dsp library not available")
        for item in items:
            if "requires_modem" in item.keywords:
                item.add_marker(skip_modem)
