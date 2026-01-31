"""
NDA Integration Tests - Simulate NDA C++ Application Usage.

This module tests that NDAAdapter works exactly as expected from NDA
(the C++ Qt6 application) via pybind11.

The tests simulate:
1. The import path NDA uses
2. The NDAAdapter constructor signature NDA expects
3. The methods NDA calls (get_tx_audio, process_rx_audio, etc.)
4. The handshake phase tracking for UI display
"""
import pytest
import numpy as np

# Check if modem (liquid-dsp) is available
_MODEM_AVAILABLE = False
try:
    from nade.modems.cpfsk import _LiquidFSKLibrary
    _LiquidFSKLibrary.instance()
    _MODEM_AVAILABLE = True
except (OSError, ImportError, Exception):
    _MODEM_AVAILABLE = False

requires_modem = pytest.mark.skipif(
    not _MODEM_AVAILABLE,
    reason="liquid-dsp library not available on this platform"
)


class TestNDAImportPath:
    """Test the exact import path NDA C++ uses."""

    def test_nda_import_path(self):
        """Test import via nade.adapters.nda_adapter (NDA expected path)."""
        # This is the exact import that NDA C++ does via pybind11
        from nade.adapters.nda_adapter import NDAAdapter
        assert NDAAdapter is not None

    def test_alternative_import_path(self):
        """Test import via nade.adapters (shorthand)."""
        from nade.adapters import NDAAdapter
        assert NDAAdapter is not None


@requires_modem
class TestNDAAdapterInstantiation:
    """Test NDAAdapter can be created with expected parameters."""

    def test_nda_adapter_with_all_parameters(self):
        """Test instantiation with all parameters NDA might pass."""
        from nade.adapters.nda_adapter import NDAAdapter

        # Generate proper X25519 keys
        from dissononce.dh.x25519.x25519 import X25519DH
        dh = X25519DH()
        alice = dh.generate_keypair()
        bob = dh.generate_keypair()

        # Create adapter with all parameters NDA might pass
        adapter = NDAAdapter(
            x25519_private_key=alice.private.data,
            x25519_local_public=alice.public.data,
            x25519_peer_public=bob.public.data,
            nda_sample_rate=48000,
            modem_mode="4fsk",
            is_initiator=True,
            enable_discovery=True,
        )

        assert adapter is not None
        assert adapter.nda_sample_rate == 48000
        assert adapter.nade_sample_rate == 8000

    def test_nda_adapter_with_minimal_parameters(self):
        """Test instantiation with minimal (default) parameters."""
        from nade.adapters.nda_adapter import NDAAdapter
        from dissononce.dh.x25519.x25519 import X25519DH

        dh = X25519DH()
        alice = dh.generate_keypair()
        bob = dh.generate_keypair()

        # Minimal parameters (defaults for others)
        adapter = NDAAdapter(
            x25519_private_key=alice.private.data,
            x25519_local_public=alice.public.data,
            x25519_peer_public=bob.public.data,
        )

        assert adapter is not None

    def test_nda_adapter_with_bfsk_mode(self):
        """Test instantiation with BFSK modem mode."""
        from nade.adapters.nda_adapter import NDAAdapter
        from dissononce.dh.x25519.x25519 import X25519DH

        dh = X25519DH()
        alice = dh.generate_keypair()
        bob = dh.generate_keypair()

        adapter = NDAAdapter(
            x25519_private_key=alice.private.data,
            x25519_local_public=alice.public.data,
            x25519_peer_public=bob.public.data,
            modem_mode="bfsk",
        )

        assert adapter is not None


@requires_modem
class TestNDAAdapterMethods:
    """Test that NDAAdapter has all required methods for NDA integration."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        from nade.adapters.nda_adapter import NDAAdapter
        from dissononce.dh.x25519.x25519 import X25519DH

        dh = X25519DH()
        alice = dh.generate_keypair()
        bob = dh.generate_keypair()

        return NDAAdapter(
            x25519_private_key=alice.private.data,
            x25519_local_public=alice.public.data,
            x25519_peer_public=bob.public.data,
            nda_sample_rate=48000,
            modem_mode="4fsk",
            is_initiator=True,
            enable_discovery=True,
        )

    def test_has_get_handshake_phase(self, adapter):
        """Verify get_handshake_phase method exists and returns int."""
        assert hasattr(adapter, 'get_handshake_phase')
        phase = adapter.get_handshake_phase()
        assert isinstance(phase, int)
        assert 0 <= phase <= 3  # 0=Idle, 1=Discovering, 2=Handshaking, 3=Established

    def test_has_get_tx_audio(self, adapter):
        """Verify get_tx_audio method exists and returns numpy array."""
        assert hasattr(adapter, 'get_tx_audio')
        tx_audio = adapter.get_tx_audio(duration_ms=10.0)
        assert isinstance(tx_audio, np.ndarray)
        assert tx_audio.dtype == np.float32

    def test_has_process_rx_audio(self, adapter):
        """Verify process_rx_audio method exists and accepts array."""
        assert hasattr(adapter, 'process_rx_audio')
        rx_audio = np.zeros(512, dtype=np.float32)
        adapter.process_rx_audio(rx_audio, duration_ms=10.0)  # Should not raise

    def test_has_send_text_message(self, adapter):
        """Verify send_text_message method exists and returns dict."""
        assert hasattr(adapter, 'send_text_message')
        result = adapter.send_text_message("Test message")
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'error' in result

    def test_has_get_received_messages(self, adapter):
        """Verify get_received_messages method exists and returns list."""
        assert hasattr(adapter, 'get_received_messages')
        messages = adapter.get_received_messages()
        assert isinstance(messages, list)

    def test_has_is_transmitting_active(self, adapter):
        """Verify is_transmitting_active method exists and returns bool."""
        assert hasattr(adapter, 'is_transmitting_active')
        is_tx = adapter.is_transmitting_active()
        assert isinstance(is_tx, bool)

    def test_has_restart_discovery(self, adapter):
        """Verify restart_discovery method exists."""
        assert hasattr(adapter, 'restart_discovery')
        adapter.restart_discovery()  # Should not raise

    def test_has_is_session_established(self, adapter):
        """Verify is_session_established method exists and returns bool."""
        assert hasattr(adapter, 'is_session_established')
        established = adapter.is_session_established()
        assert isinstance(established, bool)

    def test_has_get_mode(self, adapter):
        """Verify get_mode method exists and returns string."""
        assert hasattr(adapter, 'get_mode')
        mode = adapter.get_mode()
        assert mode in ('tx', 'rx', 'idle')

    def test_has_get_log_messages(self, adapter):
        """Verify get_log_messages method exists and returns list."""
        assert hasattr(adapter, 'get_log_messages')
        logs = adapter.get_log_messages()
        assert isinstance(logs, list)


@requires_modem
class TestNDAAdapterAudioFlow:
    """Test basic audio generation and processing flow."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        from nade.adapters.nda_adapter import NDAAdapter
        from dissononce.dh.x25519.x25519 import X25519DH

        dh = X25519DH()
        alice = dh.generate_keypair()
        bob = dh.generate_keypair()

        return NDAAdapter(
            x25519_private_key=alice.private.data,
            x25519_local_public=alice.public.data,
            x25519_peer_public=bob.public.data,
            nda_sample_rate=48000,
            enable_discovery=True,
        )

    def test_tx_audio_generation(self, adapter):
        """Test TX audio generation returns correct format."""
        tx_audio = adapter.get_tx_audio(duration_ms=10.0)

        assert isinstance(tx_audio, np.ndarray)
        assert tx_audio.dtype == np.float32
        assert len(tx_audio) > 0

        # Expected samples at 48kHz for 10ms
        expected_samples = int(10.0 * 48000 / 1000)
        assert len(tx_audio) == expected_samples

    def test_rx_audio_processing(self, adapter):
        """Test RX audio processing accepts correct format."""
        # Simulate receiving audio at 48kHz
        rx_audio = np.random.randn(512).astype(np.float32) * 0.1

        # Should not raise
        adapter.process_rx_audio(rx_audio, duration_ms=10.0)

    def test_tx_rx_loopback(self, adapter):
        """Test basic TX to RX loopback without crash."""
        # Generate TX audio
        tx_audio = adapter.get_tx_audio(duration_ms=20.0)

        # Feed it back as RX (simulates radio channel)
        adapter.process_rx_audio(tx_audio, duration_ms=20.0)

        # Check for received messages (may be empty if handshake incomplete)
        messages = adapter.get_received_messages()
        assert isinstance(messages, list)


@requires_modem
class TestNDAHandshakePhases:
    """Test handshake phase tracking for UI display."""

    def test_initial_phase_with_discovery(self):
        """Test initial phase is DISCOVERING when discovery enabled."""
        from nade.adapters.nda_adapter import NDAAdapter
        from dissononce.dh.x25519.x25519 import X25519DH

        dh = X25519DH()
        alice = dh.generate_keypair()
        bob = dh.generate_keypair()

        adapter = NDAAdapter(
            x25519_private_key=alice.private.data,
            x25519_local_public=alice.public.data,
            x25519_peer_public=bob.public.data,
            enable_discovery=True,
        )

        phase = adapter.get_handshake_phase()
        # Should be 1 (Discovering) when discovery mode is enabled
        assert phase == 1

    def test_initial_phase_without_discovery(self):
        """Test initial phase is HANDSHAKING when discovery disabled."""
        from nade.adapters.nda_adapter import NDAAdapter
        from dissononce.dh.x25519.x25519 import X25519DH

        dh = X25519DH()
        alice = dh.generate_keypair()
        bob = dh.generate_keypair()

        adapter = NDAAdapter(
            x25519_private_key=alice.private.data,
            x25519_local_public=alice.public.data,
            x25519_peer_public=bob.public.data,
            enable_discovery=False,
            is_initiator=True,
        )

        phase = adapter.get_handshake_phase()
        # Should be 2 (Handshaking) when directly starting session
        assert phase == 2


@requires_modem
class TestNDADiscovery:
    """Test discovery functionality."""

    def test_restart_discovery(self):
        """Test restarting discovery mode."""
        from nade.adapters.nda_adapter import NDAAdapter
        from dissononce.dh.x25519.x25519 import X25519DH

        dh = X25519DH()
        alice = dh.generate_keypair()
        bob = dh.generate_keypair()

        adapter = NDAAdapter(
            x25519_private_key=alice.private.data,
            x25519_local_public=alice.public.data,
            x25519_peer_public=bob.public.data,
            enable_discovery=False,  # Start without discovery
        )

        # Restart discovery
        adapter.restart_discovery()

        # Should now be in discovery phase
        phase = adapter.get_handshake_phase()
        assert phase == 1  # Discovering
