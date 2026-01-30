"""
Tests for NDA Adapter.

These tests verify the NDAAdapter works correctly for text messaging.

Note: Tests that require the modem (liquid-dsp library) are marked with
@requires_modem and will be skipped if the library is not available.
"""
import pytest
import numpy as np
from dissononce.dh.x25519.x25519 import X25519DH

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

from nade.adapters.nda_adapter import (
    NDAAdapter,
    _bytes_to_keypair,
    _bytes_to_public_key,
)


# Generate test keys using X25519
def generate_test_keypair():
    """Generate X25519 keypair for testing."""
    dh = X25519DH()
    keypair = dh.generate_keypair()
    return keypair.private.data, keypair.public.data


class TestKeyConversion:
    """Test key conversion utilities."""

    def test_bytes_to_keypair(self):
        """Verify bytes to KeyPair conversion."""
        priv, pub = generate_test_keypair()
        keypair = _bytes_to_keypair(priv, pub)
        # Note: PrivateKey may apply X25519 key clamping, so check length
        assert len(keypair.private.data) == 32
        assert keypair.public.data == pub

    def test_bytes_to_public_key(self):
        """Verify bytes to PublicKey conversion."""
        _, pub = generate_test_keypair()
        public_key = _bytes_to_public_key(pub)
        assert public_key.data == pub

    def test_keypair_with_raw_bytes(self):
        """Verify raw 32-byte keys are accepted."""
        raw_priv = bytes(range(32))
        raw_pub = bytes(range(32, 64))
        keypair = _bytes_to_keypair(raw_priv, raw_pub)
        assert len(keypair.private.data) == 32
        assert len(keypair.public.data) == 32


class TestNDAAdapterValidation:
    """Test NDAAdapter key validation (no modem required)."""

    def test_invalid_private_key_length(self):
        """Verify error on invalid private key length."""
        _, pub = generate_test_keypair()
        _, peer_pub = generate_test_keypair()

        with pytest.raises(ValueError, match="Private key must be 32 bytes"):
            NDAAdapter(
                x25519_private_key=b"short",
                x25519_local_public=pub,
                x25519_peer_public=peer_pub,
            )

    def test_invalid_local_public_key_length(self):
        """Verify error on invalid local public key length."""
        priv, _ = generate_test_keypair()
        _, peer_pub = generate_test_keypair()

        with pytest.raises(ValueError, match="Local public key must be 32 bytes"):
            NDAAdapter(
                x25519_private_key=priv,
                x25519_local_public=b"short",
                x25519_peer_public=peer_pub,
            )

    def test_invalid_peer_public_key_length(self):
        """Verify error on invalid peer public key length."""
        priv, pub = generate_test_keypair()

        with pytest.raises(ValueError, match="Peer public key must be 32 bytes"):
            NDAAdapter(
                x25519_private_key=priv,
                x25519_local_public=pub,
                x25519_peer_public=b"short",
            )


@requires_modem
class TestNDAAdapterInit:
    """Test NDAAdapter initialization (requires modem)."""

    def test_valid_initialization(self):
        """Verify NDAAdapter initializes with valid keys."""
        priv, pub = generate_test_keypair()
        _, peer_pub = generate_test_keypair()

        adapter = NDAAdapter(
            x25519_private_key=priv,
            x25519_local_public=pub,
            x25519_peer_public=peer_pub,
            nda_sample_rate=48000,
        )

        assert adapter.nda_sample_rate == 48000
        assert adapter.nade_sample_rate == 8000
        assert adapter.resample_ratio == 6.0
        assert adapter.session_started

    def test_custom_sample_rate(self):
        """Verify custom NDA sample rate."""
        priv, pub = generate_test_keypair()
        _, peer_pub = generate_test_keypair()

        adapter = NDAAdapter(
            x25519_private_key=priv,
            x25519_local_public=pub,
            x25519_peer_public=peer_pub,
            nda_sample_rate=44100,
        )

        assert adapter.nda_sample_rate == 44100
        assert adapter.resample_ratio == 44100 / 8000


@requires_modem
class TestTextMessaging:
    """Test text messaging functionality (requires modem)."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        priv, pub = generate_test_keypair()
        _, peer_pub = generate_test_keypair()
        return NDAAdapter(
            x25519_private_key=priv,
            x25519_local_public=pub,
            x25519_peer_public=peer_pub,
            nda_sample_rate=48000,
        )

    def test_send_text_message_success(self, adapter):
        """Verify message queuing works."""
        result = adapter.send_text_message("Hello")
        assert result["success"] is True
        assert result["error"] == ""
        assert adapter.is_transmitting

    def test_send_text_message_too_long(self, adapter):
        """Verify long messages are rejected."""
        long_message = "x" * 257
        result = adapter.send_text_message(long_message)
        assert result["success"] is False
        assert "too long" in result["error"]

    def test_send_text_message_utf8(self, adapter):
        """Verify UTF-8 encoding works."""
        result = adapter.send_text_message("Hello, 世界!")
        assert result["success"] is True

    def test_get_received_messages_empty(self, adapter):
        """Verify empty message retrieval."""
        messages = adapter.get_received_messages()
        assert messages == []

    def test_get_received_messages_clears(self, adapter):
        """Verify messages are cleared after retrieval."""
        adapter.received_messages = ["test1", "test2"]
        messages = adapter.get_received_messages()
        assert messages == ["test1", "test2"]
        assert adapter.received_messages == []


@requires_modem
class TestAudioProcessing:
    """Test audio processing functionality (requires modem)."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        priv, pub = generate_test_keypair()
        _, peer_pub = generate_test_keypair()
        return NDAAdapter(
            x25519_private_key=priv,
            x25519_local_public=pub,
            x25519_peer_public=peer_pub,
            nda_sample_rate=48000,
        )

    def test_get_tx_audio_no_data(self, adapter):
        """Verify silence when no TX data."""
        audio = adapter.get_tx_audio(10.67)
        assert audio.dtype == np.float32
        # Should be approximately 512 samples at 48kHz for 10.67ms
        expected_samples = int(10.67 * 48000 / 1000)
        assert len(audio) == expected_samples

    def test_get_tx_audio_with_message(self, adapter):
        """Verify FSK generation when message queued."""
        adapter.send_text_message("Test")
        audio = adapter.get_tx_audio(10.67)
        assert audio.dtype == np.float32

    def test_process_rx_audio(self, adapter):
        """Verify RX audio processing doesn't crash."""
        audio = np.random.randn(512).astype(np.float32) * 0.5
        adapter.process_rx_audio(audio, 10.67)

    def test_resampling_ratio(self, adapter):
        """Verify resampling ratio calculation."""
        assert adapter.resample_ratio == 6.0


@requires_modem
class TestDiagnostics:
    """Test diagnostic functionality (requires modem)."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        priv, pub = generate_test_keypair()
        _, peer_pub = generate_test_keypair()
        return NDAAdapter(
            x25519_private_key=priv,
            x25519_local_public=pub,
            x25519_peer_public=peer_pub,
        )

    def test_get_mode_initial(self, adapter):
        """Verify initial mode is rx."""
        assert adapter.get_mode() == "rx"

    def test_get_mode_after_send(self, adapter):
        """Verify mode changes to tx after sending."""
        adapter.send_text_message("Test")
        assert adapter.get_mode() == "tx"

    def test_rx_signal_quality(self, adapter):
        """Verify signal quality returns valid value."""
        quality = adapter.get_rx_signal_quality()
        assert 0.0 <= quality <= 1.0

    def test_get_log_messages(self, adapter):
        """Verify log message retrieval."""
        logs = adapter.get_log_messages()
        assert isinstance(logs, list)

    def test_is_session_established(self, adapter):
        """Verify session establishment check."""
        # Initially not established (handshake not complete)
        established = adapter.is_session_established()
        assert isinstance(established, bool)


@requires_modem
class TestLoopback:
    """Test loopback communication between two adapters (requires modem)."""

    def test_handshake_messages(self):
        """Test that handshake messages are generated."""
        alice_priv, alice_pub = generate_test_keypair()
        bob_priv, bob_pub = generate_test_keypair()

        alice = NDAAdapter(
            x25519_private_key=alice_priv,
            x25519_local_public=alice_pub,
            x25519_peer_public=bob_pub,
            nda_sample_rate=48000,
            is_initiator=True,
        )

        bob = NDAAdapter(
            x25519_private_key=bob_priv,
            x25519_local_public=bob_pub,
            x25519_peer_public=alice_pub,
            nda_sample_rate=48000,
            is_initiator=False,
        )

        assert alice.session_started
        assert bob.session_started

        # Alice should produce TX audio (handshake M1)
        alice_tx = alice.get_tx_audio(20.0)
        assert isinstance(alice_tx, np.ndarray)
