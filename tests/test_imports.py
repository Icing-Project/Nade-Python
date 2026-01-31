"""
Comprehensive import test suite for Nade-Python.

This module validates that all public APIs are properly exported
and importable, which is critical for pybind11 integration with NDA.
"""
import pytest


class TestTopLevelImports:
    """Test top-level package imports."""

    def test_import_nade(self):
        """Test basic nade package import."""
        import nade
        assert nade.__version__ is not None

    def test_nade_exports_core_classes(self):
        """Test that nade exports core classes."""
        from nade import (
            AudioStack,
            NoiseXKWrapper,
            NadeProtocol,
            NadeState,
            Phase,
            NadeEngine,
            ITransport,
            AudioTransport,
        )
        assert AudioStack is not None
        assert NoiseXKWrapper is not None
        assert NadeProtocol is not None
        assert NadeState is not None
        assert Phase is not None
        assert NadeEngine is not None
        assert ITransport is not None
        assert AudioTransport is not None

    def test_nade_exports_modems(self):
        """Test that nade exports modem classes."""
        from nade import (
            LiquidBFSKModem,
            LiquidFourFSKModem,
            FourFSKModem,  # Backwards-compatible alias
        )
        assert LiquidBFSKModem is not None
        assert LiquidFourFSKModem is not None
        assert FourFSKModem is LiquidFourFSKModem

    def test_nade_exports_adapter(self):
        """Test that nade exports NDAAdapter."""
        from nade import NDAAdapter
        assert NDAAdapter is not None


class TestProtocolImports:
    """Test protocol submodule imports."""

    def test_import_protocol_module(self):
        """Test basic protocol module import."""
        from nade import protocol
        assert protocol is not None

    def test_protocol_state_exports(self):
        """Test protocol state classes."""
        from nade.protocol import NadeState, Phase
        assert NadeState is not None
        assert Phase is not None
        # Check Phase enum values exist
        assert hasattr(Phase, 'IDLE')
        assert hasattr(Phase, 'ESTABLISHED')
        assert hasattr(Phase, 'PING_DISCOVERY')

    def test_protocol_event_exports(self):
        """Test protocol event classes."""
        from nade.protocol import (
            Event,
            StartSession,
            StopSession,
            TransportRxReady,
            TransportTxCapacity,
            AppSendData,
            TimerExpired,
            LinkQualityUpdate,
        )
        assert Event is not None
        assert StartSession is not None
        assert StopSession is not None
        assert TransportRxReady is not None
        assert TransportTxCapacity is not None
        assert AppSendData is not None
        assert TimerExpired is not None
        assert LinkQualityUpdate is not None

    def test_protocol_discovery_event_exports(self):
        """Test protocol discovery event classes (used by NDA)."""
        from nade.protocol import (
            StartDiscovery,
            PingReceived,
            PongReceived,
            PingTimerExpired,
            ForceHandshake,
        )
        assert StartDiscovery is not None
        assert PingReceived is not None
        assert PongReceived is not None
        assert PingTimerExpired is not None
        assert ForceHandshake is not None

    def test_protocol_action_exports(self):
        """Test protocol action classes."""
        from nade.protocol import (
            Action,
            CryptoStartHandshake,
            CryptoProcessMessage,
            CryptoEncrypt,
            CryptoDecrypt,
            TransportSend,
            TransportFlushHandshake,
            TimerStart,
            TimerCancel,
            AppDeliver,
            AppNotify,
            Log,
        )
        assert Action is not None
        assert CryptoStartHandshake is not None
        assert CryptoProcessMessage is not None
        assert CryptoEncrypt is not None
        assert CryptoDecrypt is not None
        assert TransportSend is not None
        assert TransportFlushHandshake is not None
        assert TimerStart is not None
        assert TimerCancel is not None
        assert AppDeliver is not None
        assert AppNotify is not None
        assert Log is not None

    def test_protocol_discovery_action_exports(self):
        """Test protocol discovery action classes (critical for NDA)."""
        from nade.protocol import SendPing, SendPong
        assert SendPing is not None
        assert SendPong is not None

    def test_protocol_machine_export(self):
        """Test NadeProtocol state machine export."""
        from nade.protocol import NadeProtocol
        assert NadeProtocol is not None
        assert hasattr(NadeProtocol, 'step')


class TestAdapterImports:
    """Test adapter imports (critical for NDA C++ integration)."""

    def test_import_from_nade_adapters(self):
        """Test import via nade.adapters (NDA expected path)."""
        from nade.adapters import NDAAdapter
        assert NDAAdapter is not None

    def test_import_nda_adapter_module(self):
        """Test import via nade.adapters.nda_adapter (NDA expected path)."""
        from nade.adapters.nda_adapter import NDAAdapter
        assert NDAAdapter is not None

    def test_import_drybox_adapter(self):
        """Test import via nade.adapters.drybox_adapter."""
        from nade.adapters.drybox_adapter import Adapter
        assert Adapter is not None

    @pytest.mark.skip(reason="tests/adapters shadows top-level adapters in pytest")
    def test_import_from_top_level_adapters(self):
        """Test import from top-level adapters package."""
        from adapters import NDAAdapter, Adapter
        assert NDAAdapter is not None
        assert Adapter is not None

    @pytest.mark.skip(reason="tests/adapters shadows top-level adapters in pytest")
    def test_import_from_top_level_nda_adapter(self):
        """Test import from top-level adapters.nda_adapter."""
        from adapters.nda_adapter import NDAAdapter
        assert NDAAdapter is not None


class TestEngineImports:
    """Test engine imports."""

    def test_import_engine(self):
        """Test NadeEngine import."""
        from nade.engine import NadeEngine, TimerRequest
        assert NadeEngine is not None
        assert TimerRequest is not None

    def test_engine_uses_protocol_classes(self):
        """Test that engine can import all protocol classes it needs."""
        # This reproduces the exact imports from engine.py
        from nade.protocol import (
            NadeProtocol,
            NadeState,
            Phase,
            Event,
            Action,
            CryptoStartHandshake,
            CryptoProcessMessage,
            CryptoEncrypt,
            CryptoDecrypt,
            TransportSend,
            TransportFlushHandshake,
            TimerStart,
            TimerCancel,
            AppDeliver,
            AppNotify,
            Log,
            SendPing,
            SendPong,
        )
        # All imports succeeded
        assert SendPing is not None
        assert SendPong is not None


class TestTransportImports:
    """Test transport imports."""

    def test_import_transport_interface(self):
        """Test ITransport interface import."""
        from nade.transport import ITransport
        assert ITransport is not None

    def test_import_audio_transport(self):
        """Test AudioTransport import."""
        from nade.transport import AudioTransport
        assert AudioTransport is not None


class TestCryptoImports:
    """Test crypto imports."""

    def test_import_noise_wrapper(self):
        """Test NoiseXKWrapper import."""
        from nade.crypto.noise_wrapper import NoiseXKWrapper
        assert NoiseXKWrapper is not None


class TestModemImports:
    """Test modem imports."""

    def test_import_modems_module(self):
        """Test modems module import."""
        from nade import modems
        assert modems is not None

    def test_import_liquid_modems(self):
        """Test liquid modem class imports."""
        from nade.modems import LiquidFourFSKModem, LiquidBFSKModem
        assert LiquidFourFSKModem is not None
        assert LiquidBFSKModem is not None
