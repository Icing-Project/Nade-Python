# DryBox v1-compatible adapter for Nade-Python
# - Zero-arg __init__ (runner instantiates with cls())
# - init(cfg) to receive side/mode/crypto
# - start(ctx) to receive AdapterCtx (now_ms(), emit_event(), rng, config)
# - Byte mode: minimal handshake + SDU pass-through mock
# - Audio mode: silence source/sink mock (160 @ 8kHz)

from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional, Tuple
import os
import json
from nade.audio import AudioStack

try:
    from nade.streamer import AudioStreamer
except ImportError:
    AudioStreamer = None
from nade.crypto.noise_wrapper import NoiseXKWrapper
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.x25519.private import PrivateKey

try:
    import numpy as np
except Exception:
    np = None  # Audio mock works only if numpy is present

try:
    import pyaudio
except ImportError:
    pyaudio = None  # Real-time audio playback disabled

ABI_VERSION = "dbx-v1"
HANDSHAKE_TAG = 0x20
SDU_MAX_BYTES = 1024
INITIATOR_SIDE = "L"

# Set to False to enable full Noise XK encryption
ENCRYPTION_BYPASS = False


def _merged_nade_cfg(base: dict | None) -> dict:                                    # TODO: remove when switching to DryBox controls OR effective statemachine
    """Merge base nade cfg with optional JSON in NADE_ADAPTER_CFG.
    Keeps behavior identical to previous inline merge (shallow update, best-effort).
    """
    cfg: dict = (base or {}).copy()
    try:
        env_cfg = os.environ.get("NADE_ADAPTER_CFG")
        if env_cfg:
            parsed = json.loads(env_cfg)
            if isinstance(parsed, dict):
                cfg.update(parsed)
    except Exception:
        # Preserve previous silent-fail behavior
        pass
    return cfg


class Adapter:
    """
    DryBox adapter (v1) for Nade.
    __init__(): zero-arg constructor
    init(cfg): receive side/mode/crypto (and nade cfg)
    start(ctx): receive AdapterCtx (now_ms(), emit_event(), rng, config)
    Byte mode: minimal handshake + SDU pass-through mock (no crypto)
    Audio mode: uses AudioStack (8k/160) and emits text/log/metric events
    """


    def __init__(self):
        self.cfg: dict = {}
        self.ctx: Optional[Any] = None
        self.mode: str = "byte"
        self.side: str = "L"
        self.crypto: dict = {}
        # ByteLink state (mock)
        self._byte_started: bool = False
        self._byte_done: bool = False
        self._byte_t_ms: int = 0
        self._byte_txq: Deque[Tuple[bytes, int]] = deque()
        # Audio state
        self._audio_stack: Optional[AudioStack] = None
        self._audio_streamer: Optional[AudioStreamer] = None
        self._noise: Optional[NoiseXKWrapper] = None
        self._handshake_started = False
        self._pending_handshake = True
        self._both_handshake_complete = False   # TODO: use noise_wrapper state variable

        # TX queue for encrypted SDUs (Noise inside Audio)
        self._audio_tx_sdu_q: Deque[bytes] = deque()
        
        # Echo detection: track sent message hashes to filter out our own echoes
        self._sent_msg_hashes: set = set()


    # ---- capabilities (runner les lit avant de démarrer l’I/O) ----
    def nade_capabilities(self) -> dict:
        return {
            "abi_version": ABI_VERSION,
            "bytelink": True,
            "audioblock": True,
            "sdu_max_bytes": SDU_MAX_BYTES,
            "audioparams": {"sr": 8000, "block": 160, "channels": 2},
        }

    # ---- lifecycle ----
    def init(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.side = self.cfg.get("side", "L")
        self.mode = self.cfg.get("mode", "audio")
        self.crypto = self.cfg.get("crypto", {}) or {}
        # Base config plus optional env override: NADE_ADAPTER_CFG='{"modem":"bfsk","modem_cfg":{...}}'
        self.nade_cfg = _merged_nade_cfg(self.cfg.get("nade", {}) or {})
        self.use_hardware = self.nade_cfg.get("use_hardware", False)

    def start(self, ctx):
        self.ctx = ctx

        if self.mode == "audio":

            self._audio_logger("info", f"[Adapter] start() side={self.side} mode=audio")

            modem_cfg = (self.nade_cfg or {}).get("modem_cfg") or {}
            # Use 4FSK by default - BFSK is too slow (5+ seconds per frame)
            modem_name = (self.nade_cfg or {}).get("modem", "4fsk")

            self._audio_logger("info", f"[Adapter] Creating AudioStack modem={modem_name} cfg={modem_cfg}")

            # --- Real Audio Init ---
            self._use_real_audio = False
            if AudioStreamer:
                try:
                    # For WSL2 debugging, we enable test tone by default if mic is silent
                    self._real_audio = AudioStreamer(rate=8000, chunk_size=160, channels=1, debug=True,
                                                     use_test_tone=True, test_tone_freq=440.0)
                    self._real_audio.start()
                    self._use_real_audio = True
                    self._audio_logger("info", f"[Adapter] AudioStreamer active (Test Tone enabled)")
                    self._next_wall_time = None
                except Exception as e:
                    self._audio_logger("warn", f"[Adapter] AudioStreamer failed ({e})")

            # --- Protocol Stack Init (Handshake, Modem, Encryption) ---
            from nade.audio import AudioStack
            print(f"[DEBUG] Creating AudioStack with modem={modem_name} cfg={modem_cfg}")
            self._audio_stack = AudioStack(
                modem=modem_name,
                modem_cfg=modem_cfg,
                logger=self._audio_logger
            )
            
            # --- Cryptography Setup (Noise XK) ---
            # Extract keys from self.crypto (provided by DryBox runner)
            priv_raw = self.crypto.get("priv")
            pub_raw = self.crypto.get("pub")
            peer_pub_raw = self.crypto.get("peer_pub")
            
            local_kp = None
            peer_pub_obj = None
            
            if priv_raw and pub_raw:
                local_kp = KeyPair(PublicKey(bytes(pub_raw)), PrivateKey(bytes(priv_raw)))
            if peer_pub_raw:
                peer_pub_obj = PublicKey(bytes(peer_pub_raw))
            
            if local_kp and peer_pub_obj:
                self._noise = NoiseXKWrapper(local_kp, peer_pub_obj)
                self._audio_logger("info", "[NoiseXK] Protocol wrapper initialized from DryBox keys")
                print(f"[DEBUG] NoiseXK initialized successfully with DryBox keys: {self.crypto}")
            else:
                self._audio_logger("warn", "[NoiseXK] Missing keys in 'crypto' config, encryption disabled")
                print("[DEBUG] NoiseXK MISSING KEYS - Encryption disabled")
                self._noise = None

                self._noise = None
            
            # --- RX WAV recording (debugging/verification) ---
            self._rx_wav_file = None
            try:
                import wave
                # Save to "rx_output_L.wav" or "rx_output_R.wav" in current dir
                fname = f"rx_output_{self.side}.wav"
                self._rx_wav_file = wave.open(fname, "wb")
                self._rx_wav_file.setnchannels(1)
                self._rx_wav_file.setsampwidth(2)
                self._rx_wav_file.setframerate(8000)
                print(f"[DEBUG] Recording RX audio to {os.path.abspath(fname)}")
            except Exception as e:
                print(f"[WARN] Failed to open WAV recording: {e}")
            
            # --- Real-time audio playback (optional) ---
            self._audio_output = None
            self._audio_playback_enabled = os.environ.get("NADE_AUDIO_PLAYBACK", "0") == "1"
            if self._audio_playback_enabled and pyaudio:
                try:
                    pa = pyaudio.PyAudio()
                    # Use larger buffer (1024 instead of 160) to reduce choppiness
                    self._audio_output = pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=8000,
                        output=True,
                        frames_per_buffer=1024
                    )
                    self._pa = pa  # Keep reference to prevent garbage collection
                    print(f"[AUDIO] Real-time playback ENABLED for side {self.side}")
                except Exception as e:
                    print(f"[WARN] Real-time audio playback failed: {e}")
                    self._audio_output = None
            elif self._audio_playback_enabled:
                print("[WARN] NADE_AUDIO_PLAYBACK=1 but pyaudio not available")
            
            # --- TX Audio Input from WAV file (optional) ---
            self._tx_wav_samples = None
            self._tx_wav_pos = 0
            
            # --- RX Audio time-stretch state ---
            # Track when we last wrote audio to maintain proper timing
            self._last_rx_audio_write_t = 0
            self._last_decoded_pcm = None  # Hold last decoded PCM for gap filling
            tx_wav_path = os.environ.get("NADE_TX_WAV", "")
            if tx_wav_path and os.path.exists(tx_wav_path):
                try:
                    import wave
                    with wave.open(tx_wav_path, 'rb') as wf:
                        src_rate = wf.getframerate()
                        src_channels = wf.getnchannels()
                        n_frames = wf.getnframes()
                        raw = wf.readframes(n_frames)
                        
                    # Convert to numpy
                    samples = np.frombuffer(raw, dtype=np.int16)
                    
                    # Convert stereo to mono
                    if src_channels == 2:
                        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    
                    # Resample if needed (simple decimation, not ideal but functional)
                    if src_rate != 8000:
                        ratio = src_rate / 8000
                        indices = (np.arange(len(samples) // ratio) * ratio).astype(int)
                        samples = samples[indices]
                    
                    self._tx_wav_samples = samples
                    print(f"[TX_WAV] Loaded {len(samples)} samples ({len(samples)/8000:.1f}s) from {tx_wav_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load TX WAV: {e}")

            self._handshake_started = False
            self._pending_handshake = True
            self._both_handshake_complete = False
            self._audio_tx_sdu_q.clear()
            
            print(f"[DEBUG] Start completed. Noise={self._noise is not None} AudioStack={self._audio_stack is not None} RealAudio={self._use_real_audio}")

    def stop(self) -> None:
        if getattr(self, "_use_real_audio", False):
            self._real_audio.close()
        if getattr(self, "_rx_wav_file", None):
            try:
                self._rx_wav_file.close()
                print(f"[DEBUG] Closed RX WAV file")
            except:
                pass
        if getattr(self, "_audio_output", None):
            try:
                self._audio_output.stop_stream()
                self._audio_output.close()
                self._pa.terminate()
                print(f"[AUDIO] Closed real-time playback")
            except:
                pass
            self._rx_wav_file = None

    # ---- timers ----
    def on_timer(self, t_ms: int) -> None:
        if self.mode != "audio":
            # ByteLink mock timer
            self._byte_t_ms = t_ms
            if not self._byte_started:
                self._byte_started = True
                if self.side == INITIATOR_SIDE:  # initiator sends HS1
                    self._byte_txq.append((bytes([HANDSHAKE_TAG]) + b"HS1", self._byte_t_ms))
        return

    # ----------------------------------------------------------------------
    # BYTELINK
    # ----------------------------------------------------------------------
    def poll_link_tx(self, budget: int):
        if self.mode == "audio":
            return []
        out: List[Tuple[bytes, int]] = []
        while self._byte_txq and len(out) < budget:
            out.append(self._byte_txq.popleft())
        return out

    def on_link_rx(self, sdu: bytes):
        if self.mode == "audio" or not sdu:
            return
        if sdu[0] == HANDSHAKE_TAG:
            hs = sdu[1:]
            if hs == b"HS1":
                if not self._byte_done:
                    self._byte_txq.append((bytes([HANDSHAKE_TAG]) + b"HS2", self._byte_t_ms))
                    self._byte_done = True
            elif hs == b"HS2":
                self._byte_done = True
            return
        if self._byte_done:
            # Plaintext passthrough; previously queued internally but never used externally
            # Keep behavior no-op for external surfaces
            pass

    # ----------------------------------------------------------------------
    # AUDIOBLOCK TX (device → Protocol → Modem → DryBox)
    # ----------------------------------------------------------------------
    def push_tx_block(self, t_ms):
        real_voice = None
        
        if getattr(self, "_use_real_audio", False):
             # Throttle: Only process audio block every 20ms (8k/160 samples)
             if t_ms % 20 != 0:
                 return None
             
             # Sync with real-time clock
             import time
             now = time.time()
             if self._next_wall_time is None:
                 self._next_wall_time = now
             wait = self._next_wall_time - now
             if wait > 0:
                 time.sleep(wait)
             self._next_wall_time += 0.020
             
             # READ VOICE FROM MICROPHONE/TONE
             real_voice = self._real_audio.read(160, blocking=True)

        if not self._audio_stack:
            return real_voice if real_voice is not None else np.zeros(160, dtype=np.int16)

        # ---- Handshake TX ----
        if not ENCRYPTION_BYPASS:
            if self._pending_handshake:
                self._pending_handshake = False
                print(f"[DEBUG] pending_handshake triggered. side={self.side} noise={self._noise is not None}")
                if self._noise:
                    self._noise.start_handshake(initiator=(self.side == INITIATOR_SIDE))
                    self._audio_logger("info", f"[Noise] Starting NoiseXK handshake (initiator={self.side == INITIATOR_SIDE})")
            
            # Always try to drain queued handshake messages (including final M3)
            if self._noise:
                hs_msg = self._noise.get_next_handshake_message()
                while hs_msg:
                    print(f"[DEBUG] TX handshake message len={len(hs_msg)}")
                    self._audio_logger("info",
                        f"[NoiseXK] TX handshake msg len={len(hs_msg)} hex={hs_msg.hex()[:32]}...")
                    # Track sent message hash for echo detection
                    self._sent_msg_hashes.add(hash(hs_msg))
                    self._audio_stack.tx_enqueue(hs_msg)
                    hs_msg = self._noise.get_next_handshake_message()
                
                # Check if handshake is complete and we should start the delay
                if not self._both_handshake_complete and self._noise.handshake_complete:
                    self._both_handshake_complete = True
                    # Delay audio start to ensure peer has time to process the final handshake message
                    # 1000ms delay gives ample time for the peer to complete handshake
                    self._audio_start_ts = t_ms + 1000 
                    print(f"[DEBUG] Handshake complete locally. Audio start delayed to t={self._audio_start_ts}ms")
                    self._audio_logger("info", "[NoiseXK] Handshake complete locally. Encryption tunnel established.")
                
            if self._both_handshake_complete and self._noise:
                # Respect the grace period
                if getattr(self, "_audio_start_ts", 0) > t_ms:
                    return

                # FEED REAL VOICE OR SDUs INTO MODEM
                if real_voice is not None:
                     # TODO: codec2.compress(real_voice) -> bits
                     # MOCK: Encrypt a chunk of voice to simulate traffic
                     payload = real_voice.tobytes()[:40] 
                     encrypted = self._noise.encrypt_sdu(b"", payload)
                     self._audio_stack.tx_enqueue(encrypted)
                
                # Fallback: If no real audio and queue empty, generate traffic from WAV or mock
                # This ensures simulation has data to transmit
                elif not getattr(self, "_use_real_audio", False):
                    # Generate voice frame every 40ms (25 pps)
                    if t_ms % 40 == 0:
                        try:
                            # Use WAV file samples if available
                            if getattr(self, "_tx_wav_samples", None) is not None:
                                # Read 40 bytes (20 samples) from WAV, loop if exhausted
                                chunk_samples = 20  # 20 int16 samples = 40 bytes
                                start = self._tx_wav_pos
                                end = start + chunk_samples
                                if end >= len(self._tx_wav_samples):
                                    # Loop back to beginning
                                    self._tx_wav_pos = 0
                                    start = 0
                                    end = chunk_samples
                                    
                                payload = self._tx_wav_samples[start:end].tobytes()
                                self._tx_wav_pos = end
                            else:
                                # Fallback: Create recognizable mock pattern
                                ctr = (t_ms // 40) & 0xFF
                                payload = bytes([ctr]) + os.urandom(39)
                            
                            encrypted = self._noise.encrypt_sdu(b"", payload)
                            if self._audio_stack.tx_enqueue(encrypted):
                                if t_ms % 1000 == 0:
                                    src = "WAV" if getattr(self, "_tx_wav_samples", None) is not None else "mock"
                                    self._audio_logger("debug", f"[{src}] Sent voice frame t={t_ms}")
                        except Exception as e:
                            print(f"[ERROR] TX gen failed: {e}")

                if self._audio_tx_sdu_q:
                    sdu = self._audio_tx_sdu_q.popleft()
                    encrypted = self._noise.encrypt_sdu(b"", sdu)
                    self._audio_stack.tx_enqueue(encrypted)
        else:
            # BYPASS MODE: Skip crypto, send raw audio frames
            if not getattr(self, "_use_real_audio", False):
                # Generate mock voice frame every 40ms (25 pps)
                if t_ms % 40 == 0:
                    # Create recognizable mock pattern - a simple tone-like pattern
                    ctr = (t_ms // 40) & 0xFF
                    # Create 40-byte payload with recognizable pattern
                    payload = bytes([0xAA, ctr]) + bytes(range(38))
                    self._audio_stack.tx_enqueue(payload)
                    if t_ms % 500 == 0:
                        print(f"[BYPASS] TX raw frame at t={t_ms}ms, len={len(payload)}")
        

        # MODULATE TO PCM CHIRPS
        pcm_out = self._audio_stack.push_tx_block(t_ms)
        
        # DEBUG: Check modem output
        if pcm_out is not None and pcm_out.size > 0:
            max_amp = np.max(np.abs(pcm_out))
            if t_ms % 100 == 0:
                print(f"[DEBUG] TX pcm_out t={t_ms}ms size={pcm_out.size} max_amp={max_amp}")

        # NOTE: Do NOT normalize here - the modem outputs at correct amplitude (9000).
        # Scaling to 32767 causes vocoder saturation and corrupts the FSK signal.

        return pcm_out

    # ----------------------------------------------------------------------
    # AUDIOBLOCK RX (DryBox → Modem → Protocol → Speakers)
    # ----------------------------------------------------------------------
    def pull_rx_block(self, pcm, t_ms):
        print(f"[TRACE] pull_rx_block t={t_ms} code_fresh=2")
        if not self._audio_stack:
            return

        # ---- Handshake Init (Delayed) ----
        if self._pending_handshake:
            self._pending_handshake = False
            if self._noise:
                self._noise.start_handshake(initiator=(self.side == INITIATOR_SIDE))
        
        # DEMODULATE PCM TO BYTES
        self._audio_stack.pull_rx_block(pcm, t_ms)
        payloads = self._audio_stack.pop_rx_frames()
        
        # DEBUG: Check demodulator output
        if payloads:
            print(f"[DEBUG] RX frames received: {len(payloads)} at t={t_ms}ms")
        elif t_ms % 100 == 0:
            pcm_max = np.max(np.abs(pcm)) if pcm is not None and len(pcm) > 0 else 0
            print(f"[DEBUG] RX no frames at t={t_ms}ms, pcm_max={pcm_max}")

        for frame in payloads:
            if ENCRYPTION_BYPASS:
                # BYPASS MODE: Treat frame as raw audio data
                if len(frame) >= 2 and frame[0] == 0xAA:
                    # Valid bypass frame detected
                    ctr = frame[1]
                    print(f"[BYPASS] [{self.side}] RX raw frame at t={t_ms}ms, ctr={ctr}, len={len(frame)}")
                    
                    # Convert to PCM and write to WAV
                    # Create a simple tone based on the counter for audible output
                    if self._rx_wav_file:
                        # Generate a simple audible tone (sine-like pattern using counter)
                        import struct
                        samples = []
                        for i in range(160):
                            # Create a simple buzzing tone
                            val = int(8000 * (((i + ctr * 10) % 32) / 16.0 - 1.0))
                            samples.append(val)
                        pcm_data = struct.pack('<' + 'h' * 160, *samples)
                        
                        print(f"[WAV_DEBUG] Writing {len(pcm_data)} bytes to WAV at t={t_ms}")
                        try:
                            self._rx_wav_file.writeframes(pcm_data)
                        except Exception as e:
                            print(f"[WAV_ERROR] Write failed: {e}")
                continue
            
            if not self._noise:
                continue

            # -------- echo detection --------
            # Skip frames that we sent ourselves (echo from channel)
            frame_hash = hash(frame)
            if frame_hash in self._sent_msg_hashes:
                print(f"[{self.side}] Skipping echo frame (hash match)")
                continue

            # -------- handshake path --------
            # If we haven't confirmed handshake globally, check the noise wrapper
            if not self._both_handshake_complete and not self._noise.handshake_complete:
                try:
                    print(f"[DEBUG] [{self.side}] Processing handshake msg len={len(frame)} hex={frame.hex()[:32]}...")
                    self._noise.process_handshake_message(frame)
                    print(f"[DEBUG] [{self.side}] Handshake msg processed OK. Complete={self._noise.handshake_complete}")
                    # Track received message too (in case of retransmission from peer)
                    self._sent_msg_hashes.add(frame_hash)
                except IndexError:
                    # Handshake pattern exhausted - we're done
                    print(f"[{self.side}] Handshake pattern exhausted (IndexError) - forcing COMPLETE")
                    self._both_handshake_complete = True
                except Exception as e:
                    print(f"[{self.side}] Handshake error: {type(e).__name__}: {e}")
                continue

            # -------- encrypted data path --------
            try:
                decrypted = self._noise.decrypt_sdu(b"", frame)
                print(f"[{self.side}] Decryption SUCCESS: len={len(decrypted)}")
                
                # The decrypted payload is 40 bytes = 20 int16 samples
                # This represents ~2.5ms of audio at 8kHz
                # But the modem frame period is ~76ms (time to transmit one frame)
                # We need to time-stretch to maintain proper playback timing
                
                if len(decrypted) >= 40:
                    # Parse the 20 samples from 40 bytes
                    pcm_short = np.frombuffer(decrypted[:40], dtype=np.int16)
                    
                    # Time-stretch: repeat/interpolate to fill the 76ms gap
                    # 76ms at 8kHz = 608 samples
                    # Simple approach: repeat the 20-sample pattern ~30 times
                    MODEM_FRAME_MS = 76
                    samples_needed = int(MODEM_FRAME_MS * 8)  # 608 samples
                    
                    # Repeat the short sample pattern to fill the frame period
                    pcm_stretched = np.tile(pcm_short, samples_needed // len(pcm_short) + 1)[:samples_needed]
                    
                    # Store for gap filling
                    self._last_decoded_pcm = pcm_stretched
                    
                    if getattr(self, "_use_real_audio", False):
                        self._real_audio.write(pcm_stretched)
                    
                    # Real-time audio playback (if enabled)
                    if getattr(self, "_audio_output", None):
                        try:
                            self._audio_output.write(pcm_stretched.tobytes())
                        except Exception as e:
                            print(f"[AUDIO] Playback error: {e}")
                    
                    # Write to WAV file for offline playback
                    if self._rx_wav_file:
                        data = pcm_stretched.tobytes()
                        print(f"[WAV_DEBUG] Writing {len(data)} bytes ({len(pcm_stretched)} samples) to WAV at t={t_ms}")
                        try:
                            self._rx_wav_file.writeframes(data)
                            self._last_rx_audio_write_t = t_ms
                        except Exception as e:
                            print(f"[WAV_ERROR] Write failed: {e}")
                else:
                    print(f"[TRACE] Decrypted len={len(decrypted)} < 40")
                    self._audio_logger("info", f"[Nade] RX decrypted SDU len={len(decrypted)}")
            except:
                import traceback
                traceback.print_exc()
                self._audio_logger("error", f"[Noise] Decryption failed (logged traceback)")

        if not ENCRYPTION_BYPASS and self._noise.handshake_complete and not self._both_handshake_complete:
            self._both_handshake_complete = True
            print("[DEBUG] Handshake COMPLETED on RX side")
            self.ctx.emit_event("handshake_complete", {"side": self.side})
    # ----------------------------------------------------------------------
    # API: external system injects SDUs into Audio mode
    # ----------------------------------------------------------------------
    def send_sdu(self, data: bytes):

        if not self._noise or not self._noise.handshake_complete:
            self._audio_logger("warn",
                f"[Nade] Tried to send SDU but handshake is incomplete. Dropped.")
            return

        try:
            ct = self._noise.encrypt_sdu(b"", data)
        except Exception as e:
            self._audio_logger("error",
                f"[NoiseXK] encrypt_sdu FAILED: {e} plaintext={data!r}")
            return

        self._audio_logger(
            "info",
            f"[Nade] Queue TX SDU plaintext_len={len(data)} ct_len={len(ct)}"
        )

        self._audio_tx_sdu_q.append(ct)

    
    # ----------------------------------------------------------------------
    # Logger
    # ----------------------------------------------------------------------
    def _audio_logger(self, level: str, payload: Any) -> None:
        if level == "metric" and isinstance(payload, dict):
            self.ctx.emit_event("metric", payload)
        elif isinstance(payload, str):
            self.ctx.emit_event("log", {"level": "info", "msg": payload})
        else:
            self.ctx.emit_event("log", {"level": str(level), "msg": str(payload)})
