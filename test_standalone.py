import pyaudio
import numpy as np
from typing import Optional, Tuple
import logging
import time

# --- COPIED FROM nade/streamer.py ---
class AudioStreamer:
    """
    Handles audio I/O using PyAudio.
    Supports 2 inputs / 2 outputs (Stereo) at a configurable sample rate and buffer size.
    """

    def __init__(self, 
                 rate: int = 8000, 
                 chunk_size: int = 160, 
                 channels: int = 2,
                 input_device_index: Optional[int] = None,
                 output_device_index: Optional[int] = None):
        """
        Initialize the AudioStreamer.

        Args:
            rate: Sample rate in Hz (default: 8000).
            chunk_size: Number of frames per buffer (default: 160).
            channels: Number of channels (default: 2 for Stereo).
            input_device_index: Index of the input device (None for default).
            output_device_index: Index of the output device (None for default).
        """
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.logger = logging.getLogger(__name__)

        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                output=True,
                input_device_index=input_device_index,
                output_device_index=output_device_index,
                frames_per_buffer=chunk_size
            )
            self.logger.info(f"AudioStreamer initialized: {rate}Hz, {channels}ch, {chunk_size} frames/chunk")
        except Exception as e:
            self.logger.error(f"Failed to open audio stream: {e}")
            self.close()
            raise

    def read_block(self) -> np.ndarray:
        """
        Read a block of audio from the input stream.
        
        Returns:
            np.ndarray: A numpy array of shape (chunk_size, channels) with dtype int16.
        """
        if not self.stream:
            raise RuntimeError("Stream is not open")
        
        try:
            # Read raw bytes
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            # Convert to numpy array
            # Expected byte length = chunk_size * channels * 2 (for int16)
            arr = np.frombuffer(data, dtype=np.int16)
            # Reshape to (chunk_size, channels)
            return arr.reshape((self.chunk_size, self.channels))
        except IOError as e:
            self.logger.warning(f"Audio read overflow/error: {e}")
            return np.zeros((self.chunk_size, self.channels), dtype=np.int16)

    def write_block(self, data: np.ndarray) -> None:
        """
        Write a block of audio to the output stream.

        Args:
            data: np.ndarray of shape (chunk_size, channels) with dtype int16.
        """
        if not self.stream:
            raise RuntimeError("Stream is not open")
        
        # Ensure data is int16 and C-contiguous
        if data.dtype != np.int16:
            data = data.astype(np.int16)
        
        try:
            self.stream.write(data.tobytes())
        except IOError as e:
            self.logger.warning(f"Audio write underflow/error: {e}")

    def close(self):
        """
        Close the stream and terminate PyAudio.
        """
        if self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.pa:
            self.pa.terminate()
            self.pa = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# --- VERIFICATION LOGIC ---
def test_audio_streamer():
    print("Testing AudioStreamer standalone...")
    try:
        # 160 frames, stereo, 8000Hz
        with AudioStreamer(rate=8000, chunk_size=160, channels=2) as stream:
            print("Stream initialized successfully.")
            
            print("Reading block...")
            data = stream.read_block()
            print(f"Read block with shape: {data.shape}, dtype: {data.dtype}")
            
            # Verify shape
            if data.shape != (160, 2):
                print(f"FAIL: Expected shape (160, 2), got {data.shape}")
            else:
                print("PASS: Shape is correct.")
                
            # Loopback test (short)
            print("Running 0.5 second loopback...")
            start_time = time.time()
            blocks = 0
            while time.time() - start_time < 0.5:
                data = stream.read_block()
                stream.write_block(data)
                blocks += 1
            print(f"Processed {blocks} blocks.")
            
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_streamer()
