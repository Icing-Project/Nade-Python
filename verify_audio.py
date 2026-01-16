import sys
import os
sys.path.append(os.getcwd())
from nade.streamer import AudioStreamer
import numpy as np
import time

def test_audio_streamer():
    print("Initializing AudioStreamer...")
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
            print("Running 1 second loopback...")
            start_time = time.time()
            blocks = 0
            while time.time() - start_time < 1.0:
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
