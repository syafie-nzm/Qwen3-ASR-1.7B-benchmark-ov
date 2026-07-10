"""
stream_client.py — run this on your LOCAL machine.

Captures microphone audio and streams raw float32 PCM chunks over TCP to
cohere_stream_vad.py running in server mode on the remote machine.

Requirements on local machine (no GPU needed):
    pip install sounddevice numpy

Usage:
    python stream_client.py --host <remote-server-ip> --port 9876
"""

import argparse
import socket
import sys

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000   # must match server
VAD_CHUNK = 512        # must match server
CHUNK_BYTES = VAD_CHUNK * 4  # float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Mic-to-TCP audio streamer")
    parser.add_argument("--host", required=True, help="Remote server IP or hostname.")
    parser.add_argument("--port", type=int, default=9876, help="Remote server TCP port.")
    args = parser.parse_args()

    print(f"Connecting to {args.host}:{args.port} ...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print("Connected. Streaming microphone audio — speak now. (Ctrl+C to stop)")

    def audio_callback(
        indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags
    ) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        chunk = indata[:, 0].astype(np.float32).tobytes()
        try:
            sock.sendall(chunk)
        except (BrokenPipeError, OSError):
            raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_CHUNK,
            callback=audio_callback,
        ):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
