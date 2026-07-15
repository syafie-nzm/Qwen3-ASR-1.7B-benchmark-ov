"""
Real-time streaming ASR using VAD-segmented chunking with Qwen3-ASR OpenVINO.

Two modes:
  mic     (default) — captures mic locally; use only if running directly on
                       a machine with a microphone.
  server            — listens on a TCP port for raw float32 PCM audio streamed
                       by stream_client.py running on your local machine.
                       Use this when running on a remote/headless server via SSH.

Usage:
  # local machine (mic attached):
  python qwen3_stream_vad.py --mode mic

  # remote server (run this first, then stream_client.py on local machine):
  python qwen3_stream_vad.py --mode server --host 0.0.0.0 --port 9876

Pipeline:
  [mic | TCP socket] → Silero VAD → utterance buffer → Qwen3-ASR OpenVINO → printed text

Silero VAD detects speech/silence at 32ms resolution (512 samples @ 16kHz).
When silence longer than SILENCE_THRESHOLD is detected after speech,
the accumulated utterance is sent to Qwen3-ASR for transcription.
"""

import argparse
import queue
import socket
import sys
import threading
from pathlib import Path
import time
import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

from qwen_3_asr_helper import OVQwen3ASRModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # Hz — Qwen3 and Silero both require 16kHz
VAD_CHUNK = 512               # samples — Silero VAD fixed requirement at 16kHz
CHUNK_BYTES = VAD_CHUNK * 4   # float32 = 4 bytes per sample
SILENCE_THRESHOLD_SEC = 0.4   # seconds of silence that marks end-of-utterance
MIN_UTTERANCE_SEC = 0.3       # discard segments shorter than this (noise/clicks)
VAD_SPEECH_THRESHOLD = 0.5    # Silero probability above which chunk is speech
MODEL_DIR = Path("Qwen/Qwen3-ASR-1.7B-OV")  # OpenVINO model directory
DEVICE = "GPU"                # change to "CPU" or "NPU" as needed
DEFAULT_HOST = "0.0.0.0"      # server listen address
DEFAULT_PORT = 8000           # server listen port

# Optional domain context for guided transcription (can be empty)
context = "Transcribe in comma-separated clinical keyword/list style without paraphrasing. Preserve dictated order and short tokens like A, B, C, RSJ. Use this lower-GI term bank when acoustically plausible: insertion level, terminal ileum, cecum, ascending colon, hepatic flexure, transverse colon, splenic flexure, descending colon, sigmoid colon, rectum, anastomosis, anus, premedication, colon cleansing agent, preparation time, morning single dose, evening single dose, split dose, colon cleansing level, excellent, good, fail, poor finding, A, normal, negative finding, negative finding in the observable segment, poor preparation, B, hemorrhoids, external hemorrhoids, mixed hemorrhoids, internal hemorrhoids, C, polyp, hyperplastic polyp, tubular adenoma, tubulovillous adenoma, villous adenoma, sessile serrated lesion, SSL, traditional serrated adenoma, post-treatment residual neoplasm, inflammatory polyp, juvenile polyp, Peutz-Jeghers syndrome, colon polyposis, familiar, colon polyposis, early colorectal cancer, advanced colorectal cancer, lymphangioma, lipoma, carcinoid, submucosal tumor, colonmaltoma, lymphoma, colitis, non-specific colitis, ischemic colitis, infectious colitis, amebic colitis, ulcerative colitis, radiation colitis, pseudo-membranous colitis, drug induced colitis, cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, colonic ulcer, Bechet's disease, proctitis, hemorrhagic colitis, colitis aphthosa, colonic diverticulum, chronic diverticulosis, melanosis coli, xanthoma, post partial colectomy, post left hemicolectomy, post right hemicolectomy, situs inversus, colonic wall cyst, angiodysplasia, angiectasia, lymphoid follicles, operation scar, suture granuloma, petechia, colonic tuberculosis, amyloidosis, mega colon, rectal varices, mucosal prolapse, intussusception, colon fistula, post endoscopy treatment scar, colonic stricture, rectosigmoid junction, RSJ."

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading Silero VAD...", flush=True)
vad_model = load_silero_vad()
vad_model.eval()

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"OpenVINO model directory not found: {MODEL_DIR}")

print("Loading Qwen3-ASR OpenVINO...", flush=True)
asr_model = OVQwen3ASRModel.from_pretrained(
    model_dir=str(MODEL_DIR),
    device=DEVICE,
    max_inference_batch_size=-1,
    max_new_tokens=512,
)

# ---------------------------------------------------------------------------
# Shared queues
# ---------------------------------------------------------------------------
audio_q: queue.Queue[np.ndarray | None] = queue.Queue()
utterance_q: queue.Queue[np.ndarray | None] = queue.Queue()


# ---------------------------------------------------------------------------
# Transcription worker (runs in a dedicated thread)
# ---------------------------------------------------------------------------
def transcribe_worker() -> None:
    """Pull utterances from the queue, run Qwen3-ASR, print results."""
    while True:
        audio_np = utterance_q.get()
        if audio_np is None:  # poison pill — shut down
            break

        try:
            start = time.perf_counter()
            # Qwen3-ASR transcribe API: expects (numpy_array, sample_rate) tuple
            results = asr_model.transcribe(
                audio=(audio_np, SAMPLE_RATE),
                language="English",  # auto-detect language
                context=context,
            )
            end = time.perf_counter()
            time_taken = end - start

            text = results[0].text.strip()
            language = results[0].language
            
            # Extract only transcribed text, removing system context and template artifacts
            if text:
                # Look for <asr_text> marker which indicates start of actual transcription
                if "<asr_text>" in text:
                    # Everything after <asr_text> is the transcription
                    text = text.split("<asr_text>", 1)[1].strip()
                
                # Remove any "language" prefix that may appear
                if text.lower().startswith("language "):
                    # Remove "language English" or similar prefixes
                    parts = text.split(None, 1)  # split on first whitespace
                    if len(parts) > 1:
                        text = parts[1]
                
                # Only print if there's actual transcribed text remaining
                if text:
                    print(f"\n>>> [{language}] {text}", flush=True)
                    print(f"Time taken: {time_taken*1000:.2f} ms", flush=True)
        except Exception as e:
            print(f"\n[ERROR] Transcription failed: {e}", file=sys.stderr, flush=True)

        utterance_q.task_done()


# ---------------------------------------------------------------------------
# Audio sources
# ---------------------------------------------------------------------------
def audio_callback(
    indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags
) -> None:
    """sounddevice callback — used in 'mic' mode."""
    if status:
        print(f"[audio] {status}", file=sys.stderr)
    audio_q.put(indata[:, 0].copy())


def tcp_receiver(host: str, port: int, stop_event: threading.Event) -> None:
    """
    Server mode: accept a single client connection and read raw float32 PCM
    chunks (VAD_CHUNK samples each) sent by stream_client.py.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(1.0)
    print(f"[server] Waiting for client on {host}:{port} ...", flush=True)

    conn = None
    while not stop_event.is_set():
        try:
            conn, addr = srv.accept()
            break
        except socket.timeout:
            continue

    if conn is None:
        return

    print(f"[server] Client connected from {addr}", flush=True)
    conn.settimeout(1.0)

    try:
        while not stop_event.is_set():
            data = b""
            while len(data) < CHUNK_BYTES:
                try:
                    packet = conn.recv(CHUNK_BYTES - len(data))
                except socket.timeout:
                    if stop_event.is_set():
                        return
                    continue
                if not packet:
                    print("[server] Client disconnected.", flush=True)
                    return
                data += packet
            chunk = np.frombuffer(data, dtype=np.float32).copy()
            audio_q.put(chunk)
    finally:
        conn.close()
        srv.close()


# ---------------------------------------------------------------------------
# VAD loop (runs in a dedicated thread)
# ---------------------------------------------------------------------------
def vad_loop(stop_event: threading.Event) -> None:
    """
    Consume raw audio chunks from audio_q.
    Accumulate speech frames; flush to utterance_q when silence detected.
    """
    speech_buffer: list[np.ndarray] = []
    silence_count = 0
    speaking = False
    silence_limit = int(SILENCE_THRESHOLD_SEC * SAMPLE_RATE / VAD_CHUNK)

    vad_model.reset_states()

    while not stop_event.is_set():
        try:
            chunk = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if chunk is None:
            break

        tensor = torch.from_numpy(chunk).float()
        speech_prob: float = vad_model(tensor, SAMPLE_RATE).item()  # type: ignore[operator]
        is_speech = speech_prob >= VAD_SPEECH_THRESHOLD

        if is_speech:
            if not speaking:
                print(".", end="", flush=True)  # visual indicator: speech started
            speaking = True
            silence_count = 0
            speech_buffer.append(chunk)

        elif speaking:
            # Still in post-speech tail — include for natural end of words
            speech_buffer.append(chunk)
            silence_count += 1

            if silence_count >= silence_limit:
                # Utterance complete — dispatch for transcription
                utterance = np.concatenate(speech_buffer)
                duration = len(utterance) / SAMPLE_RATE
                if duration >= MIN_UTTERANCE_SEC:
                    utterance_q.put(utterance)

                speech_buffer.clear()
                silence_count = 0
                speaking = False
                vad_model.reset_states()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR OpenVINO — VAD streaming")
    parser.add_argument(
        "--mode",
        choices=["mic", "server"],
        default="mic",
        help="'mic': capture local microphone. 'server': receive audio over TCP.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server bind address (server mode only).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server TCP port (server mode only).")
    parser.add_argument("--device", default=DEVICE, help="OpenVINO device (CPU, GPU, NPU).")
    args = parser.parse_args()

    stop_event = threading.Event()

    transcribe_thread = threading.Thread(target=transcribe_worker, daemon=True)
    transcribe_thread.start()

    vad_thread = threading.Thread(target=vad_loop, args=(stop_event,), daemon=True)
    vad_thread.start()

    print("\nDots (.) indicate detected speech. Transcription appears after each pause.")
    print(f"Model: {MODEL_DIR} | Device: {args.device}")
    print("-" * 60)

    try:
        if args.mode == "server":
            # TCP receiver runs on main thread; no sounddevice needed on the server
            tcp_receiver(args.host, args.port, stop_event)
        else:
            print("Listening on microphone... (Ctrl+C to stop)", flush=True)
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
        print("\n\nStopped.")
    finally:
        stop_event.set()
        audio_q.put(None)      # unblock vad_loop
        utterance_q.put(None)  # unblock transcribe_worker
        vad_thread.join(timeout=2)
        transcribe_thread.join(timeout=10)


if __name__ == "__main__":
    main()
