"""Split an audio file into speech-only chunks using Silero VAD.

Companion to chunk_audio.py, which splits blindly by count/duration. This script
runs Silero VAD over the whole file offline, keeps only the detected speech
regions, and writes them as 16 kHz mono WAVs ready for Qwen3-ASR.

Usage:
  python chunk_audio_vad.py
  python chunk_audio_vad.py --audio ./endoscopy_internal.wav --max-chunk-seconds 20
  python qwen3_asr_benchmark_chunks.py --chunks-dir ./chunks_vad --warmup 1
"""

import argparse
import csv
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO = "./endoscopy_internal.wav"
DEFAULT_OUTPUT_DIR = "./chunks_vad"
TARGET_SR = 16_000

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SILENCE_MS = 400
DEFAULT_SPEECH_PAD_MS = 100
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MIN_CHUNK_SEC = 0.3
DEFAULT_MAX_CHUNK_SEC = 30.0


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def load_mono_16k(audio_path: Path) -> np.ndarray:
    """Read any WAV/FLAC/etc. as float32 mono at TARGET_SR."""
    audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if sample_rate != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=TARGET_SR)
    return np.ascontiguousarray(mono, dtype=np.float32)


def split_long(start: int, end: int, max_frames: int) -> list[tuple[int, int]]:
    """Cut [start, end) into near-equal parts no longer than max_frames."""
    length = end - start
    if length <= max_frames:
        return [(start, end)]

    parts = math.ceil(length / max_frames)
    edges = np.linspace(start, end, parts + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(parts) if edges[i + 1] > edges[i]]


def refine_segments(
    segments: list[tuple[int, int]],
    min_frames: int,
    max_frames: int,
) -> tuple[list[tuple[int, int]], int]:
    """Split over-long segments, absorb under-long ones into the previous chunk.

    Returns the kept segments and the number of short segments dropped outright.
    """
    kept: list[tuple[int, int]] = []
    dropped = 0

    for start, end in segments:
        for piece_start, piece_end in split_long(start, end, max_frames):
            if piece_end - piece_start >= min_frames:
                kept.append((piece_start, piece_end))
                continue

            # Too short on its own: extend the previous chunk if that keeps it
            # under the max, otherwise discard it.
            if kept and piece_end - kept[-1][0] <= max_frames:
                kept[-1] = (kept[-1][0], piece_end)
            else:
                dropped += 1

    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Input audio file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for chunk files.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Silero speech probability above which a frame counts as speech.",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=DEFAULT_MIN_SILENCE_MS,
        help="Silence this long ends a speech segment.",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=DEFAULT_SPEECH_PAD_MS,
        help="Padding added to both ends of every detected segment.",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=DEFAULT_MIN_SPEECH_MS,
        help="Silero discards detected speech shorter than this.",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=DEFAULT_MIN_CHUNK_SEC,
        help="Chunks shorter than this are merged into the previous chunk, or dropped.",
    )
    parser.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=DEFAULT_MAX_CHUNK_SEC,
        help="Segments longer than this are split into near-equal parts.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only write the first N chunks.")
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Also write segments.csv mapping each chunk back to its source timestamps.",
    )
    args = parser.parse_args()

    if args.min_chunk_seconds <= 0:
        raise ValueError("--min-chunk-seconds must be positive.")
    if args.max_chunk_seconds <= args.min_chunk_seconds:
        raise ValueError("--max-chunk-seconds must be greater than --min-chunk-seconds.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    audio_path = resolve_path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = load_mono_16k(audio_path)
    total_sec = len(audio) / TARGET_SR

    print(f"Input: {audio_path}")
    print(f"Duration: {total_sec:.2f}s, resampled to {TARGET_SR} Hz mono")
    print("Loading Silero VAD...", flush=True)

    vad_model = load_silero_vad()
    vad_model.eval()

    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
        vad_model,
        sampling_rate=TARGET_SR,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        return_seconds=False,
    )
    segments = [(int(item["start"]), int(item["end"])) for item in timestamps]
    if not segments:
        print("No speech detected — nothing written.")
        return

    kept, dropped = refine_segments(
        segments,
        min_frames=int(round(args.min_chunk_seconds * TARGET_SR)),
        max_frames=int(round(args.max_chunk_seconds * TARGET_SR)),
    )
    if args.limit is not None:
        kept = kept[: args.limit]

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width = len(str(len(kept)))

    print(f"VAD found {len(segments)} speech segment(s) -> {len(kept)} chunk(s), {dropped} dropped")
    print(f"Writing to {output_dir}")

    rows: list[dict[str, object]] = []
    speech_frames = 0

    for index, (start, end) in enumerate(kept, start=1):
        out_path = output_dir / f"{audio_path.stem}_vad{index:0{width}d}.wav"
        sf.write(str(out_path), audio[start:end], TARGET_SR, subtype="PCM_16")
        speech_frames += end - start

        start_sec, end_sec = start / TARGET_SR, end / TARGET_SR
        print(
            f"  [{index:>{width}}] {start_sec:8.2f}s -> {end_sec:8.2f}s "
            f"({end_sec - start_sec:6.2f}s)  {out_path.name}"
        )
        rows.append(
            {
                "chunk_index": index,
                "chunk_name": out_path.name,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
            }
        )

    speech_sec = speech_frames / TARGET_SR
    print(
        f"Speech kept: {speech_sec:.2f}s of {total_sec:.2f}s "
        f"({speech_sec / total_sec * 100:.1f}%), silence removed: {total_sec - speech_sec:.2f}s"
    )

    if args.manifest:
        manifest_path = output_dir / "segments.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
