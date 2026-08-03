"""Split an audio file into fixed-count or fixed-duration chunks."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO = "./endoscopy_internal.wav"
DEFAULT_NUM_CHUNKS = 100
DEFAULT_OUTPUT_DIR = "./chunks"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def chunk_bounds(num_frames: int, num_chunks: int, chunk_frames: int | None) -> list[tuple[int, int]]:
    if chunk_frames is not None:
        return [
            (start, min(start + chunk_frames, num_frames))
            for start in range(0, num_frames, chunk_frames)
        ]

    edges = np.linspace(0, num_frames, num_chunks + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(num_chunks) if edges[i + 1] > edges[i]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Input audio file.")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help="Number of equal-length chunks to produce.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="If set, split by this fixed duration instead of --num-chunks.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for chunk files.")
    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1.")
    if args.chunk_seconds is not None and args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive.")

    audio_path = resolve_path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    num_frames = audio.shape[0]
    chunk_frames = int(round(args.chunk_seconds * sample_rate)) if args.chunk_seconds else None

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bounds = chunk_bounds(num_frames, args.num_chunks, chunk_frames)
    width = len(str(len(bounds)))

    print(f"Input: {audio_path}")
    print(f"Duration: {num_frames / sample_rate:.2f}s @ {sample_rate} Hz, channels={audio.shape[1]}")
    print(f"Writing {len(bounds)} chunk(s) to {output_dir}")

    for index, (start, end) in enumerate(bounds, start=1):
        out_path = output_dir / f"{audio_path.stem}_chunk{index:0{width}d}.wav"
        sf.write(str(out_path), audio[start:end], sample_rate)
        print(
            f"  [{index:>{width}}] {start / sample_rate:8.2f}s -> {end / sample_rate:8.2f}s "
            f"({(end - start) / sample_rate:6.2f}s)  {out_path.name}"
        )


if __name__ == "__main__":
    main()
