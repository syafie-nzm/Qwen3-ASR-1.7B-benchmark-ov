# Qwen3-ASR-1.7B Benchmark (OpenVINO)

## Prerequisites

- Python 3.10+ installed
- `uv` installed

Install `uv` using the official guide:

- https://docs.astral.sh/uv/getting-started/installation/

Quick install examples:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
uv sync
```

## Benchmark

1. Copy `.env.example` to `.env` and edit the values.
2. Run:

```bash
uv run qwen3_asr_benchmark.py
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `QWEN_MODEL_ID` | HF model id or local source path for conversion | `Qwen/Qwen3-ASR-1.7B` |
| `MODEL_PRECISION` | `int8` or `full_precision` (aliases: `full`, `fp16`, `unquantized`) | `int8` |
| `DEVICE` | OpenVINO device: `CPU`, `GPU`, or `NPU` | `GPU` |
| `SAMPLE_AUDIO` | Path to the audio file to transcribe | — |
| `MODEL_OUTPUT_ROOT` | Base folder used to resolve the model directory | `./Qwen` |
| `MODEL_DIR` | Explicit model directory; overrides `MODEL_OUTPUT_ROOT` when set | — |
| `MAX_NEW_TOKENS` | Token generation limit | `512` |

Model directory resolution:

- If the target directory does not exist, the model is converted automatically at the requested precision.
- If the directory already contains `config.json`, it is used as-is.
- If the directory exists but looks incomplete, the script stops — remove it or point `MODEL_DIR` elsewhere.

## Chunk-Based Latency Benchmark

This repository also includes a chunk-based latency benchmark for measuring
Qwen3-ASR on reusable audio slices. It is a TTFT-style workflow, but the
current implementation measures per-chunk transcription latency rather than
literal token-level first-token timing.

Recommended flow:

1. Split a source audio file into chunks.
2. Benchmark the resulting chunk directory.
3. Compare the generated CSV and summary files across devices or precisions.

### 1) Chunk creation

You can create chunks in two ways.

#### Fixed chunking

Use this when you want blind, equal-ish splits with no speech detection.

```bash
# split into a fixed number of chunks
uv run chunk_audio.py --audio ./endoscopy_internal.wav --num-chunks 100 --output-dir ./chunks

# split by fixed chunk length in seconds
uv run chunk_audio.py --audio ./endoscopy_internal.wav --chunk-seconds 5 --output-dir ./chunks
```

`chunk_audio.py` writes WAV chunks named like `*_chunk001.wav`, `*_chunk002.wav`, and so on.

#### VAD chunking

Use this when you want speech-only chunks. This is the recommended path for
benchmarking because it removes silence and produces reusable chunks for fair
comparisons across devices and precisions.

```bash
uv run chunk_audio_vad.py --audio ./endoscopy_internal.wav --output-dir ./chunks_vad --manifest
```

Useful tuning flags:

| Flag | Description |
|---|---|
| `--threshold` | Speech probability threshold used by Silero VAD |
| `--min-silence-ms` | Silence duration that ends a speech segment |
| `--speech-pad-ms` | Padding added to the start and end of each detected segment |
| `--min-speech-ms` | Short speech segments Silero keeps before chunk post-processing |
| `--min-chunk-seconds` | Short chunks are merged into the previous chunk or dropped |
| `--max-chunk-seconds` | Long chunks are split into near-equal parts |
| `--limit` | Optional cap on the number of chunks written |
| `--manifest` | Also write `segments.csv` with chunk timestamps |

`chunk_audio_vad.py` resamples audio to 16 kHz mono and writes 16 kHz mono PCM_16 WAV chunks into `./chunks_vad` by default.

### 2) Run the chunk benchmark

After chunking, benchmark the chunk directory with:

```bash
uv run qwen3_asr_benchmark_chunks.py --chunks-dir ./chunks_vad --warmup 1
```

Common command-line options:

| Option | Description |
|---|---|
| `--chunks-dir` | Directory containing chunked audio files |
| `--pattern` | Glob pattern used to select chunk files, for example `*.wav` |
| `--limit` | Only benchmark the first N chunks |
| `--warmup` | Number of warmup transcriptions to discard before timing starts |
| `--run-label` | Name used for the output files |
| `--output-dir` | Directory where benchmark results are written |

The benchmark script also accepts env-backed inputs such as `CHUNKS_DIR`,
`WARMUP_RUNS`, and `EXPERIMENT_NAME`.

### 3) Metrics and outputs

The chunk benchmark records per-chunk transcription latency in `time_taken_sec`,
plus chunk duration statistics, RTF, detected language, prediction text, and
concatenated WER over the full run.

Outputs are written under `./results/` by default:

| File | Description |
|---|---|
| `{run_label}_chunks.csv` | Per-chunk timing and prediction details |
| `{run_label}_chunks_summary.txt` | Aggregate summary including duration stats, RTF, and WER |

If you want to inspect the latency distribution after a run, `plot_time_distribution.py` can plot the `time_taken_sec` column from the generated CSV.

## Streaming ASR

Two script variants are available:

| Script | Behaviour |
|---|---|
| `qwen3_stream_vad_no_context.py` | Each utterance is transcribed without context or input prompt |
| `qwen3_stream_vad_with_context.py` | Each utterance is transcribed with context or input prompt |

### Local machine (microphone attached)

Run directly on the machine that has the microphone:

```bash
# without context
uv run qwen3_stream_vad_no_context.py --mode mic

# with context
uv run qwen3_stream_vad_with_context.py --mode mic
```

### Remote server (headless / SSH)

When the model runs on a remote server without a microphone, start the script in server mode on the remote machine first, then run `stream_client.py` on your local machine to stream microphone audio over TCP.

**Step 1 — on the remote server:**

```bash
# without context
uv run qwen3_stream_vad_no_context.py --mode server --host 0.0.0.0 --port 9876

# with context
uv run qwen3_stream_vad_with_context.py --mode server --host 0.0.0.0 --port 9876
```

**Step 2 — on your local machine:**

```bash
pip install sounddevice numpy

python stream_client.py --host <remote-server-ip> --port 9876
```
