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
