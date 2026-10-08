"""Sequentially benchmark Qwen3-ASR over a directory of audio chunks and write per-chunk timings to CSV."""

import argparse
import csv
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from jiwer import wer
from dotenv import load_dotenv
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from qwen_3_asr_helper import OVQwen3ASRModel, convert_qwen3_asr_model

from nncf import CompressWeightsMode

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_DEVICE = "GPU"
DEFAULT_PRECISION = "int8"
DEFAULT_CHUNKS_DIR = "./chunks_vad"
DEFAULT_CHUNK_PATTERN = "*.wav"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_OUTPUT_ROOT = "./Qwen"
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_WARMUP_RUNS = 1
INT8_DIR_SUFFIX = "-OV-int8"
INT4_DIR_SUFFIX = "-OV-int4"
FULL_PRECISION_DIR_SUFFIX = "-OV-full-precision"

CSV_FIELDS = [
    "run_label",
    "chunk_index",
    "chunk_name",
    "chunk_path",
    "duration_sec",
    "time_taken_sec",
    "rtf",
    "dur_min_sec",
    "dur_mean_sec",
    "dur_median_sec",
    "dur_max_sec",
    "dur_stdev_sec",
    "dur_p90_sec",
    "dur_p95_sec",
    "detected_language",
    "num_words",
    "num_chars",
    "prediction",
    "timestamp_utc",
    "device",
    "precision",
    "model_dir",
]


def load_env() -> None:
    load_dotenv(SCRIPT_DIR / ".env")


def get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    stripped_value = value.strip()
    return stripped_value or default


def get_env_int(name: str, default: int) -> int:
    raw_value = get_env(name, str(default))
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer. Received: {raw_value}") from error


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def normalize_precision(raw_precision: str) -> str:
    normalized = raw_precision.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"int8", "quantized"}:
        return "int8"
    if normalized in {"full_precision", "full", "fp16", "fp32", "unquantized"}:
        return "full_precision"
    if normalized in {"int4"}:
        return "int4"
    raise ValueError("Unsupported MODEL_PRECISION. Use int8, int4, or full_precision.")


def resolve_model_dir(model_id: str, model_precision: str) -> Path:
    model_dir_override = os.getenv("MODEL_DIR")
    if model_dir_override and model_dir_override.strip():
        return resolve_path(model_dir_override)

    output_root = resolve_path(get_env("MODEL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    model_name = model_id.rstrip("/").split("/")[-1]
    suffix = INT8_DIR_SUFFIX if model_precision == "int8" else INT4_DIR_SUFFIX if model_precision == "int4" else FULL_PRECISION_DIR_SUFFIX
    return output_root / f"{model_name}{suffix}"


def ensure_model(model_id: str, model_dir: Path, model_precision: str) -> None:
    config_path = model_dir / "config.json"
    if model_dir.exists():
        if config_path.exists():
            return
        raise FileNotFoundError(
            f"Model directory exists but looks incomplete: {model_dir}. "
            "Missing config.json. Remove the directory or point MODEL_DIR somewhere else."
        )

    quantization_config = None
    if model_precision == "int8":
        quantization_config = {"mode": CompressWeightsMode.INT8_SYM}

    elif model_precision == "int4":
        quantization_config = {"mode": CompressWeightsMode.INT4_ASYM}

    print(f"Converting {model_id} to {model_dir} ({model_precision})...")
    convert_qwen3_asr_model(
        model_id=model_id,
        output_dir=model_dir,
        quantization_config=quantization_config,
    )


TRUTH = "Insertion level, Terminal ileum, Cecum, Ascending colon, Hepatic flexure, Transverse colon, Splenic flexure, Descending colon, Sigmoid Colon, Rectum, Anastomosis, Anus, Premedication, Colon cleansing agent, Preparation time, Morning single dose, Evening single dose, Split dose, Colon cleansing level, Excellent, Good, Fail, Poor finding, A, normal, Negative finding, Negative finding in the observable segment, Poor preparation, B, Hemorrhoids, External Hemorrhoids, Mixed hemorrhoids, Internal hemorrhoids, C, polyp, Hyperplastic polyp, Tubular adenoma, Tubulovillous adenoma, Villous adenoma, Sessile serrated lesion, SSL, Traditional serrated adenoma, Post-treatment residual neoplasm, Inflammatory polyp, Juvenile polyp, Peutz-Jeghers syndrome, Colon polyposis, familiar, Colon polyposis, Early colorectal cancer, Advanced colorectal cancer, Lymphangioma, Lipoma, Carcinoid, Submucosal tumor, Colonmaltoma, Lymphoma, Colitis, Non-specific colitis, Ischemic colitis, Infectious colitis, Amebic colitis, Ulcerative colitis, Radiation colitis, Pseudo-membranous colitis, Drug induced colitis, Cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, Colonic ulcer, Bechet's disease, Proctitis, Hemorrhagic colitis, Colitis aphthosa, Colonic diverticulum, Chronic diverticulosis, Melanosis coloi, Xanthoma, Post partial colectomy, Post left hemicolectomy, Post right hemicolectomy, Situs inversus, Colonic wall cyst, Angiodysplasia, Angiectasia, Lymphoid follicles, Operation scar, Suture granuloma, Petechia, Colonic tuberculosis, Amyloidosis, Mega colon, Rectal varices, Mucosal prolapse, Intussusception, Colon fistula, Post endoscopy treatment scar, Colonic stricture, Rectosigmoid junction RSJ"

# Keep the same context style used in the notebook for domain-guided transcription.
REFERENCE = "Transcribe in comma-separated clinical keyword/list style without paraphrasing. Preserve dictated order and short tokens like A, B, C, RSJ. Use this lower-GI term bank when acoustically plausible: insertion level, terminal ileum, cecum, ascending colon, hepatic flexure, transverse colon, splenic flexure, descending colon, sigmoid colon, rectum, anastomosis, anus, premedication, colon cleansing agent, preparation time, morning single dose, evening single dose, split dose, colon cleansing level, excellent, good, fail, poor finding, A, normal, negative finding, negative finding in the observable segment, poor preparation, B, hemorrhoids, external hemorrhoids, mixed hemorrhoids, internal hemorrhoids, C, polyp, hyperplastic polyp, tubular adenoma, tubulovillous adenoma, villous adenoma, sessile serrated lesion, SSL, traditional serrated adenoma, post-treatment residual neoplasm, inflammatory polyp, juvenile polyp, Peutz-Jeghers syndrome, colon polyposis, familiar, colon polyposis, early colorectal cancer, advanced colorectal cancer, lymphangioma, lipoma, carcinoid, submucosal tumor, colonmaltoma, lymphoma, colitis, non-specific colitis, ischemic colitis, infectious colitis, amebic colitis, ulcerative colitis, radiation colitis, pseudo-membranous colitis, drug induced colitis, cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, colonic ulcer, Bechet's disease, proctitis, hemorrhagic colitis, colitis aphthosa, colonic diverticulum, chronic diverticulosis, melanosis coli, xanthoma, post partial colectomy, post left hemicolectomy, post right hemicolectomy, situs inversus, colonic wall cyst, angiodysplasia, angiectasia, lymphoid follicles, operation scar, suture granuloma, petechia, colonic tuberculosis, amyloidosis, mega colon, rectal varices, mucosal prolapse, intussusception, colon fistula, post endoscopy treatment scar, colonic stricture, rectosigmoid junction, RSJ."


def audio_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def duration_stats(values: List[float]) -> dict:
    """Aggregate stats over chunk durations (VAD chunks vary widely in length)."""
    values_array = np.asarray(values, dtype=float)
    return {
        "min": float(values_array.min()),
        "mean": float(values_array.mean()),
        "median": float(np.median(values_array)),
        "max": float(values_array.max()),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p90": float(np.percentile(values_array, 90)),
        "p95": float(np.percentile(values_array, 95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-dir",
        default=get_env("CHUNKS_DIR", DEFAULT_CHUNKS_DIR),
        help="Directory containing the chunked audio files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_CHUNK_PATTERN,
        help="Glob pattern used to select chunk files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only benchmark the first N chunks (useful for smoke tests).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=get_env_int("WARMUP_RUNS", DEFAULT_WARMUP_RUNS),
        help="Discarded warmup transcriptions on the first chunk before timing starts.",
    )
    parser.add_argument(
        "--run-label",
        default=get_env("EXPERIMENT_NAME", ""),
        help="Label used to name the CSV and summary files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory where the CSV and summary are written.",
    )
    return parser.parse_args()


def write_summary(
    summary_path: Path,
    run_label: str,
    model_dir: Path,
    device: str,
    precision: str,
    chunks_dir: Path,
    times: List[float],
    durations: List[float],
    raw_wer: float,
    normalized_wer: float,
    csv_path: Path,
) -> None:
    times_array = np.asarray(times, dtype=float)
    total_time = float(times_array.sum())
    total_duration = float(sum(durations))
    stats = duration_stats(durations)
    lines = [
        f"run_label: {run_label}",
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"model_dir: {model_dir}",
        f"device: {device}",
        f"precision: {precision}",
        f"chunks_dir: {chunks_dir}",
        f"metrics_csv: {csv_path}",
        f"num_chunks: {len(times)}",
        f"total_audio_sec: {total_duration:.4f}",
        f"min_duration_sec: {stats['min']:.4f}",
        f"mean_duration_sec: {stats['mean']:.4f}",
        f"median_duration_sec: {stats['median']:.4f}",
        f"max_duration_sec: {stats['max']:.4f}",
        f"stdev_duration_sec: {stats['stdev']:.4f}" if len(durations) > 1 else "stdev_duration_sec: n/a",
        f"p90_duration_sec: {stats['p90']:.4f}",
        f"p95_duration_sec: {stats['p95']:.4f}",
        f"total_time_taken_sec: {total_time:.4f}",
        f"aggregate_rtf: {total_time / total_duration:.4f}" if total_duration > 0 else "aggregate_rtf: n/a",
        f"mean_time_taken_sec: {times_array.mean():.4f}",
        f"median_time_taken_sec: {float(np.median(times_array)):.4f}",
        f"min_time_taken_sec: {times_array.min():.4f}",
        f"max_time_taken_sec: {times_array.max():.4f}",
        f"p90_time_taken_sec: {float(np.percentile(times_array, 90)):.4f}",
        f"p95_time_taken_sec: {float(np.percentile(times_array, 95)):.4f}",
        f"stdev_time_taken_sec: {statistics.stdev(times):.4f}" if len(times) > 1 else "stdev_time_taken_sec: n/a",
        f"concatenated_raw_wer: {raw_wer:.4f}",
        f"concatenated_normalized_wer: {normalized_wer:.4f}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    load_env()
    args = parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    model_id = get_env("QWEN_MODEL_ID", DEFAULT_MODEL_ID)
    model_precision = normalize_precision(get_env("MODEL_PRECISION", DEFAULT_PRECISION))
    device = get_env("DEVICE", DEFAULT_DEVICE)
    max_new_tokens = get_env_int("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)
    model_dir = resolve_model_dir(model_id, model_precision)

    chunks_dir = resolve_path(args.chunks_dir)
    if not chunks_dir.is_dir():
        raise NotADirectoryError(f"Chunks directory not found: {chunks_dir}")

    chunk_paths = sorted(p for p in chunks_dir.glob(args.pattern) if p.is_file())
    if not chunk_paths:
        raise FileNotFoundError(f"No files matching {args.pattern!r} in {chunks_dir}")
    if args.limit is not None:
        chunk_paths = chunk_paths[: args.limit]

    durations = [audio_duration_sec(path) for path in chunk_paths]
    stats = duration_stats(durations)

    run_label = args.run_label.strip() or (
        f"chunks_{model_precision}_{device.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_label}_chunks.csv"
    summary_path = output_dir / f"{run_label}_chunks_summary.txt"

    ensure_model(model_id, model_dir, model_precision)

    ov_model = OVQwen3ASRModel.from_pretrained(
        model_dir=str(model_dir),
        device=device,
        max_inference_batch_size=-1,
        max_new_tokens=max_new_tokens,
    )

    print(f"Model precision: {model_precision}")
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")
    print(f"Chunks directory: {chunks_dir} ({len(chunk_paths)} file(s))")
    print(
        f"Chunk durations: min={stats['min']:.2f}s  mean={stats['mean']:.2f}s  "
        f"median={stats['median']:.2f}s  max={stats['max']:.2f}s"
    )
    print(f"Run label: {run_label}")

    for warmup_index in range(args.warmup):
        print(f"Warmup {warmup_index + 1}/{args.warmup} on {chunk_paths[0].name}...")
        ov_model.transcribe(audio=str(chunk_paths[0]), language=None, 
        # context=REFERENCE
        )

    times: List[float] = []
    predictions: List[str] = []

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for index, chunk_path in enumerate(chunk_paths, start=1):
            duration = durations[index - 1]

            start = time.perf_counter()
            results = ov_model.transcribe(
                audio=str(chunk_path),
                language=None,
                # context=REFERENCE,
            )
            end = time.perf_counter()

            time_taken = end - start
            prediction = results[0].text
            rtf = time_taken / duration if duration > 0 else float("nan")

            times.append(time_taken)
            if prediction.strip():
                predictions.append(prediction.strip())

            writer.writerow(
                {
                    "run_label": run_label,
                    "chunk_index": index,
                    "chunk_name": chunk_path.name,
                    "chunk_path": str(chunk_path),
                    "duration_sec": f"{duration:.6f}",
                    "time_taken_sec": f"{time_taken:.6f}",
                    "rtf": f"{rtf:.6f}",
                    "dur_min_sec": f"{stats['min']:.6f}",
                    "dur_mean_sec": f"{stats['mean']:.6f}",
                    "dur_median_sec": f"{stats['median']:.6f}",
                    "dur_max_sec": f"{stats['max']:.6f}",
                    "dur_stdev_sec": f"{stats['stdev']:.6f}",
                    "dur_p90_sec": f"{stats['p90']:.6f}",
                    "dur_p95_sec": f"{stats['p95']:.6f}",
                    "detected_language": results[0].language,
                    "num_words": len(prediction.split()),
                    "num_chars": len(prediction),
                    "prediction": prediction,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "device": device,
                    "precision": model_precision,
                    "model_dir": str(model_dir),
                }
            )
            csv_file.flush()

            print(
                f"[{index:>3}/{len(chunk_paths)}] {chunk_path.name}  "
                f"dur={duration:6.2f}s  t={time_taken:7.3f}s  rtf={rtf:6.3f}"
            )

    full_prediction = ", ".join(predictions)
    normalizer = BasicTextNormalizer()
    raw_wer = 100 * wer(TRUTH, full_prediction) if full_prediction else float("nan")
    normalized_wer = (
        100 * wer(normalizer(TRUTH).strip(), normalizer(full_prediction).strip())
        if full_prediction
        else float("nan")
    )

    write_summary(
        summary_path=summary_path,
        run_label=run_label,
        model_dir=model_dir,
        device=device,
        precision=model_precision,
        chunks_dir=chunks_dir,
        times=times,
        durations=durations,
        raw_wer=raw_wer,
        normalized_wer=normalized_wer,
        csv_path=csv_path,
    )

    times_array = np.asarray(times, dtype=float)
    total_time = float(times_array.sum())
    total_duration = float(sum(durations))

    print("\n=== Summary ===")
    print(f"Chunks processed: {len(times)}")
    print(f"Total audio: {total_duration:.2f}s, total time taken: {total_time:.2f}s")
    if total_duration > 0:
        print(f"Aggregate RTF: {total_time / total_duration:.4f}")
    print(
        f"duration_sec    mean={stats['mean']:.3f}  median={stats['median']:.3f}  "
        f"min={stats['min']:.3f}  max={stats['max']:.3f}  "
        f"p90={stats['p90']:.3f}  p95={stats['p95']:.3f}  stdev={stats['stdev']:.3f}"
    )
    print(
        f"time_taken_sec  mean={times_array.mean():.3f}  median={float(np.median(times_array)):.3f}  "
        f"min={times_array.min():.3f}  max={times_array.max():.3f}  "
        f"p90={float(np.percentile(times_array, 90)):.3f}  p95={float(np.percentile(times_array, 95)):.3f}"
    )
    print(f"Concatenated WER: {raw_wer:.4f}")
    print(f"Concatenated normalized WER: {normalized_wer:.4f}")
    print(f"\nCSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
