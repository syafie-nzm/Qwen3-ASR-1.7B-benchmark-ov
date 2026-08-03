"""Plot the distribution of per-chunk transcription time as a binned histogram."""

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_CSV_GLOB = "*_chunks.csv"
DEFAULT_COLUMN = "time_taken_sec"
DEFAULT_BINS = 10
DEFAULT_IQR_FACTOR = 1.5


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def latest_csv(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob(DEFAULT_CSV_GLOB), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No files matching {DEFAULT_CSV_GLOB!r} in {results_dir}. Pass --csv explicitly."
        )
    return candidates[-1]


def read_column(csv_path: Path, column: str) -> np.ndarray:
    values: List[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise KeyError(
                f"Column {column!r} not found in {csv_path}. Available: {reader.fieldnames}"
            )
        for row in reader:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No numeric values found in column {column!r} of {csv_path}.")
    return np.asarray(values, dtype=float)


def print_bin_table(counts: np.ndarray, edges: np.ndarray, total: int) -> None:
    width = 40
    max_count = int(counts.max()) if counts.size else 0
    print(f"{'bin range (s)':<24}{'count':>7}{'pct':>8}  distribution")
    for index, count in enumerate(counts):
        closing = "]" if index == len(counts) - 1 else ")"
        label = f"[{edges[index]:.3f}, {edges[index + 1]:.3f}{closing}"
        pct = 100 * count / total
        bar = "#" * int(round(width * count / max_count)) if max_count else ""
        print(f"{label:<24}{int(count):>7}{pct:>7.1f}%  {bar}")


def iqr_bounds(values: np.ndarray, factor: float) -> tuple[float, float]:
    q1, q3 = np.percentile(values, [25, 75])
    spread = factor * (q3 - q1)
    return float(q1 - spread), float(q3 + spread)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV produced by qwen3_asr_benchmark_chunks.py. Defaults to the newest results/*_chunks.csv.",
    )
    parser.add_argument("--column", default=DEFAULT_COLUMN, help="Numeric column to plot.")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of equal-width bins.")
    parser.add_argument(
        "--exclude-outliers",
        action="store_true",
        help="Drop values outside the Tukey IQR fence before binning.",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=DEFAULT_IQR_FACTOR,
        help="Multiplier for the IQR fence used by --exclude-outliers.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Drop values above this threshold (seconds).",
    )
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--title", default=None, help="Custom plot title.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.bins < 1:
        print("--bins must be at least 1.", file=sys.stderr)
        return 2
    if args.iqr_factor <= 0:
        print("--iqr-factor must be positive.", file=sys.stderr)
        return 2

    try:
        csv_path = resolve_path(args.csv) if args.csv else latest_csv(resolve_path(DEFAULT_RESULTS_DIR))
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        values = read_column(csv_path, args.column)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    total_count = values.size
    dropped: List[float] = []

    if args.exclude_outliers:
        lower, upper = iqr_bounds(values, args.iqr_factor)
        keep = (values >= lower) & (values <= upper)
        dropped.extend(values[~keep].tolist())
        values = values[keep]
        print(f"IQR fence (factor {args.iqr_factor}): [{lower:.3f}, {upper:.3f}]")

    if args.max_value is not None:
        keep = values <= args.max_value
        dropped.extend(values[~keep].tolist())
        values = values[keep]

    if values.size == 0:
        print("All values were filtered out. Loosen --iqr-factor or --max-value.", file=sys.stderr)
        return 1

    filtered = args.exclude_outliers or args.max_value is not None
    if dropped:
        preview = ", ".join(f"{value:.3f}" for value in sorted(dropped, reverse=True)[:10])
        print(f"Excluded {len(dropped)}/{total_count} outlier(s): {preview}")

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts, edges = np.histogram(values, bins=args.bins)
    mean = float(values.mean())
    median = float(np.median(values))
    stdev = float(values.std(ddof=1)) if values.size > 1 else 0.0

    figure, axes = plt.subplots(figsize=(9, 5))
    axes.hist(values, bins=edges, edgecolor="black", color="#4c72b0", alpha=0.85)
    axes.axvline(mean, color="#c44e52", linestyle="--", linewidth=1.5, label=f"mean = {mean:.3f}s")
    axes.axvline(median, color="#55a868", linestyle="--", linewidth=1.5, label=f"median = {median:.3f}s")
    axes.set_xlabel("Time taken (s)")
    axes.set_ylabel("Number of chunks")
    title_suffix = ", outliers excluded" if filtered else ""
    axes.set_title(
        args.title
        or f"Distribution of {args.column} — {csv_path.stem} ({args.bins} bins{title_suffix})"
    )
    axes.legend(loc="upper right")
    axes.grid(axis="y", alpha=0.3)

    stats_text = (
        f"n = {values.size}\n"
        f"mean = {mean:.3f}s\n"
        f"median = {median:.3f}s\n"
        f"std = {stdev:.3f}s\n"
        f"min = {values.min():.3f}s\n"
        f"max = {values.max():.3f}s\n"
        f"p90 = {float(np.percentile(values, 90)):.3f}s\n"
        f"p95 = {float(np.percentile(values, 95)):.3f}s"
    )
    axes.text(
        0.985,
        0.62,
        stats_text,
        transform=axes.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    figure.tight_layout()

    name_suffix = "_no_outliers" if filtered else ""
    output_path = (
        resolve_path(args.output)
        if args.output
        else csv_path.parent / f"{csv_path.stem}_{args.column}{name_suffix}_hist.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)

    print(f"Source CSV: {csv_path}")
    print(f"Column: {args.column}  (n={values.size}, bins={args.bins})\n")
    print_bin_table(counts, edges, values.size)
    print(f"\nSaved plot: {output_path}")

    if args.show:
        plt.show()
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
