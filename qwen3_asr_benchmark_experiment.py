from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from jiwer import wer
import librosa
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from qwen_3_asr_helper import OVQwen3ASRModel, convert_qwen3_asr_model

from nncf import CompressWeightsMode

MODEL_DIR = Path("Qwen/Qwen3-ASR-1.7B-OV")
DEVICE = "GPU"
SAMPLE_AUDIO = Path("../endoscopy_internal.wav")
RESULTS_DIR = Path("results")
PROMPT_LOG_FILE = Path("prompt_versions.txt")

TRUTH = "Insertion level, Terminal ileum, Cecum, Ascending colon, Hepatic flexure, Transverse colon, Splenic flexure, Descending colon, Sigmoid Colon, Rectum, Anastomosis, Anus, Premedication, Colon cleansing agent, Preparation time, Morning single dose, Evening single dose, Split dose, Colon cleansing level, Excellent, Good, Fail, Poor finding, A, normal, Negative finding, Negative finding in the observable segment, Poor preparation, B, Hemorrhoids, External Hemorrhoids, Mixed hemorrhoids, Internal hemorrhoids, C, polyp, Hyperplastic polyp, Tubular adenoma, Tubulovillous adenoma, Villous adenoma, Sessile serrated lesion, SSL, Traditional serrated adenoma, Post-treatment residual neoplasm, Inflammatory polyp, Juvenile polyp, Peutz-Jeghers syndrome, Colon polyposis, familiar, Colon polyposis, Early colorectal cancer, Advanced colorectal cancer, Lymphangioma, Lipoma, Carcinoid, Submucosal tumor, Colonmaltoma, Lymphoma, Colitis, Non-specific colitis, Ischemic colitis, Infectious colitis, Amebic colitis, Ulcerative colitis, Radiation colitis, Pseudo-membranous colitis, Drug induced colitis, Cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, Colonic ulcer, Bechet's disease, Proctitis, Hemorrhagic colitis, Colitis aphthosa, Colonic diverticulum, Chronic diverticulosis, Melanosis coloi, Xanthoma, Post partial colectomy, Post left hemicolectomy, Post right hemicolectomy, Situs inversus, Colonic wall cyst, Angiodysplasia, Angiectasia, Lymphoid follicles, Operation scar, Suture granuloma, Petechia, Colonic tuberculosis, Amyloidosis, Mega colon, Rectal varices, Mucosal prolapse, Intussusception, Colon fistula, Post endoscopy treatment scar, Colonic stricture, Rectosigmoid junction RSJ"

# Baseline context intentionally mirrors the current script and is only used as reference.
BASELINE_CONTEXT = TRUTH

PROMPT_VARIANTS = [
    (
        "v0_lexicon_sequence",
        "Transcribe in comma-separated clinical keyword/list style without paraphrasing. Preserve dictated order and short tokens like A, B, C, RSJ. Use this lower-GI term bank when acoustically plausible: insertion level, terminal ileum, cecum, ascending colon, hepatic flexure, transverse colon, splenic flexure, descending colon, sigmoid colon, rectum, anastomosis, anus, premedication, colon cleansing agent, preparation time, morning single dose, evening single dose, split dose, colon cleansing level, excellent, good, fail, poor finding, A, normal, negative finding, negative finding in the observable segment, poor preparation, B, hemorrhoids, external hemorrhoids, mixed hemorrhoids, internal hemorrhoids, C, polyp, hyperplastic polyp, tubular adenoma, tubulovillous adenoma, villous adenoma, sessile serrated lesion, SSL, traditional serrated adenoma, post-treatment residual neoplasm, inflammatory polyp, juvenile polyp, Peutz-Jeghers syndrome, colon polyposis, familiar, colon polyposis, early colorectal cancer, advanced colorectal cancer, lymphangioma, lipoma, carcinoid, submucosal tumor, colonmaltoma, lymphoma, colitis, non-specific colitis, ischemic colitis, infectious colitis, amebic colitis, ulcerative colitis, radiation colitis, pseudo-membranous colitis, drug induced colitis, cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, colonic ulcer, Bechet's disease, proctitis, hemorrhagic colitis, colitis aphthosa, colonic diverticulum, chronic diverticulosis, melanosis coli, xanthoma, post partial colectomy, post left hemicolectomy, post right hemicolectomy, situs inversus, colonic wall cyst, angiodysplasia, angiectasia, lymphoid follicles, operation scar, suture granuloma, petechia, colonic tuberculosis, amyloidosis, mega colon, rectal varices, mucosal prolapse, intussusception, colon fistula, post endoscopy treatment scar, colonic stricture, rectosigmoid junction, RSJ.",
        "High-coverage lexicon sequence format to preserve token-level fidelity.",
    ),
    (
        "v1_structured_core",
        "Task: transcribe lower GI endoscopy dictation literally. Preferred anatomy terms: terminal ileum, cecum, ascending colon, hepatic flexure, transverse colon, splenic flexure, descending colon, sigmoid colon, rectum, anus, anastomosis, rectosigmoid junction (RSJ). Preferred finding terms: hemorrhoids (external, internal, mixed), polyp, hyperplastic polyp, tubular adenoma, tubulovillous adenoma, villous adenoma, sessile serrated lesion (SSL), traditional serrated adenoma, inflammatory polyp, juvenile polyp, diverticulum, angiodysplasia, angiectasia, stricture.",
        "Structured core anatomy/findings vocabulary.",
    ),
    (
        "v2_inflammation_pathology",
        "Transcribe medical speech for colonoscopy reports. Prefer exact disease labels when heard: non-specific colitis, ischemic colitis, infectious colitis, amebic colitis, ulcerative colitis, radiation colitis, pseudo-membranous colitis, drug induced colitis, cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, proctitis, hemorrhagic colitis, colitis aphthosa, colonic ulcer, colonic tuberculosis, amyloidosis. Keep multi-word disease names intact.",
        "Pathology-heavy prompt with exact disease labels.",
    ),
    (
        "v3_disambiguation_rules",
        "Disambiguation rules for ASR: use ileum (not ilium), cecum (not secum), anastomosis, hemorrhoids, angiodysplasia, angiectasia, intussusception, mucosal prolapse, rectal varices, post endoscopy treatment scar. Prefer Bechet's disease, Peutz-Jeghers syndrome, tubulovillous adenoma, pseudo-membranous colitis when acoustically plausible. Do not replace clinical terms with everyday synonyms.",
        "Explicit disambiguation and anti-synonym substitutions.",
    ),
    (
        "v4_literal_list_mode",
        "Output mode: literal clinical dictation. Keep item order, preserve punctuation-like pauses as separators, and avoid adding explanations. If uncertain, keep closest medical term spelling from colonoscopy vocabulary. Common terms include melanosis coli, lymphoid follicles, operation scar, suture granuloma, petechia, mega colon, colon fistula, colonic wall cyst, situs inversus, post partial colectomy, post left hemicolectomy, post right hemicolectomy.",
        "Literal list-preserving style with additional uncommon terms.",
    ),
    (
        "v5_balanced_hybrid",
        "Transcribe lower GI endoscopy dictation faithfully using this vocabulary bank when acoustically plausible. Anatomy and procedure terms: insertion level, terminal ileum, cecum, ascending colon, hepatic flexure, transverse colon, splenic flexure, descending colon, sigmoid colon, rectum, anus, anastomosis, premedication, colon cleansing agent, preparation time, morning single dose, evening single dose, split dose, colon cleansing level, rectosigmoid junction RSJ. Findings and pathology terms: hemorrhoids (external, internal, mixed), polyp, hyperplastic polyp, tubular adenoma, tubulovillous adenoma, villous adenoma, sessile serrated lesion SSL, traditional serrated adenoma, post-treatment residual neoplasm, inflammatory polyp, juvenile polyp, Peutz-Jeghers syndrome, colon polyposis, early colorectal cancer, advanced colorectal cancer, lymphangioma, lipoma, carcinoid, submucosal tumor, lymphoma, colitis, non-specific colitis, ischemic colitis, infectious colitis, amebic colitis, ulcerative colitis, radiation colitis, pseudo-membranous colitis, drug induced colitis, cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, colonic ulcer, Bechet's disease, proctitis, hemorrhagic colitis, colitis aphthosa, colonic diverticulum, chronic diverticulosis, melanosis coli, xanthoma, situs inversus, colonic wall cyst, angiodysplasia, angiectasia, lymphoid follicles, operation scar, suture granuloma, petechia, colonic tuberculosis, amyloidosis, mega colon, rectal varices, mucosal prolapse, intussusception, colon fistula, post endoscopy treatment scar, colonic stricture. Keep list order if dictated and avoid paraphrasing.",
        "Broad ontology-style term bank for robust domain boosting.",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline + iterative context-prompt experiments for Qwen3-ASR."
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--sample-audio", type=Path, default=SAMPLE_AUDIO)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--prompt-log", type=Path, default=PROMPT_LOG_FILE)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--target-delta", type=float, default=5.0)
    parser.add_argument("--run-label", default=f"prompt_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="Run all iterations even if target delta is met.",
    )
    return parser.parse_args()


def ensure_model(model_dir: Path) -> None:
    if model_dir.exists():
        return

    model_name = "Qwen/Qwen3-ASR-1.7B"
    ov_model_dir = Path(f"{model_name}-OV")
    convert_qwen3_asr_model(
        model_id=model_name,
        output_dir=ov_model_dir,
        quantization_config={"mode": CompressWeightsMode.INT8_SYM},
    )


def append_prompt_version(
    prompt_log: Path,
    run_label: str,
    version_id: str,
    rationale: str,
    prompt_text: str,
    baseline_only: bool,
) -> None:
    prompt_log.parent.mkdir(parents=True, exist_ok=True)
    mode_label = "baseline_reference" if baseline_only else "candidate"
    with prompt_log.open("a", encoding="utf-8") as handle:
        handle.write(f"timestamp_utc: {utc_now()}\n")
        handle.write(f"run_label: {run_label}\n")
        handle.write(f"version: {version_id}\n")
        handle.write(f"type: {mode_label}\n")
        handle.write(f"rationale: {rationale}\n")
        handle.write("prompt:\n")
        handle.write(prompt_text.strip() + "\n")
        handle.write("-" * 80 + "\n")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def append_candidate_default(
    prompt_log: Path,
    run_label: str,
    version_id: str,
    normalized_wer: float,
    delta_vs_baseline: float,
) -> None:
    with prompt_log.open("a", encoding="utf-8") as handle:
        handle.write(f"timestamp_utc: {utc_now()}\n")
        handle.write(f"run_label: {run_label}\n")
        handle.write("version: candidate_default\n")
        handle.write("type: selection\n")
        handle.write(
            "rationale: selected lowest normalized WER among generic candidate prompts.\n"
        )
        handle.write(f"selected_version: {version_id}\n")
        handle.write(f"selected_normalized_wer: {normalized_wer:.4f}\n")
        handle.write(f"selected_delta_vs_baseline: {delta_vs_baseline:+.4f}\n")
        handle.write("-" * 80 + "\n")


def run_single_eval(
    ov_model: OVQwen3ASRModel,
    audio_path: Path,
    truth_text: str,
    context_text: str,
    context_version: str,
    normalizer: BasicTextNormalizer,
    audio_duration: float,
) -> dict:
    start = time.perf_counter()
    results = ov_model.transcribe(
        audio=str(audio_path),
        language=None,
        context=context_text,
    )
    end = time.perf_counter()

    time_taken = end - start
    prediction = results[0].text
    normalized_truth = normalizer(truth_text).strip()
    normalized_prediction = normalizer(prediction).strip()

    raw_wer = 100 * wer(truth_text, prediction)
    normalized_wer = 100 * wer(normalized_truth, normalized_prediction)
    rtf = time_taken / audio_duration if audio_duration else float("inf")

    return {
        "timestamp_utc": utc_now(),
        "audio_path": str(audio_path),
        "context_version": context_version,
        "prediction": prediction,
        "detected_language": results[0].language,
        "raw_wer": raw_wer,
        "normalized_wer": normalized_wer,
        "time_taken_sec": time_taken,
        "rtf": rtf,
    }


def write_summary(
    summary_path: Path,
    run_label: str,
    baseline_norm_wer: float,
    best_result: dict,
    target_delta: float,
    executed_iterations: int,
) -> None:
    best_delta = best_result["normalized_wer"] - baseline_norm_wer
    lines = [
        f"run_label: {run_label}",
        f"baseline_normalized_wer: {baseline_norm_wer:.4f}",
        f"best_version: {best_result['context_version']}",
        f"best_normalized_wer: {best_result['normalized_wer']:.4f}",
        f"delta_vs_baseline: {best_delta:+.4f}",
        f"target_delta: +{target_delta:.4f}",
        f"met_target: {best_delta <= target_delta}",
        f"executed_iterations: {executed_iterations}",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_audio = args.sample_audio

    if not sample_audio.exists():
        raise FileNotFoundError(f"Audio file not found: {sample_audio}")

    ensure_model(args.model_dir)

    ov_model = OVQwen3ASRModel.from_pretrained(
        model_dir=str(args.model_dir),
        device=args.device,
        max_inference_batch_size=-1,
        max_new_tokens=args.max_new_tokens,
    )

    normalizer = BasicTextNormalizer()
    audio_duration = librosa.get_duration(path=sample_audio)
    metrics_path = args.results_dir / f"{args.run_label}_metrics.jsonl"
    summary_path = args.results_dir / f"{args.run_label}_summary.txt"

    append_prompt_version(
        prompt_log=args.prompt_log,
        run_label=args.run_label,
        version_id="baseline_truth_context",
        rationale="Reference-only upper bound using transcript-leaking context.",
        prompt_text=BASELINE_CONTEXT,
        baseline_only=True,
    )
    baseline = run_single_eval(
        ov_model=ov_model,
        audio_path=sample_audio,
        truth_text=TRUTH,
        context_text=BASELINE_CONTEXT,
        context_version="baseline_truth_context",
        normalizer=normalizer,
        audio_duration=audio_duration,
    )
    append_jsonl(metrics_path, {"run_label": args.run_label, **baseline})

    print(f"Baseline normalized WER: {baseline['normalized_wer']:.4f}")
    print(f"Baseline RTF: {baseline['rtf']:.4f}")

    best_result: dict | None = None
    executed_iterations = 0
    variant_limit = min(args.max_iterations, len(PROMPT_VARIANTS))

    for idx, (version_id, prompt_text, rationale) in enumerate(
        PROMPT_VARIANTS[:variant_limit], start=1
    ):
        append_prompt_version(
            prompt_log=args.prompt_log,
            run_label=args.run_label,
            version_id=version_id,
            rationale=rationale,
            prompt_text=prompt_text,
            baseline_only=False,
        )

        result = run_single_eval(
            ov_model=ov_model,
            audio_path=sample_audio,
            truth_text=TRUTH,
            context_text=prompt_text,
            context_version=version_id,
            normalizer=normalizer,
            audio_duration=audio_duration,
        )

        delta = result["normalized_wer"] - baseline["normalized_wer"]
        payload = {
            "run_label": args.run_label,
            "iteration": idx,
            "delta_vs_baseline": delta,
            **result,
        }
        append_jsonl(metrics_path, payload)
        executed_iterations += 1

        print(
            f"[{version_id}] normalized WER: {result['normalized_wer']:.4f} | "
            f"delta vs baseline: {delta:+.4f} | RTF: {result['rtf']:.4f}"
        )

        if best_result is None or result["normalized_wer"] < best_result["normalized_wer"]:
            best_result = result

        if delta <= args.target_delta and not args.disable_early_stop:
            print(
                f"Early stop: {version_id} reached target delta <= +{args.target_delta:.2f}."
            )
            break

    if best_result is None:
        raise RuntimeError("No prompt iterations were executed. Increase --max-iterations.")

    write_summary(
        summary_path=summary_path,
        run_label=args.run_label,
        baseline_norm_wer=baseline["normalized_wer"],
        best_result=best_result,
        target_delta=args.target_delta,
        executed_iterations=executed_iterations,
    )

    print("\nExperiment complete")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Prompt log: {args.prompt_log}")
    print(
        f"Best version: {best_result['context_version']} | "
        f"normalized WER: {best_result['normalized_wer']:.4f}"
    )
    append_candidate_default(
        prompt_log=args.prompt_log,
        run_label=args.run_label,
        version_id=best_result["context_version"],
        normalized_wer=best_result["normalized_wer"],
        delta_vs_baseline=best_result["normalized_wer"] - baseline["normalized_wer"],
    )


if __name__ == "__main__":
    main()
