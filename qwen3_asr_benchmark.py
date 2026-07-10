import os
from pathlib import Path
import time

from jiwer import wer
import librosa
from dotenv import load_dotenv
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from qwen_3_asr_helper import OVQwen3ASRModel, convert_qwen3_asr_model

from nncf import CompressWeightsMode

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_DEVICE = "GPU"
DEFAULT_PRECISION = "int8"
DEFAULT_SAMPLE_AUDIO = "./endoscopy_internal.wav"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_OUTPUT_ROOT = "./Qwen"
INT8_DIR_SUFFIX = "-OV"
FULL_PRECISION_DIR_SUFFIX = "-OV-full-precision"


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
    raise ValueError("Unsupported MODEL_PRECISION. Use int8 or full_precision.")


def resolve_model_dir(model_id: str, model_precision: str) -> Path:
    model_dir_override = os.getenv("MODEL_DIR")
    if model_dir_override and model_dir_override.strip():
        return resolve_path(model_dir_override)

    output_root = resolve_path(get_env("MODEL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    model_name = model_id.rstrip("/").split("/")[-1]
    suffix = INT8_DIR_SUFFIX if model_precision == "int8" else FULL_PRECISION_DIR_SUFFIX
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

    print(f"Converting {model_id} to {model_dir} ({model_precision})...")
    convert_qwen3_asr_model(
        model_id=model_id,
        output_dir=model_dir,
        quantization_config=quantization_config,
    )


TRUTH = "Insertion level, Terminal ileum, Cecum, Ascending colon, Hepatic flexure, Transverse colon, Splenic flexure, Descending colon, Sigmoid Colon, Rectum, Anastomosis, Anus, Premedication, Colon cleansing agent, Preparation time, Morning single dose, Evening single dose, Split dose, Colon cleansing level, Excellent, Good, Fail, Poor finding, A, normal, Negative finding, Negative finding in the observable segment, Poor preparation, B, Hemorrhoids, External Hemorrhoids, Mixed hemorrhoids, Internal hemorrhoids, C, polyp, Hyperplastic polyp, Tubular adenoma, Tubulovillous adenoma, Villous adenoma, Sessile serrated lesion, SSL, Traditional serrated adenoma, Post-treatment residual neoplasm, Inflammatory polyp, Juvenile polyp, Peutz-Jeghers syndrome, Colon polyposis, familiar, Colon polyposis, Early colorectal cancer, Advanced colorectal cancer, Lymphangioma, Lipoma, Carcinoid, Submucosal tumor, Colonmaltoma, Lymphoma, Colitis, Non-specific colitis, Ischemic colitis, Infectious colitis, Amebic colitis, Ulcerative colitis, Radiation colitis, Pseudo-membranous colitis, Drug induced colitis, Cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, Colonic ulcer, Bechet's disease, Proctitis, Hemorrhagic colitis, Colitis aphthosa, Colonic diverticulum, Chronic diverticulosis, Melanosis coloi, Xanthoma, Post partial colectomy, Post left hemicolectomy, Post right hemicolectomy, Situs inversus, Colonic wall cyst, Angiodysplasia, Angiectasia, Lymphoid follicles, Operation scar, Suture granuloma, Petechia, Colonic tuberculosis, Amyloidosis, Mega colon, Rectal varices, Mucosal prolapse, Intussusception, Colon fistula, Post endoscopy treatment scar, Colonic stricture, Rectosigmoid junction RSJ"

# Keep the same context style used in the notebook for domain-guided transcription.
REFERENCE = "Transcribe in comma-separated clinical keyword/list style without paraphrasing. Preserve dictated order and short tokens like A, B, C, RSJ. Use this lower-GI term bank when acoustically plausible: insertion level, terminal ileum, cecum, ascending colon, hepatic flexure, transverse colon, splenic flexure, descending colon, sigmoid colon, rectum, anastomosis, anus, premedication, colon cleansing agent, preparation time, morning single dose, evening single dose, split dose, colon cleansing level, excellent, good, fail, poor finding, A, normal, negative finding, negative finding in the observable segment, poor preparation, B, hemorrhoids, external hemorrhoids, mixed hemorrhoids, internal hemorrhoids, C, polyp, hyperplastic polyp, tubular adenoma, tubulovillous adenoma, villous adenoma, sessile serrated lesion, SSL, traditional serrated adenoma, post-treatment residual neoplasm, inflammatory polyp, juvenile polyp, Peutz-Jeghers syndrome, colon polyposis, familiar, colon polyposis, early colorectal cancer, advanced colorectal cancer, lymphangioma, lipoma, carcinoid, submucosal tumor, colonmaltoma, lymphoma, colitis, non-specific colitis, ischemic colitis, infectious colitis, amebic colitis, ulcerative colitis, radiation colitis, pseudo-membranous colitis, drug induced colitis, cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, colonic ulcer, Bechet's disease, proctitis, hemorrhagic colitis, colitis aphthosa, colonic diverticulum, chronic diverticulosis, melanosis coli, xanthoma, post partial colectomy, post left hemicolectomy, post right hemicolectomy, situs inversus, colonic wall cyst, angiodysplasia, angiectasia, lymphoid follicles, operation scar, suture granuloma, petechia, colonic tuberculosis, amyloidosis, mega colon, rectal varices, mucosal prolapse, intussusception, colon fistula, post endoscopy treatment scar, colonic stricture, rectosigmoid junction, RSJ."


def main() -> None:
    load_env()

    model_id = get_env("QWEN_MODEL_ID", DEFAULT_MODEL_ID)
    model_precision = normalize_precision(get_env("MODEL_PRECISION", DEFAULT_PRECISION))
    device = get_env("DEVICE", DEFAULT_DEVICE)
    sample_audio = resolve_path(get_env("SAMPLE_AUDIO", DEFAULT_SAMPLE_AUDIO))
    max_new_tokens = get_env_int("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)
    model_dir = resolve_model_dir(model_id, model_precision)

    if not sample_audio.exists():
        raise FileNotFoundError(f"Audio file not found: {sample_audio}")

    ensure_model(model_id, model_dir, model_precision)

    ov_model = OVQwen3ASRModel.from_pretrained(
        model_dir=str(model_dir),
        device=device,
        max_inference_batch_size=-1,
        max_new_tokens=max_new_tokens,
    )

    start = time.perf_counter()
    results = ov_model.transcribe(
        audio=str(sample_audio),
        language=None,
        context=REFERENCE,
    )
    end = time.perf_counter()

    time_taken = end - start
    prediction = results[0].text

    print(f"Model precision: {model_precision}")
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")
    print(f"Processed {sample_audio}: {prediction[:50]}...")
    print(f"Detected Language: {results[0].language}")
    print(f"\nTranscription: {prediction}")
    print(f"Time taken: {time_taken} seconds")

    normalizer = BasicTextNormalizer()
    normalized_truth = normalizer(TRUTH).strip()
    normalized_prediction = normalizer(prediction).strip()

    raw_wer = 100 * wer(TRUTH, prediction)
    normalized_wer = 100 * wer(normalized_truth, normalized_prediction)

    print(f"\nWER: {raw_wer}")
    print(f"Normalized WER: {normalized_wer}")

    duration = librosa.get_duration(path=sample_audio)
    rtf = time_taken / duration

    print(f"\nRTF: {rtf:.4f}")


if __name__ == "__main__":
    main()
