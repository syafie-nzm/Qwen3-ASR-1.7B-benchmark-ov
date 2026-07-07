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

truth = "Insertion level, Terminal ileum, Cecum, Ascending colon, Hepatic flexure, Transverse colon, Splenic flexure, Descending colon, Sigmoid Colon, Rectum, Anastomosis, Anus, Premedication, Colon cleansing agent, Preparation time, Morning single dose, Evening single dose, Split dose, Colon cleansing level, Excellent, Good, Fail, Poor finding, A, normal, Negative finding, Negative finding in the observable segment, Poor preparation, B, Hemorrhoids, External Hemorrhoids, Mixed hemorrhoids, Internal hemorrhoids, C, polyp, Hyperplastic polyp, Tubular adenoma, Tubulovillous adenoma, Villous adenoma, Sessile serrated lesion, SSL, Traditional serrated adenoma, Post-treatment residual neoplasm, Inflammatory polyp, Juvenile polyp, Peutz-Jeghers syndrome, Colon polyposis, familiar, Colon polyposis, Early colorectal cancer, Advanced colorectal cancer, Lymphangioma, Lipoma, Carcinoid, Submucosal tumor, Colonmaltoma, Lymphoma, Colitis, Non-specific colitis, Ischemic colitis, Infectious colitis, Amebic colitis, Ulcerative colitis, Radiation colitis, Pseudo-membranous colitis, Drug induced colitis, Cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, Colonic ulcer, Bechet's disease, Proctitis, Hemorrhagic colitis, Colitis aphthosa, Colonic diverticulum, Chronic diverticulosis, Melanosis coloi, Xanthoma, Post partial colectomy, Post left hemicolectomy, Post right hemicolectomy, Situs inversus, Colonic wall cyst, Angiodysplasia, Angiectasia, Lymphoid follicles, Operation scar, Suture granuloma, Petechia, Colonic tuberculosis, Amyloidosis, Mega colon, Rectal varices, Mucosal prolapse, Intussusception, Colon fistula, Post endoscopy treatment scar, Colonic stricture, Rectosigmoid junction RSJ"

# Keep the same context style used in the notebook for domain-guided transcription.
reference = "Transcribe in comma-separated clinical keyword/list style without paraphrasing. Preserve dictated order and short tokens like A, B, C, RSJ. Use this lower-GI term bank when acoustically plausible: insertion level, terminal ileum, cecum, ascending colon, hepatic flexure, transverse colon, splenic flexure, descending colon, sigmoid colon, rectum, anastomosis, anus, premedication, colon cleansing agent, preparation time, morning single dose, evening single dose, split dose, colon cleansing level, excellent, good, fail, poor finding, A, normal, negative finding, negative finding in the observable segment, poor preparation, B, hemorrhoids, external hemorrhoids, mixed hemorrhoids, internal hemorrhoids, C, polyp, hyperplastic polyp, tubular adenoma, tubulovillous adenoma, villous adenoma, sessile serrated lesion, SSL, traditional serrated adenoma, post-treatment residual neoplasm, inflammatory polyp, juvenile polyp, Peutz-Jeghers syndrome, colon polyposis, familiar, colon polyposis, early colorectal cancer, advanced colorectal cancer, lymphangioma, lipoma, carcinoid, submucosal tumor, colonmaltoma, lymphoma, colitis, non-specific colitis, ischemic colitis, infectious colitis, amebic colitis, ulcerative colitis, radiation colitis, pseudo-membranous colitis, drug induced colitis, cytomegalovirus colitis, CMV colitis, GVHD related colitis, Crohn's disease, colonic ulcer, Bechet's disease, proctitis, hemorrhagic colitis, colitis aphthosa, colonic diverticulum, chronic diverticulosis, melanosis coli, xanthoma, post partial colectomy, post left hemicolectomy, post right hemicolectomy, situs inversus, colonic wall cyst, angiodysplasia, angiectasia, lymphoid follicles, operation scar, suture granuloma, petechia, colonic tuberculosis, amyloidosis, mega colon, rectal varices, mucosal prolapse, intussusception, colon fistula, post endoscopy treatment scar, colonic stricture, rectosigmoid junction, RSJ."

if not MODEL_DIR.exists():

    model_name = "Qwen/Qwen3-ASR-1.7B"
    ov_model_dir = Path(f"{model_name}-OV")
    # Convert model to OpenVINO format
    # This will skip conversion if the model already exists
    convert_qwen3_asr_model(
        model_id=model_name,
        output_dir=ov_model_dir,
        quantization_config={"mode": CompressWeightsMode.INT8_SYM},  # Set to {"mode": CompressWeightsMode.INT8_SYM} for INT8 quantization
    )

if not SAMPLE_AUDIO.exists():
    raise FileNotFoundError(f"Audio file not found: {SAMPLE_AUDIO}")

ov_model = OVQwen3ASRModel.from_pretrained(
    model_dir=str(MODEL_DIR),
    device=DEVICE,
    max_inference_batch_size=-1,
    max_new_tokens=512,
)

start = time.perf_counter()
results = ov_model.transcribe(
    audio=str(SAMPLE_AUDIO),
    language=None,
    context=reference,
)
end = time.perf_counter()

time_taken = end - start
prediction = results[0].text

print(f"Processed {SAMPLE_AUDIO}: {prediction[:50]}...")
print(f"Detected Language: {results[0].language}")
print(f"\nTranscription: {prediction}")
print(f"Time taken: {time_taken} seconds")

normalizer = BasicTextNormalizer()

normalized_truth = normalizer(truth).strip()
normalized_prediction = normalizer(prediction).strip()

raw_wer = 100 * wer(truth, prediction)
normalized_wer = 100 * wer(normalized_truth, normalized_prediction)

print(f"\nWER: {raw_wer}")
print(f"Normalized WER: {normalized_wer}")

duration = librosa.get_duration(path=SAMPLE_AUDIO)
rtf = time_taken / duration

print(f"\nRTF: {rtf:.4f}")
