import gradio as gr
import torch
import torchaudio
import warnings
import uuid
import re
from pathlib import Path
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from pyannote.audio import Pipeline as DiarizationPipeline

warnings.filterwarnings("ignore")

################################################################################
# 1) CHUẨN BỊ THIẾT BỊ
################################################################################
device_for_asr = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("ASR Device:", device_for_asr)

device_for_summary = 0 if torch.cuda.is_available() else -1
print("Summary Device:", "GPU" if device_for_summary == 0 else "CPU")

################################################################################
# 2) PIPELINE DIARIZATION (PYANNOTE)
################################################################################
# NOTE: Điền token của bạn hoặc để trống nếu model public
huggingface_token = "hf_*"
diarization_pipeline = DiarizationPipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=huggingface_token
)
diarization_pipeline.to(device_for_asr)

################################################################################
# 3) PIPELINE ASR (PhoWhisper)
################################################################################
phowhisper = pipeline(
    task="automatic-speech-recognition",
    model="vinai/PhoWhisper-medium",
    device=device_for_asr,
    return_timestamps=False
)

################################################################################
# 4) SUMMARIZER MODEL (BARTPHO-SYLLABLE) - ZERO-SHOT
################################################################################
MODEL_NAME = "vinai/bartpho-syllable"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device_for_summary
    )
    print("Mô hình BARTpho-syllable đã được tải thành công.")
except Exception as e:
    print(f"Lỗi tải mô hình BARTpho-syllable: {e}")
    summarizer = None
    tokenizer = None

################################################################################
# 5) HÀM PHỤ: CHIA AUDIO THÀNH CHUNKS, CẮT SUBSEGMENT
################################################################################
def split_audio_to_chunks(wav, sr, chunk_length_s=30.0):
    """Cắt audio thành nhiều chunk ~30s."""
    chunks = []
    total_duration = wav.shape[1] / sr
    num_chunks = int((total_duration // chunk_length_s) + 1)
    for i in range(num_chunks):
        start_time = i * chunk_length_s
        end_time = min((i + 1) * chunk_length_s, total_duration)
        if start_time >= total_duration:
            break
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        sub_wav = wav[:, start_idx:end_idx]
        chunks.append((sub_wav, start_time, end_time))
    return chunks

def extract_subsegment(wav_chunk, sr, seg_start, seg_end):
    """Cắt subsegment (theo diarization) ra khỏi chunk audio."""
    start_sample = int(seg_start * sr)
    end_sample = int(seg_end * sr)
    return wav_chunk[:, start_sample:end_sample]

################################################################################
# 6) QUY TRÌNH XỬ LÝ AUDIO -> ASR -> KẾT QUẢ TỪNG CHUNK
################################################################################
def process_audio_stream(audio_file):
    """
    Xử lý file audio -> diarization -> ASR -> hiển thị dần.
    """
    if not audio_file:
        yield "Không có file audio."
        return

    yield f"Đã nhận file: {audio_file}"

    # 1) Đọc file
    try:
        wav, sr = torchaudio.load(audio_file)
        yield f" - SR gốc: {sr}, shape: {list(wav.shape)}"
    except Exception as e:
        yield f"Lỗi đọc file: {e}"
        return

    # 2) Convert sang mono (nếu stereo)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        yield " - Đã convert sang mono."

    # 3) Normalize âm lượng (tuỳ chọn)
    max_amp = wav.abs().max().item()
    if max_amp < 1e-9:
        yield " - Cảnh báo: Tín hiệu quá nhỏ / trống."
    else:
        scale = 0.9 / max_amp
        if scale < 1.0:
            wav = wav * scale
            yield f" - Đã normalize âm lượng. max_amp={max_amp:.5f}, scale={scale:.3f}"
        else:
            yield " - Âm lượng OK."

    # 4) Resample 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
        yield " - Đã resample 16kHz."

    # 5) Chia audio thành chunks 30s
    chunk_list = split_audio_to_chunks(wav, sr, 30.0)
    yield f" - Tổng số chunks: {len(chunk_list)}."

    partial_text = ""

    # 6) Xử lý từng chunk
    for i, (chunk_wav, chunk_start, chunk_end) in enumerate(chunk_list):
        yield f"\n[Chunk {i}] time=({chunk_start:.2f}s-{chunk_end:.2f}s)..."

        # Tạo file tạm
        tmp_wav_path = f"/tmp/chunk_{uuid.uuid4()}.wav"
        torchaudio.save(tmp_wav_path, chunk_wav, sr)

        # 6.1) Diarization
        diar_result = diarization_pipeline(tmp_wav_path)
        diar_segments = []
        for turn, _, speaker_label in diar_result.itertracks(yield_label=True):
            diar_segments.append({
                "speaker": speaker_label,
                "start": turn.start,
                "end": turn.end
            })
        diar_segments.sort(key=lambda x: x["start"])

        # 6.2) ASR từng segment
        for j, seg in enumerate(diar_segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            sub_wav = extract_subsegment(chunk_wav, sr, seg_start, seg_end)
            sub_wav_np = sub_wav.squeeze(0).numpy()

            # Gọi pipeline ASR
            asr_out = phowhisper(sub_wav_np)
            text = ""
            if isinstance(asr_out, dict):
                text = asr_out.get("text", "")
            elif isinstance(asr_out, list) and len(asr_out) > 0:
                text = asr_out[0].get("text", "")

            abs_start = chunk_start + seg_start
            abs_end   = chunk_start + seg_end
            seg_text = text.strip()

            # Thêm line kết quả
            partial_line = f"[Chunk {i}] {seg['speaker']} ({abs_start:.2f}-{abs_end:.2f}s): {seg_text}"
            partial_text += partial_line + "\n"

            # Yield tạm cho Gradio
            yield partial_text

    # 7) Kết thúc
    final_text = partial_text + "\n===== HOÀN THÀNH! ====="
    yield final_text

################################################################################
# 7) CLEAN_TRANSCRIPT - LOẠI BỎ [CHUNK], SPEAKER, HOÀN THÀNH
################################################################################
def clean_transcript(raw_text):
    """
    Loại bỏ [Chunk X], SPEAKER, HOÀN THÀNH, chỉ giữ nội dung thoại.
    """
    lines = []
    pattern = re.compile(r'^\[Chunk\s+\d+\]\s+SPEAKER_\d+\s*\(\d+\.\d+-\d+\.\d+s\):\s*(.*)$')

    for line in raw_text.split('\n'):
        line_strip = line.strip()
        if not line_strip:
            continue
        if "HOÀN THÀNH" in line_strip:
            continue

        # Bắt theo pattern
        match = pattern.match(line_strip)
        if match:
            speaker_text = match.group(1)
            lines.append(speaker_text)
        else:
            # Mặc định: bỏ luôn
            pass

    cleaned_text = " ".join(lines)
    # Loại bỏ ký tự lạ, khoảng trắng thừa
    cleaned_text = cleaned_text.encode("utf-8", errors="ignore").decode("utf-8")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

################################################################################
# 8) HÀM CHUNK TEXT TRƯỚC KHI SUMMARIZE (TRÁNH LỖI CUDA)
################################################################################
def chunk_text_by_tokens(text, tokenizer, max_tokens=512):
    """
    Cắt text (tính theo số tokens) để tránh quá dài.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for w in words:
        token_count = len(tokenizer.tokenize(w))
        # Nếu thêm word này vào mà vượt max_tokens => kết thúc chunk
        if current_len + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [w]
            current_len = token_count
        else:
            current_chunk.append(w)
            current_len += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_chunk(c):
    """
    Summarize 1 chunk văn bản bằng pipeline BartPho-syllable.
    Bật truncation=True để tránh tokenizer out-of-bound.
    """
    result = summarizer(
        c,
        max_length=200,
        min_length=30,
        truncation=True,      # cắt bớt nếu vượt model_max_length
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        length_penalty=1.0,
        temperature=0.0,
        do_sample=False,
        early_stopping=True
    )
    # Lấy field summary_text
    key = 'summary_text' if 'summary_text' in result[0] else 'generated_text'
    return result[0][key]

def summarize_big_text(text):
    """
    Tóm tắt 'text' bằng cách chia nhỏ thành nhiều chunk (max 512 tokens).
    Summarize từng chunk -> nối kết quả.
    """
    # 1) Chia nhỏ
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=512)

    # 2) Summarize từng chunk
    summaries = []
    for c in chunks:
        sub_sum = summarize_chunk(c)
        summaries.append(sub_sum.strip())

    # 3) Gộp các tóm tắt chunk => 1 đoạn
    final_summary = " ".join(summaries)
    return final_summary

################################################################################
# 9) SUMMARIZE_TEXT - GỌI CHUNKING
################################################################################
def summarize_text(full_text):
    """
    Tóm tắt zero-shot bằng BartPho-syllable, có chunking để tránh crash CUDA.
    """
    if not summarizer:
        return "Mô hình tóm tắt chưa sẵn sàng."
    if not full_text.strip():
        return "Không có nội dung để tóm tắt."

    # 1) Clean text
    cleaned_text = clean_transcript(full_text)
    if not cleaned_text:
        return "Nội dung sau khi clean trống."

    try:
        # 2) Summarize chunk-by-chunk
        final_summary = summarize_big_text(cleaned_text)
        return final_summary
    except Exception as e:
        return f"Tóm tắt lỗi: {e}"

################################################################################
# 10) TẠO GRADIO UI
################################################################################
def build_gradio_app():
    with gr.Blocks(title="ASR + Diarization + Summarization") as demo:
        gr.Markdown(
            "## PhoWhisper (ASR) + Pyannote (Diarization) + BartPho (Zero-shot Summarization)"
        )

        audio_input = gr.Audio(
            label="Upload/Record Audio",
            type="filepath",
            sources=["upload", "microphone"]
        )

        text_output_full = gr.Textbox(
            label="Kết quả ASR",
            lines=15
        )

        text_output_summary = gr.Textbox(
            label="Tóm tắt (BartPho - zero-shot) [Đã chunk]",
            lines=10
        )

        # Nút 1: Xử lý audio (ASR) => streaming output
        btn_transcribe = gr.Button("Transcribe & Diarize (Streaming)")
        btn_transcribe.click(
            fn=process_audio_stream,
            inputs=audio_input,
            outputs=text_output_full
            # stream=True  # Nếu Gradio >=3.23, có thể bật
        )

        # Nút 2: Tóm tắt zero-shot (có chunking)
        btn_summarize = gr.Button("Summarize (BartPho, chunk)")
        btn_summarize.click(
            fn=summarize_text,
            inputs=text_output_full,
            outputs=text_output_summary
        )

    return demo

if __name__ == "__main__":
    app = build_gradio_app()
    # Nếu muốn public link => share=True
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
