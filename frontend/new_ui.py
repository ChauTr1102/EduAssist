import os
import sys
import threading
from typing import Optional

import gradio as gr
import pynini
import onnxruntime as ort
from punctuators.models import PunctCapSegModelONNX

from api.services.chunkformer_stt import ChunkFormer
from api.private_config import *
import warnings
warnings.filterwarnings("ignore")


# =========================
# ITN MODEL
# =========================
def init_itn_model(itn_model_dir: str):
    print(f"Loading ITN model from: {itn_model_dir}")
    far_dir = os.path.join(itn_model_dir, "far")
    classifier_far = os.path.join(far_dir, "classify/tokenize_and_classify.far")
    verbalizer_far = os.path.join(far_dir, "verbalize/verbalize.far")

    if not (os.path.exists(classifier_far) and os.path.exists(verbalizer_far)):
        print(f"ERROR: Missing .far files in {far_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        reader_classifier = pynini.Far(classifier_far)
        reader_verbalizer = pynini.Far(verbalizer_far)
        classifier = reader_classifier.get_fst()
        verbalizer = reader_verbalizer.get_fst()
        print("ITN model ready.")
        return classifier, verbalizer
    except Exception as e:
        print(f"Error loading ITN model: {e}", file=sys.stderr)
        sys.exit(1)


# =========================
# INIT GLOBAL MODELS
# =========================
chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)
punc_model = PunctCapSegModelONNX.from_pretrained("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
                                                   ort_providers=["CPUExecutionProvider"])
itn_classifier, itn_verbalizer = init_itn_model(ITN_REPO)

# =========================
# GLOBAL STATE
# =========================
asr_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
transcript_lock = threading.Lock()
transcript_text = ""  # formatted transcript (Punc + ITN)


# =========================
# CALLBACK FROM CHUNKFORMER
# =========================
def on_update(event: str, payload: dict):
    """
    Callback từ chunkformer_asr_realtime (bản mới):
      - event = "partial":
            payload: {"display", "committed", "active"}
      - event = "commit":
            payload: {"new_commit", "committed", "display"}
      - event = "final_flush":
            payload: {"text"}
    Ta sẽ dùng:
      - partial/commit: transcript_text = display/committed mới nhất
      - final_flush: transcript_text = text
    """
    global transcript_text

    with transcript_lock:
        if event == "partial":
            display = (payload.get("display") or "").strip()
            if display:
                transcript_text = display

        elif event == "commit":
            # ưu tiên display nếu có, fallback committed
            display = (payload.get("display")
                       or payload.get("committed")
                       or "").strip()
            if display:
                transcript_text = display

        elif event == "final_flush":
            text = (payload.get("text") or "").strip()
            if text:
                transcript_text = text


# =========================
# ASR WORKER
# =========================
def asr_worker():
    """
    Chạy trên server, đọc mic local qua chunkformer_asr_realtime (đã tích hợp VAD + Punc + ITN).
    Dừng khi stop_event được set.
    """
    try:
        chunkformer.chunkformer_asr_realtime_punc_norm(
            mic_sr=16000,
            stream_chunk_sec=0.5,
            lookahead_sec=0.5,
            left_context_size=128,
            right_context_size=32,
            max_overlap_match=32,
            # VAD
            vad_threshold=0.01,
            vad_min_silence_blocks=2,
            # Punc + ITN
            punc_model=punc_model,
            itn_classifier=itn_classifier,
            itn_verbalizer=itn_verbalizer,
            # Control
            on_update=on_update,
            stop_event=stop_event,
            return_final=False,
        )
    except Exception as e:
        print("[ASR] Error:", e, file=sys.stderr)


# =========================
# GRADIO CALLBACKS
# =========================
def start_asr():
    """
    Start button:
      - reset transcript
      - clear stop_event
      - spawn asr_worker thread nếu chưa chạy
    """
    global asr_thread, transcript_text
    with transcript_lock:
        transcript_text = ""

    stop_event.clear()

    if asr_thread is None or not asr_thread.is_alive():
        t = threading.Thread(target=asr_worker, daemon=True)
        t.start()
        asr_thread = t
        return gr.update(value=""), "ASR started ✅ (listening on server mic)"
    else:
        return gr.update(), "ASR is already running"


def stop_asr():
    """
    Stop button: set stop_event, worker sẽ tự thoát vòng while.
    """
    stop_event.set()
    return "Stop signal sent ⏹️"


def poll_transcript():
    """
    Được gọi bởi gr.Timer để cập nhật UI định kỳ.
    """
    with transcript_lock:
        txt = transcript_text
    return gr.update(value=txt)


# =========================
# BUILD UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown(
        "## 🎧 Realtime ASR + Punc + ITN (Server Mic)\n"
        "- Dùng microphone trên **server** (sounddevice).\n"
        "- ChunkFormer streaming + VAD + Punctuation + Inverse Text Normalization.\n"
        "- Gradio chỉ hiển thị kết quả đã xử lý.\n"
    )

    with gr.Row():
        start_btn = gr.Button("▶️ Start", variant="primary")
        stop_btn = gr.Button("⏹️ Stop", variant="stop")

    status = gr.Markdown("Idle")
    transcript_box = gr.Textbox(
        label="Transcript (Punctuated + Normalized)",
        lines=20,
        interactive=False,
    )

    # Start: reset + chạy thread ASR
    start_btn.click(
        fn=start_asr,
        outputs=[transcript_box, status],
    )

    # Stop: set stop_event
    stop_btn.click(
        fn=stop_asr,
        outputs=[status],
    )

    # Timer: gọi poll_transcript định kỳ để sync UI
    timer = gr.Timer(value=0.25, active=True)
    timer.tick(
        fn=poll_transcript,
        outputs=transcript_box,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
