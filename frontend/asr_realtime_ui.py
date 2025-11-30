import os
import sys
import threading
from typing import Optional
from queue import Empty
import time
import warnings

import gradio as gr
import onnxruntime as ort
from punctuators.models import PunctCapSegModelONNX

from api.services.chunkformer_stt import ChunkFormer

# Try to import config constants; provide safe defaults if missing
try:
    from api.private_config import CHUNKFORMER_CHECKPOINT, ITN_REPO
except Exception:
    CHUNKFORMER_CHECKPOINT = None
    ITN_REPO = None

warnings.filterwarnings("ignore")


def init_itn_model(itn_model_dir: str):
    if not itn_model_dir:
        return None, None

    far_dir = os.path.join(itn_model_dir, "far")
    classifier_far = os.path.join(far_dir, "classify/tokenize_and_classify.far")
    verbalizer_far = os.path.join(far_dir, "verbalize/verbalize.far")

    if not (os.path.exists(classifier_far) and os.path.exists(verbalizer_far)):
        print(f"ITN .far files not found in {far_dir}, skipping ITN.")
        return None, None

    try:
        import pynini

        reader_classifier = pynini.Far(classifier_far)
        reader_verbalizer = pynini.Far(verbalizer_far)
        classifier = reader_classifier.get_fst()
        verbalizer = reader_verbalizer.get_fst()
        print("ITN model ready.")
        return classifier, verbalizer
    except Exception as e:
        print(f"Error loading ITN model: {e}", file=sys.stderr)
        return None, None


# Initialize ASR / punctuation / ITN components
chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)
punc_model = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    ort_providers=["CPUExecutionProvider"],
)
itn_classifier, itn_verbalizer = init_itn_model(ITN_REPO)


# Global state for UI
asr_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
transcript_lock = threading.Lock()
transcript_text = ""


def on_update(event: str, payload: dict):
    """Callback from the chunkformer ASR realtime pipeline.
    Only updates the `transcript_text` (display / commit / final flush).
    """
    global transcript_text

    with transcript_lock:
        if event == "partial":
            display = (payload.get("display") or "").strip()
            if display:
                transcript_text = display

        elif event == "commit":
            display = (payload.get("display") or payload.get("committed") or "").strip()
            if display:
                transcript_text = display

        elif event == "final_flush":
            text = (payload.get("text") or "").strip()
            if text:
                transcript_text = text


def asr_worker():
    """Run chunkformer realtime ASR (with punctuation + ITN if available).
    This function blocks until `stop_event` is set by `stop_asr`.
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
            punc_window_words=100,
            punc_commit_margin_words=50,
            itn_classifier=itn_classifier,
            itn_verbalizer=itn_verbalizer,
            # Control
            on_update=on_update,
            stop_event=stop_event,
            return_final=False,
        )
    except Exception as e:
        print("[ASR] Error:", e, file=sys.stderr)


def start_asr():
    """Start ASR: clear transcript, clear stop_event, spawn worker thread."""
    global asr_thread, transcript_text
    with transcript_lock:
        transcript_text = ""

    stop_event.clear()

    if asr_thread is None or not asr_thread.is_alive():
        t = threading.Thread(target=asr_worker, daemon=True)
        t.start()
        asr_thread = t
        return gr.update(value=""), "Đang lắng nghe 🎧"
    else:
        return gr.update(), "ASR đã chạy rồi ✅"


def stop_asr():
    stop_event.set()
    return "Đã gửi tín hiệu dừng ⏹️"


def poll_ui():
    with transcript_lock:
        txt = transcript_text
    return gr.update(value=txt)


with gr.Blocks() as demo:
    gr.Markdown("### 📝 ASR Realtime (ASR-only UI)")

    with gr.Row():
        with gr.Column(scale=1):
            start_btn = gr.Button("▶️ Bắt đầu ghi âm", variant="primary")
        with gr.Column(scale=1):
            stop_btn = gr.Button("⏹️ Dừng", variant="secondary")
        with gr.Column(scale=1):
            status = gr.Markdown("Đang idle…")

    with gr.Row(elem_classes=["main-row"]):
        with gr.Column(scale=1, elem_classes=["main-col"]):
            gr.Markdown("**Transcript (Realtime ASR)**")
            transcript_box = gr.Textbox(
                show_label=False,
                placeholder="Transcript sẽ hiển thị ở đây...",
                lines=30,
                interactive=False,
                elem_classes=["full-height-box"],
            )

    start_btn.click(fn=start_asr, outputs=[transcript_box, status])
    stop_btn.click(fn=stop_asr, outputs=[status])

    timer = gr.Timer(value=0.25, active=True)
    timer.tick(fn=poll_ui, outputs=[transcript_box])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
