import os
import sys
import threading
from typing import Optional
import asyncio
from queue import Queue, Empty
import time
import warnings

import gradio as gr
import pynini
import onnxruntime as ort
from punctuators.models import PunctCapSegModelONNX

from api.services.chunkformer_stt import ChunkFormer
from api.private_config import *
from api.config import *          # để lấy SUMMARIZE_DOCUMENT_PROMPT, v.v.
from api.services.vcdb_faiss import VectorStore
from api.services.local_llm import LanguageModelOllama  # bản đã có async_generate

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
punc_model = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    ort_providers=["CPUExecutionProvider"],
)
itn_classifier, itn_verbalizer = init_itn_model(ITN_REPO)

# ---- RAG / LLM từ luồng cũ ----
llm = LanguageModelOllama("shmily_006/Qw3:4b_4bit", temperature=0.5)
faiss = VectorStore("luat_hon_nhan_gia_dinh")

# =========================
# RAG QUEUES & GLOBALS (luồng cũ)
# =========================
job_queue = Queue(maxsize=0)
summarizer_queue = Queue(maxsize=0)

# =========================
# ASYNC EVENT LOOP THREAD (luồng cũ)
# =========================
_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_ASYNC_THREAD: threading.Thread | None = None

def _loop_worker(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

def start_async_loop():
    global _ASYNC_LOOP, _ASYNC_THREAD
    if _ASYNC_LOOP is None:
        _ASYNC_LOOP = asyncio.new_event_loop()
        _ASYNC_THREAD = threading.Thread(target=_loop_worker, args=(_ASYNC_LOOP,), daemon=True)
        _ASYNC_THREAD.start()

def stop_async_loop():
    global _ASYNC_LOOP
    if _ASYNC_LOOP and _ASYNC_LOOP.is_running():
        _ASYNC_LOOP.call_soon_threadsafe(_ASYNC_LOOP.stop)

def run_async(coro, timeout: float | None = None):
    """
    Submit coroutine to the background loop from any thread and wait for result.
    """
    fut = asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
    return fut.result(timeout=timeout)

# Khởi động event loop nền NGAY từ đầu
start_async_loop()

# =========================
# TÓM TẮT: prompt builder (luồng cũ)
# =========================
def build_summary_prompt(utterance: str, docs) -> str:
    return SUMMARIZE_DOCUMENT_PROMPT.format(utterance=utterance, related_docs=docs)

# =========================
# GLOBAL STATE CHO UI MỚI
# =========================
asr_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
transcript_lock = threading.Lock()

# formatted transcript (Punc + ITN)
transcript_text = ""   # cột trái: script cuộc họp
commit_log = ""        # log new_commit (backend dùng nếu cần)

# summary từ RAG (cột phải)
summary_lock = threading.Lock()
summary_text = ""      # cột phải: summarize docs / tóm tắt

# buffer để gom new_commit trước khi đẩy sang RAG
rag_buffer = []
rag_buffer_lock = threading.Lock()

# =========================
# WORKER CHÍNH CHO RAG (luồng cũ)
# =========================
def worker_loop(worker_id: int):
    print(f"[Worker-{worker_id}] Starting")
    while True:
        try:
            text = job_queue.get(timeout=1.0)
        except Empty:
            continue

        try:
            # 1) Chuẩn hoá bằng async_generate (non-stream) chạy trên loop nền
            normalize_prompt = llm.normalize_text(text)
            normalized = run_async(llm.async_generate(normalize_prompt), timeout=60.0)
            print(f"[Worker-{worker_id}] Câu đã được chuẩn hóa và tối ưu:", text, "\n", normalized)
            print("___________________________________________________________________________________________________________")

            if not normalized or normalized.strip().casefold() == "none":
                continue
            else:
                # 2) Retrieve tài liệu liên quan
                related_docs = faiss.hybrid_search(normalized)

                # 3) Đẩy sang summarizer_queue để tóm tắt song song
                summarizer_queue.put({
                    "utterance": normalized,
                    "related_docs": related_docs,
                    "ts": time.time()
                })

        except Exception as e:
            print(f"[Worker-{worker_id}] ERROR processing job: {e}")
        finally:
            job_queue.task_done()

def start_workers(num_workers: int = 2):
    for i in range(num_workers):
        t = threading.Thread(target=worker_loop, args=(i+1,), daemon=True)
        t.start()

# =========================
# SUMMARIZER LOOP (song song, luồng cũ)
# =========================
def summarizer_loop():
    global summary_text
    print("[Summarizer] Starting")
    while True:
        try:
            item = summarizer_queue.get(timeout=1.0)
        except Empty:
            continue

        try:
            utter = item.get("utterance", "")
            docs = item.get("related_docs", [])

            # Build prompt và gọi async_generate trên loop nền
            sum_prompt = build_summary_prompt(utter, docs)
            summary = run_async(llm.async_generate(sum_prompt), timeout=60.0)

            # Cập nhật vùng summary cho UI (cột phải)
            with summary_lock:
                if summary_text:
                    summary_text = (
                        summary_text
                        + "\n\n────────  NEW SUMMARY  ────────\n"
                        + summary.strip()
                    )
                else:
                    summary_text = summary.strip()

            # (tùy chọn) log ra console
            print("\n================= [SUMMARY] =================")
            print(summary.strip())
            print("=============================================\n")

        except Exception as e:
            print(f"[Summarizer] ERROR: {e}")
        finally:
            try:
                summarizer_queue.task_done()
            except Exception:
                pass

def start_summarizer():
    t = threading.Thread(target=summarizer_loop, daemon=True)
    t.start()

# Khởi động workers & summarizer
start_workers(num_workers=2)
start_summarizer()


# =========================
# CALLBACK FROM CHUNKFORMER (luồng mới + RAG)
# =========================
def enqueue_rag_with_overlap(new_commit: str):
    """
    Gom 2 new_commit from asr lại với nhau (cửa sổ trượt size=2, overlap=1),
    sau đó mới đẩy vào job_queue.

    Ví dụ:
      commits: A, B, C, D
      gửi vào RAG: (A+B), (B+C), (C+D)
    """
    global rag_buffer

    new_commit = (new_commit or "").strip()
    if not new_commit:
        return

    with rag_buffer_lock:
        # Thêm commit mới vào buffer
        rag_buffer.append(new_commit)

        # Chưa đủ 2 câu thì chưa gửi vào RAG
        if len(rag_buffer) < 2:
            return

        # Đủ 2 câu: gom lại thành 1 đoạn
        combined = rag_buffer[0] + " " + rag_buffer[1]

        try:
            job_queue.put_nowait(combined)
        except Exception as e:
            print(f"[enqueue_rag_with_overlap] Cannot enqueue combined text: {e}")

        # Giữ overlap = 1: giữ lại câu cuối cùng để ghép với commit tiếp theo
        rag_buffer = [rag_buffer[-1]]


def on_update(event: str, payload: dict):
    """
    Callback từ chunkformer_asr_realtime_punc_norm:

      - event = "partial":
            payload: {"display", "committed", "active"}
      - event = "commit":
            payload: {"new_commit", "committed", "display"}
      - event = "final_flush":
            payload: {"text"}
    """
    global transcript_text, commit_log

    with transcript_lock:
        if event == "partial":
            display = (payload.get("display") or "").strip()
            if display:
                transcript_text = display

        elif event == "commit":
            # update transcript (ưu tiên display nếu có, fallback committed)
            display = (payload.get("display")
                       or payload.get("committed")
                       or "").strip()
            if display:
                transcript_text = display
            # Lấy new_commit để log (backend) + gom vào chunk 2-câu trước khi đẩy sang RAG
            new_commit = (payload.get("new_commit") or "").strip()
            if new_commit:
                if commit_log:
                    commit_log_val = f"{commit_log}\n{new_commit}"
                else:
                    commit_log_val = new_commit
                commit_log = commit_log_val

                enqueue_rag_with_overlap(new_commit)

        elif event == "final_flush":
            text = (payload.get("text") or "").strip()
            if text:
                transcript_text = text
                # try:
                #     job_queue.put_nowait(text)
                # except Exception as e:
                #     print(f"[on_update] Cannot enqueue final text to job_queue: {e}")


# =========================
# ASR WORKER (luồng mới)
# =========================
def asr_worker():
    """
    Chạy trên server, đọc mic local qua chunkformer_asr_realtime_punc_norm
    (đã tích hợp VAD + Punc + ITN).
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


# =========================
# GRADIO CALLBACKS
# =========================
def start_asr():
    """
    Start button:
      - reset transcript + summary
      - clear stop_event
      - spawn asr_worker thread nếu chưa chạy
    """
    global asr_thread, transcript_text, commit_log, summary_text
    with transcript_lock:
        transcript_text = ""
        commit_log = ""
    with summary_lock:
        summary_text = ""

    stop_event.clear()

    if asr_thread is None or not asr_thread.is_alive():
        t = threading.Thread(target=asr_worker, daemon=True)
        t.start()
        asr_thread = t
        return (
            gr.update(value=""),  # transcript_box
            gr.update(value=""),  # summary_box
            "Đang lắng nghe 🎧",   # status
        )
    else:
        return (
            gr.update(),          # transcript_box
            gr.update(),          # summary_box
            "ASR đã chạy rồi ✅",
        )


def stop_asr():
    """
    Stop button: set stop_event, worker sẽ tự thoát vòng while.
    """
    stop_event.set()
    return "Đã gửi tín hiệu dừng ⏹️"


def poll_ui():
    """
    Được gọi bởi gr.Timer để cập nhật UI định kỳ.
    """
    with transcript_lock:
        txt = transcript_text
    with summary_lock:
        sumtxt = summary_text
    return gr.update(value=txt), gr.update(value=sumtxt)


# =========================
# CHATBOT HANDLER (cột giữa)
# =========================
def chat_qa(history, message):
    """
    Handler tạm thời cho chatbot.
    Backend RAG hỏi đáp sẽ thay thế logic này sau.
    """
    if not message:
        return history, ""
    response = (
        "Chức năng hỏi đáp Chatbot về cuộc họp chưa hoàn thiện.\n"
        f"Bạn vừa hỏi: {message}"
    )
    history = history + [(message, response)]
    return history, ""


# =========================
# BUILD UI (3 cột, tối giản & full-height)
# =========================

with gr.Blocks() as demo:
    gr.Markdown("### 📝 Realtime Meeting Assistant")

    # Hàng nút điều khiển
    with gr.Row():
        with gr.Column(scale=1):
            start_btn = gr.Button("▶️ Bắt đầu ghi âm", variant="primary")
        with gr.Column(scale=1):
            stop_btn = gr.Button("⏹️ Dừng", variant="secondary")
        with gr.Column(scale=1):
            status = gr.Markdown("Đang idle…")

    # Ba cột chính
    with gr.Row(elem_classes=["main-row"]):
        # Cột trái: Transcript
        with gr.Column(scale=2, elem_classes=["main-col"]):
            gr.Markdown("**Transcript cuộc họp**")
            transcript_box = gr.Textbox(
                show_label=False,
                placeholder="Transcript",
                lines=37,
                interactive=False,
                elem_classes=["full-height-box"],
            )

        # Cột giữa: Chatbot
        with gr.Column(scale=3, elem_classes=["main-col"]):
            gr.Markdown("**Chatbot hỏi đáp về cuộc họp**")
            chatbot = gr.Chatbot(
                label="",
                elem_classes=["full-height-chatbot"], resizable=True, height=680
            )
            # Ô nhập + nút gửi nhỏ nằm trong Textbox (submit_btn)
            chat_input = gr.Textbox(
                show_label=False,
                placeholder="Đặt câu hỏi về nội dung cuộc họp...",
                lines=2,
                submit_btn=True,   # nút gửi nhỏ ở trong textbox
            )

        # Cột phải: Summaries / Docs
        with gr.Column(scale=2, elem_classes=["main-col"]):
            gr.Markdown("**Tóm tắt & tài liệu liên quan**")
            summary_box = gr.Textbox(
                show_label=False,
                placeholder="Các đoạn tóm tắt từ RAG sẽ hiển thị tại đây...",
                lines=37,
                interactive=False,
                elem_classes=["full-height-box"],
            )

    # Start: reset + chạy thread ASR
    start_btn.click(
        fn=start_asr,
        outputs=[transcript_box, summary_box, status],
    )

    # Stop: set stop_event
    stop_btn.click(
        fn=stop_asr,
        outputs=[status],
    )

    # Timer: cập nhật transcript & summary
    timer = gr.Timer(value=0.25, active=True)
    timer.tick(
        fn=poll_ui,
        outputs=[transcript_box, summary_box],
    )

    # Chatbot wiring: chỉ dùng submit của Textbox
    chat_input.submit(
        fn=chat_qa,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    # stop_async_loop()
