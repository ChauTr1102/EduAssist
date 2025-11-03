import re
import asyncio
from collections import deque
from api.services.chunkformer_stt import ChunkFormer
from punctuators.models import PunctCapSegModelONNX
from api.private_config import *
from api.config import *
from api.services.vcdb_faiss import VectorStore
from api.services.punctuation_processing import PunctProcessor
from api.services.local_llm import LanguageModelOllama  # bản đã có async_generate

import threading
import time
from queue import Queue, Empty
from utils.time_format import ms_to_hms_pad
import warnings
warnings.filterwarnings("ignore")

# =========================
# QUEUES & GLOBALS
# =========================
job_queue = Queue(maxsize=0)
summarizer_queue = Queue(maxsize=0)

chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)
# Lưu ý: class này phải có async_generate như bạn vừa cập nhật
llm = LanguageModelOllama("shmily_006/Qw3:4b_4bit", temperature=0.5)

# qwen3:8b
# shmily_006/Qw3:4b_4bit
# hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:q4_K_M

faiss = VectorStore("Baocaouyvienbochinhtri")

## Punct model dùng CPU
punct_model = PunctCapSegModelONNX.from_pretrained("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
                                                   ort_providers=["CPUExecutionProvider"])

# Nơi gom kết quả thô từ emitter
results = []

# =========================
# ASYNC EVENT LOOP THREAD
# =========================
_ASYNC_LOOP = None
_ASYNC_THREAD = None

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
    Sử dụng run_coroutine_threadsafe -> thread-safe theo asyncio docs.
    """
    fut = asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
    return fut.result(timeout=timeout)

# Khởi động event loop nền NGAY từ đầu
start_async_loop()

# =========================
# TÓM TẮT: prompt builder
# =========================
def build_summary_prompt(utterance: str, docs) -> str:
    return SUMMARIZE_DOCUMENT_PROMPT.format(utterance=utterance, related_docs=docs)

# =========================
# EMIT CALLBACK TỪ PUNCT
# =========================
def on_emit_from_timer(event: str, payload: dict, full_text: str):
    results.append(full_text)
    print("___________________________________________________________________________________________________________")
    print("[EMIT]", event, "→", payload["text"])
    job_queue.put(full_text)

# =========================
# WORKER CHÍNH
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
            print("Câu đã được chuẩn hóa và tối ưu:", normalized)
            print("___________________________________________________________________________________________________________")

            if not normalized or normalized.strip().casefold() == "none":
                continue
            else:
                # 2) Retrieve tài liệu liên quan
                related_docs = faiss.hybrid_search(normalized)

                # 3) Đẩy sang summarizer_queue để tóm tắt song song (không chặn worker)
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
# SUMMARIZER LOOP (song song)
# =========================
def summarizer_loop():
    print("[Summarizer] Starting")
    while True:
        try:
            item = summarizer_queue.get(timeout=1.0)
        except Empty:
            continue

        try:
            utter = item.get("utterance", "")
            docs = item.get("related_docs", [])

            # Build prompt và gọi async_generate trên loop nền (non-block đối với loop, chỉ block thread này)
            sum_prompt = build_summary_prompt(utter, docs)
            summary = run_async(llm.async_generate(sum_prompt), timeout=60.0)

            # Xuất/ghi nhận bản tóm tắt (bạn có thể emit websocket hoặc lưu DB tại đây)
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

# =========================
# KHỞI ĐỘNG WORKERS & SUMMARIZER
# =========================
start_workers(num_workers=2)
start_summarizer()

# =========================
# PUNCT PROCESSOR & ASR
# =========================
proc = PunctProcessor(
    model=punct_model,
    number_payload=30,
    timeout_sec=5.0,       # im lặng 5 giây thì tự flush phần còn lại
    on_emit=on_emit_from_timer
)

def on_update(event: str, payload: dict, full: str):
    out = proc.punct_process(event, payload, full)

# after ASR ends
final_text = chunkformer.chunkformer_asr_realtime(
    mic_sr=16000,
    stream_chunk_sec=0.5,
    lookahead_sec=0.5,
    left_context_size=128,
    right_context_size=32,
    max_overlap_match=32,
    on_update=on_update,
)

print("__________________________","\n",results)

# (tuỳ chọn) khi kết thúc toàn app:
# stop_async_loop()
