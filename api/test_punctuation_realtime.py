import re
from collections import deque
from api.services.chunkformer_stt import ChunkFormer
from punctuators.models import PunctCapSegModelONNX
from api.private_config import *
from api.services.vcdb_faiss import VectorStore
from api.services.punctuation_processing import PunctProcessor
from api.services.local_llm import LanguageModelOllama

import threading
import time
from queue import Queue, Empty
from utils.time_format import ms_to_hms_pad
import warnings
warnings.filterwarnings("ignore")

job_queue = Queue(maxsize=0)

chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)
llm = LanguageModelOllama("qwen3:8b", temperature=0.5)
faiss = VectorStore("Baocaouyvienbochinhtri")

## Punct model dùng CPU
punct_model = PunctCapSegModelONNX.from_pretrained("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
                                                   ort_providers=["CPUExecutionProvider"])

# 2) Nơi bạn gom kết quả
results = []

# 3) Callback nhận kết quả do timer flush (im lặng lâu, tự flush)
def on_emit_from_timer(event: str, payload: dict, full_text: str):
    # event sẽ là "timeout_flush" (do mình đặt), hoặc "flush"/"final_flush" nếu bạn chủ động gọi
    results.append(full_text)
    print("___________________________________________________________________________________________________________")
    print("[EMIT]", event, "→", payload["text"])
    job_queue.put(full_text)

def worker_loop(worker_id: int):
    print(f"[Worker-{worker_id}] Starting")
    while True:
        try:
            text = job_queue.get(timeout=1.0)  # đợi max 1 giây rồi lại kiểm tra
        except Empty:
            # Có thể chấm dứt worker nếu muốn khi queue rỗng và có điều kiện dừng
            continue

        try:

            # 1) Gọi LLM để xử lý
            prompt = llm.normalize_text(text)
            response = llm.generate(prompt=prompt)
            print("Câu đã được chuẩn hóa và tối ưu:", response)
            print("___________________________________________________________________________________________________________")

            related_docs = faiss.hybrid_search(response)
            # print(f"Related Documents: {related_docs}")

        except Exception as e:
            print(f"[Worker-{worker_id}] ERROR processing job: {e}")
        finally:
            job_queue.task_done()

def start_workers(num_workers: int = 2):
    for i in range(num_workers):
        t = threading.Thread(target=worker_loop, args=(i+1,), daemon=True)
        t.start()

# Khởi worker trước hoặc cùng lúc ASR pipeline chạy
start_workers(num_workers=2)


# 4) Tạo processor với on_emit
proc = PunctProcessor(
    model=punct_model,
    number_payload=50,
    timeout_sec=5.0,       # im lặng 5 giây thì tự flush phần còn lại
    on_emit=on_emit_from_timer
)

# 5) Callback mà ASR sẽ gọi mỗi lần có update
def on_update(event: str, payload: dict, full: str):
    # print(payload["text"], end=" ")
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