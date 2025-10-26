import re
from collections import deque
from api.services.chunkformer_stt import ChunkFormer
from punctuators.models import PunctCapSegModelONNX
from api.private_config import *
from api.services.vcdb_faiss import VectorStore
from api.services.punctuation_processing import PunctProcessor
from utils.time_format import ms_to_hms_pad
import warnings
warnings.filterwarnings("ignore")


chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)

## Punct model dùng CPU
punct_model = PunctCapSegModelONNX.from_pretrained("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
                                                   ort_providers=["CPUExecutionProvider"])

# 2) Nơi bạn gom kết quả
results = []

# 3) Callback nhận kết quả do timer flush (im lặng lâu, tự flush)
def on_emit_from_timer(event: str, payload: dict, full_text: str):
    # event sẽ là "timeout_flush" (do mình đặt), hoặc "flush"/"final_flush" nếu bạn chủ động gọi
    results.append(full_text)
    print("[EMIT]", event, "→", payload["text"])

# 4) Tạo processor với on_emit
proc = PunctProcessor(
    model=punct_model,
    number_payload=20,     # tuỳ bạn: gom 30 payload rồi mới commit 80%
    timeout_sec=5.0,       # im lặng 5 giây thì tự flush phần còn lại
    on_emit=on_emit_from_timer
)

# 5) Callback mà ASR sẽ gọi mỗi lần có update
def on_update(event: str, payload: dict, full: str):
    """
    - event: "commit" hoặc "flush" hoặc "final_flush" (tuỳ ASR của bạn)
    - payload: {"start": ms, "end": ms, "text": "..."}
    - full: toàn bộ transcript đến thời điểm hiện tại (nếu ASR có cung cấp)
    """
    print(payload["text"])
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