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
results = []
proc = PunctProcessor(punct_model, number_payload=30, timeout_sec=5.0)

def on_update_wrap(event, payload, full):
    print(payload["text"])
    res = proc.punct_process(event, payload, full)
    if res:
        results.append(res)
        print("APPEND confirmed:", res)
        print("Current results list:", results)

# after ASR ends
final_text = chunkformer.chunkformer_asr_realtime(
    mic_sr=16000,
    stream_chunk_sec=0.5,
    lookahead_sec=0.5,
    left_context_size=128,
    right_context_size=32,
    max_overlap_match=32,
    on_update=on_update_wrap,
)

print("=== DONE ASR ===")
print("All confirmed sentences:", results)
print("Full transcript:", final_text)
