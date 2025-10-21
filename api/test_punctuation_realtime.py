import re
from collections import deque
from api.services.chunkformer_stt import ChunkFormer
from punctuators.models import PunctCapSegModelONNX
from api.private_config import *
from api.services.vcdb_faiss import VectorStore
import warnings
warnings.filterwarnings("ignore")


chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)

## Punct model dùng CPU
punct_model = PunctCapSegModelONNX.from_pretrained("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
                                                   ort_providers=["CPUExecutionProvider"])

class PunctProcessor:
    def __init__(self, model, number_payload=50):
        self.model = model
        self.number_payload = number_payload
        self.unconfirmed = None
        self.buffer = []

    def punct_process(self, event, payload, full):
        force = event in ("flush", "final_flush")
        text = (payload or {}).get("text", "")
        if not text.strip():
            if not force:
                return

        if text.strip():
            self.buffer.append(text)
            print(text, end="\n")

        if len(self.buffer) >= self.number_payload:
            x = " ".join(self.buffer[:self.number_payload])
            if self.unconfirmed:
                full_buffer = self.unconfirmed + " " + x
            else:
                full_buffer = x
            result = self.model.infer([full_buffer], apply_sbd=True)
            confirmed = " ".join(result[0][:int(len(result[0]) * 0.8)])
            n_confirmed = len(confirmed.split())
            self.unconfirmed = " ".join(full_buffer.split()[n_confirmed:])
            print(confirmed, end=" ")
            self.buffer.clear()

def just_print(event, payload, full):
    print(payload["text"], end=" ")

proc = PunctProcessor(punct_model, number_payload=10)

final_text = chunkformer.stream_mic(
    stream_chunk_sec=0.5,
    left_context_size=128, right_context_size=32,
    mic_sr=16000, lookahead_sec=0.5,
    silence_rms=0.005, silence_runs=1,
    stable_reserve_words=1,
    max_duration_sec=None,
    on_update=proc.punct_process,
)

# ## Dùng thread để chạy model punctuation song song với chunkformer dùng gpu
# import threading, queue, time
#
# class AsyncPunctProcessor:
#     def __init__(self, model, number_payload=70):
#         self.model = model
#         self.number_payload = number_payload
#         self.unconfirmed = None
#         self.buffer = []
#         self.q = queue.Queue()
#         self.out_q = queue.Queue()
#         self._stop = False
#         self.worker = threading.Thread(target=self._run, daemon=True)
#         self.worker.start()
#
#     def _run(self):
#         while not self._stop:
#             item = self.q.get()
#             if item is None: break
#             event, text = item
#             if text: self.buffer.append(text)
#             if len(self.buffer) >= self.number_payload or event in ("flush", "final_flush"):
#                 x = " ".join(self.buffer[:self.number_payload])
#                 full_buffer = (self.unconfirmed + " " + x) if self.unconfirmed else x
#                 result = self.model.infer([full_buffer], apply_sbd=True)
#                 confirmed = " ".join(result[0][:int(len(result[0])*0.8)])
#                 n_confirmed = len(confirmed.split())
#                 self.unconfirmed = " ".join(full_buffer.split()[n_confirmed:])
#                 self.buffer = []
#                 print(confirmed, end=" ")
#                 self.out_q.put(confirmed)
#
#     def punct_process(self, event, payload, full):
#         text = (payload or {}).get("text", "").strip()
#         # print(text)
#         # Đẩy nhanh vào hàng đợi, không chặn luồng stream
#         self.q.put((event, text))
#
#     def read_outputs_nonblocking(self):
#         """Gọi định kỳ để lấy kết quả đã confirmed."""
#         outs = []
#         try:
#             while True:
#                 outs.append(self.out_q.get_nowait())
#         except queue.Empty:
#             pass
#         return outs
#
#     def stop(self):
#         self._stop = True
#         self.q.put(None)
#
# # Dùng async model đa luồng gpu thì uncomment dòng này
# async_punct_model = PunctCapSegModelONNX.from_pretrained("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
#                                                          ort_providers=["CUDAExecutionProvider"])
# async_proc = AsyncPunctProcessor(async_punct_model, number_payload=50)
#
#
# final_text = chunkformer.stream_mic(
#     stream_chunk_sec=0.5,
#     left_context_size=128, right_context_size=32,
#     mic_sr=16000, lookahead_sec=0.5,
#     silence_rms=0.005, silence_runs=1,
#     stable_reserve_words=1,
#     max_duration_sec=None,
#     on_update=async_proc.punct_process,
# )
#
# ## Dùng với model đa luồng bên trên, uncomment trước khi dùng
# for s in async_proc.read_outputs_nonblocking():
#     print(s, end=" ")
# async_proc.stop()