from api.services.chunkformer_stt import ChunkFormer
from api.services.vcdb_faiss import VectorStore
from api.private_config import *
from collections import deque

chunkformer = ChunkFormer(model_checkpoint=CHUNKFORMER_CHECKPOINT)
faiss = VectorStore("hop_nextstart")

buffer = deque()

NUMBER_WORD = 200
def on_update(event, payload, full):
    """
    event: "commit" | "flush" | "final_flush"
    payload: {"start": int(ms), "end": int(ms), "text": str}
    full: toàn bộ transcript tới thời điểm hiện tại (không dùng ở đây)
    """
    print(payload["text"])
    force = event in ("flush", "final_flush")
    # Bỏ payload rỗng
    if not payload or not payload.get("text", "").strip():
        # Nếu là final_flush mà còn gì trong buffer thì vẫn nên flush ở cuối (xử lý bên dưới)
        if force:
            pass
        else:
            return

    # Gom payload vào buffer khi có text
    if payload and payload.get("text", "").strip():
        buffer.append(payload)

    # Tính tổng số từ hiện có trong buffer
    total_words = sum(len(item["text"].split()) for item in buffer)

    # Điều kiện index: đủ số từ hoặc bị buộc flush
    if force or total_words >= NUMBER_WORD:
        if buffer:
            print("------------Buffer is full, flushing!-----------")
            # Lấy thời gian đầu và cuối theo phần tử đầu/đầu cuối trong buffer
            start = buffer[0]["start"]
            end   = buffer[-1]["end"]
            # Ghép transcript từ các mảnh text
            transcript = " ".join(item["text"].strip() for item in buffer).strip()
            buffer.clear()

            if transcript:
                faiss.add_transcript(transcript, start, end)
                print("_______Added into vector store!__________")

final_text = chunkformer.stream_mic(
    stream_chunk_sec=0.5,
    left_context_size=128, right_context_size=8,
    mic_sr=16000, lookahead_sec=0.5,
    silence_rms=0.005, silence_runs=3,
    stable_reserve_words=1,
    max_duration_sec=None,
    on_update=on_update,
)