# rag_processor.py

import threading
from typing import Optional, Callable

class RagProcessor:
    """
    Gom new_commit từ ASR thành các chunk để:
      - đưa sang job_queue (RAG + summarize)
      - đưa sang embedding_queue (FAISS embedding)

    Đồng thời có cơ chế:
      - Cửa sổ trượt (sliding window): n_commits_to_combine + overlap_m
      - Timeout: nếu im lặng > timeout_sec mà vẫn còn buffer -> tự flush chunk cuối.

    on_emit(event, payload) là callback optional để bạn debug/log nếu muốn.
    """

    def __init__(
        self,
        job_queue,
        embedding_queue,
        n_commits_to_combine: int = 2,
        overlap_m: int = 1,
        timeout_sec: float = 5.0,
        on_emit: Optional[Callable[[str, dict], None]] = None,
    ):
        assert n_commits_to_combine > 0
        assert 0 <= overlap_m < n_commits_to_combine

        self.job_queue = job_queue
        self.embedding_queue = embedding_queue
        self.n_commits_to_combine = n_commits_to_combine
        self.overlap_m = overlap_m
        self.timeout_sec = timeout_sec
        self.on_emit = on_emit

        self.buffer: list[str] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    # ===== Timer helpers =====
    def _arm_timer(self):
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.timeout_sec, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def _on_timeout(self):
        """
        Khi im lặng đủ lâu:
          - flush hết phần còn lại trong buffer thành 1 chunk (hoặc nhiều)
          - đẩy vào queue
        """
        chunks = self._flush_all_locked(reason="timeout")
        if chunks and self.on_emit is not None:
            try:
                self.on_emit("timeout_flush", {"chunks": chunks})
            except Exception:
                pass

    # ===== Core logic =====
    def _emit_chunk_locked(self, chunk):
        """
        Đẩy 1 chunk vào 2 queue. Gọi hàm này khi đã cầm _lock.
        """
        # if isinstance(chunk, dict):
        text = chunk.get("text", "").strip()
        if not text:
            return
        # else:
        #     text = (chunk or "").strip()
        #     if not text:
        #         return
        #     chunk = {"text": text}
        
        try:
            self.job_queue.put_nowait(text)  # job_queue nhận text thôi
        except Exception as e:
            print(f"[RagProcessor] Cannot enqueue chunk to job_queue: {e}")
        try:
            self.embedding_queue.put_nowait(chunk)  # embedding_queue nhận dict
        except Exception as e:
            print(f"[RagProcessor] Cannot enqueue chunk to embedding_queue: {e}")

    def _combine_buffer_items(self, items):
        """
        Gom các item (có thể là dict hoặc str) trong buffer thành 1 chunk.
        Nếu đều là dict: lấy start_time_ms từ item đầu, end_time_ms từ item cuối.
        """
        texts = []

        for item in items:
            texts.append(item.get("new_commit", ""))

        start_ms = items[0]["start_time_ms"]
        end_ms = items[-1]["end_time_ms"]
        chunk_text = " ".join(texts)

        return {
            "text": chunk_text,
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
        }


        # for item in items:
        #     if isinstance(item, dict):
        #         texts.append(item.get("text", ""))
        #         if start_ms is None and "start_time_ms" in item:
        #             start_ms = item["start_time_ms"]
        #         if "end_time_ms" in item:
        #             end_ms = item["end_time_ms"]
        #     else:
        #         texts.append(str(item))
        #
        # chunk_text = " ".join(texts)
        # if start_ms is not None or end_ms is not None:
        #     return {
        #         "text": chunk_text,
        #         "start_time_ms": start_ms,
        #         "end_time_ms": end_ms,
        #     }
        # return chunk_text


    def _flush_all_locked(self, reason: str = ""):
        """
        Gom toàn bộ buffer hiện tại thành ít nhất 1 chunk, rồi clear buffer.
        Trả về list chunks đã emit (để log/debug nếu cần).
        Buffer chứa các item (dict với timestamps).
        """
        with self._lock:
            if not self.buffer:
                return []

            chunks = []

            if len(self.buffer) <= self.n_commits_to_combine:
                # ít commit: gom hết thành 1 chunk
                chunk = self._combine_buffer_items(self.buffer)
                self._emit_chunk_locked(chunk)
                chunks.append(chunk)
            else:
                # nhiều commit: chia thành các block size = n_commits_to_combine (không overlap)
                for i in range(0, len(self.buffer), self.n_commits_to_combine):
                    chunk = self._combine_buffer_items(self.buffer[i:i + self.n_commits_to_combine])
                    self._emit_chunk_locked(chunk)
                    chunks.append(chunk)

            # clear buffer sau flush
            self.buffer.clear()

            return chunks

    def process_new_commit(self, new_commit):
        """
        Gọi hàm này mỗi khi có new_commit từ on_update(event="commit").
        Áp dụng sliding window + timer như PunctProcessor.
        """
        # Lưu full item (có timestamps nếu có)
        if isinstance(new_commit, dict):
            item_to_buffer = new_commit
        else:
            item_to_buffer = new_commit

        with self._lock:
            self.buffer.append(item_to_buffer)

            # Nếu đủ n_commit => tạo 1 chunk (có overlap) và emit ngay
            if len(self.buffer) >= self.n_commits_to_combine:
                # lấy n commit gần nhất làm chunk
                chunk = self._combine_buffer_items(self.buffer[-self.n_commits_to_combine:])
                self._emit_chunk_locked(chunk)

                # giữ overlap_m commit cuối lại, bỏ phần còn lại
                if self.overlap_m > 0:
                    self.buffer = self.buffer[-self.overlap_m:]
                else:
                    self.buffer = []

        # Mỗi lần có commit mới thì reset timer
        self._arm_timer()

    def flush_all(self, reason: str = "manual_flush"):
        """
        Gọi từ luồng chính khi muốn flush (vd: event final_flush, bấm Stop, ...).
        """
        chunks = self._flush_all_locked(reason=reason)
        if self._timer is not None:
            self._timer.cancel()
        if chunks and self.on_emit is not None:
            try:
                self.on_emit(reason, {"chunks": chunks})
            except Exception:
                pass
        return chunks
