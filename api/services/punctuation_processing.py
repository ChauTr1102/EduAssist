# punctprocessing.py

import threading
import time
from typing import Optional, Callable

class PunctProcessor:
    """
    ...
    """

    def __init__(
        self,
        model,
        number_payload: int = 50,
        timeout_sec: float = 2.0,
        on_emit: Optional[Callable[[str, dict, str], None]] = None,  # NEW
    ):
        """
        Args:
            model: ...
            number_payload: ...
            timeout_sec: ...
            on_emit: hàm callback kiểu on_emit(event, payload, full_text_emitted)
                     sẽ được gọi cả khi flush vì timeout (ở luồng timer).
                     Nếu không truyền, mặc định là None.
        """
        self.model = model
        self.number_payload = number_payload
        self.timeout_sec = timeout_sec
        self.on_emit = on_emit  # NEW

        self.buffer = []
        self.unconfirmed: Optional[str] = None
        self.confirmed_sentences = []

        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def _arm_timer(self):
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.timeout_sec, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def _on_timeout(self):
        # Timeout: flush hết phần còn lại
        out = self._flush_now(commit_all=True, reason="timeout")
        # ĐẨY KẾT QUẢ RA NGOÀI Ở LUỒNG TIMER
        if out and self.on_emit is not None:
            try:
                # event = "timeout_flush" để bạn phân biệt
                self.on_emit("timeout_flush", {"text": out}, out)
            except Exception:
                # không để lỗi user-callback làm chết timer
                pass

    def _flush_now(self, commit_all: bool, reason: str = "") -> Optional[str]:
        with self._lock:
            full_buffer = " ".join(self.buffer).strip()
            parts = []
            if self.unconfirmed:
                parts.append(self.unconfirmed)
            if full_buffer:
                parts.append(full_buffer)
            if not parts:
                return None

            text = " ".join(parts)
            result = self.model.infer([text], apply_sbd=True)[0]

            if commit_all:
                confirmed = " ".join(result).strip()
                self.unconfirmed = ""
                self.buffer.clear()
            else:
                total = len(result)
                n_confirm = max(1, int(total * 0.8))
                confirmed = " ".join(result[:n_confirm]).strip()
                n_confirmed_words = len(confirmed.split())
                self.unconfirmed = " ".join(text.split()[n_confirmed_words:])
                self.buffer.clear()

            if confirmed:
                self.confirmed_sentences.append(confirmed)
                return confirmed
            return None

    def punct_process(self, event: str, payload: dict, full: str) -> Optional[str]:
        force = (event in ("flush", "final_flush"))
        text = (payload or {}).get("text", "")

        if text.strip():
            with self._lock:
                self.buffer.append(text)

        # chỉ cần arm timer khi có thêm text mới hoặc khi force flush
        if text.strip() or force:
            self._arm_timer()

        if force:
            out = self._flush_now(commit_all=True, reason=event)
            if event == "final_flush" and self._timer is not None:
                self._timer.cancel()
            # BẮN RA NGOÀI CẢ Ở LUỒNG CHÍNH (giữ nguyên hành vi cũ + callback)
            if out and self.on_emit is not None:
                try:
                    self.on_emit(event, {"text": out}, out)
                except Exception:
                    pass
            return out

        with self._lock:
            if len(self.buffer) >= self.number_payload:
                segment = " ".join(self.buffer[:self.number_payload])
                full_buffer = (self.unconfirmed + " " + segment).strip() if self.unconfirmed else segment
                result = self.model.infer([full_buffer], apply_sbd=True)[0]
                n_confirm = max(1, int(len(result) * 0.8))
                confirmed = " ".join(result[:n_confirm]).strip()
                n_words = len(confirmed.split())
                self.unconfirmed = " ".join(full_buffer.split()[n_words:])
                del self.buffer[:self.number_payload]
                self.confirmed_sentences.append(confirmed)
                # reset timer vì vừa xử lý xong một block
                self._arm_timer()

        # BẮN RA NGOÀI Ở LUỒNG CHÍNH
        if 'confirmed' in locals() and confirmed and self.on_emit is not None:
            try:
                self.on_emit("commit", {"text": confirmed}, confirmed)
            except Exception:
                pass
        return locals().get("confirmed", None)
