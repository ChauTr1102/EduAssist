# punctprocessing.py (phiên bản không dùng threading)

import time
from typing import Optional
import re

class PunctProcessor:
    """
    Xử lý punctuation streaming không dùng thread:
    - Khi nhận text mới, nếu khoảng cách thời gian từ lần nhận trước >= timeout_sec
      và đang còn buffer/unconfirmed, ta flush phần cũ trước.
    """

    def __init__(self, model, number_payload: int = 50, timeout_sec: float = 2.0):
        """
        Args:
            model: có phương thức infer(list_of_str, apply_sbd=True) -> list[str]
            number_payload: số payload gom mỗi lần xử lý
            timeout_sec: nếu im lặng quá ngưỡng, phần cũ sẽ được flush ngay khi có text mới
        """
        self.model = model
        self.number_payload = number_payload
        self.timeout_sec = timeout_sec

        self.buffer = []                   # các đoạn text nhỏ nhận được
        self.unconfirmed: Optional[str] = None  # phần giữ lại chưa xác nhận
        self.confirmed_sentences = []      # các câu đã confirmed

        self._last_input_ts: Optional[float] = None  # time.monotonic() của lần nhận text gần nhất

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _flush_now(self, commit_all: bool, reason: str = "") -> Optional[str]:
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
        """
        event: "commit", "flush", "final_flush" (hoặc tuỳ bạn)
        payload: {"start": ms, "end": ms, "text": str}
        full: toàn bộ transcript hiện tại
        """
        force = (event in ("flush", "final_flush"))
        text = (payload or {}).get("text", "")
        now = self._now()

        out_from_gap = None

        # 1) Nếu có text mới và đã từng nhận trước đó, kiểm tra "khoảng lặng"
        if text.strip():
            if (self._last_input_ts is not None
                and (now - self._last_input_ts) >= self.timeout_sec
                and (self.buffer or (self.unconfirmed and self.unconfirmed.strip()))):
                # Có khoảng lặng đủ dài, flush phần cũ TRƯỚC khi thêm text mới
                out_from_gap = self._flush_now(commit_all=True, reason="gap_silence")

            # Sau khi xử lý khoảng lặng (nếu có), thêm text mới
            self.buffer.append(text)
            self._last_input_ts = now

        # 2) Nếu là flush cưỡng bức
        if force:
            out = self._flush_now(commit_all=True, reason=event)
            # Quy ước: ưu tiên trả về output do gap nếu vừa tạo ở trên, nếu không thì trả out
            return out_from_gap or out

        # 3) Nếu đủ payload thì xác nhận ~80%
        confirmed = None
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
            # cập nhật mốc thời gian xử lý gần nhất (không bắt buộc)
            self._last_input_ts = now

        # 4) Nếu có flush do khoảng lặng, trả nó trước; nếu không thì trả confirmed (nếu có)
        return out_from_gap or confirmed
