from pydantic import BaseModel
import httpx
import requests
from typing import AsyncGenerator, Dict, Optional
from api.services import *

class LanguageModelOllama:
    def __init__(self, model: str, stream: bool = False, temperature: float = 0.7,
                 host: str = "http://localhost:11434", request_timeout: float = 60.0, max_retries: int = 2):
        """
        model       : tên mô hình đã pull trong Ollama
        stream      : nếu True sẽ sử dụng streaming response
        temperature : tham số nhiệt độ sinh tạo (creativity)
        host        : địa chỉ server Ollama REST API
        """
        self.model = model
        self.stream = stream
        self.temperature = temperature
        self.host = host.rstrip("/")
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        # Async client dùng lại kết nối (HTTP/1.1 keep-alive)
        self._aclient: Optional[httpx.AsyncClient] = None

    # -----------------------------------------
    # Helpers
    # -----------------------------------------
    def _endpoint(self) -> str:
        if self.stream:
            return f"{self.host}/api/generate?stream=true"
        else:
            return f"{self.host}/api/generate"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self.request_timeout)
        return self._aclient

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    # ---------------------------
    # Prompt builders
    # ---------------------------
    def prompt_process(self, sentence: str):
        prompt = f"""
    Câu hội thoại sau có phải liên quan đến nội dung cuộc họp (kế hoạch, ý kiến, đề xuất, báo cáo...) không?
    Nếu có, hãy viết lại ngắn gọn và rõ nghĩa để dùng làm truy vấn tìm tài liệu, 
    đồng thời chuẩn hóa lại từ ngữ, nếu là số viết dưới dạng chữ hay tên riêng nước ngoài nhưng bị phiên âm sang tiếng Việt thì viết chuẩn lại. 
    Nếu không liên quan hoặc là câu nói linh tinh thì trả về "None" và không giải thích gì thêm.

    Câu: "{sentence}"
    """
        return prompt

    def normalize_text(self, meeting_document_summarize: str, transcript: str):
        prompt = NORMALIZE_PROMPT.format(meeting_document_summarize=meeting_document_summarize, text=transcript)
        return prompt

    # async def generate(self, prompt: str):
    #     async with httpx.AsyncClient() as client:
    #         response = await client.post(self._endpoint(),
    #                                      json={"model": self.model, "prompt": prompt, "stream": self.stream,
    #                                            "think": False},
    #                                      timeout=60.0)
    #     return response.json()["response"]

    def generate(self, prompt: str) -> str:
        """Hàm đồng bộ: gửi yêu cầu tới server và trả về kết quả."""
        # Sử dụng sync Client
        with httpx.Client() as client:
            response = client.post(self._endpoint(),
                                    json={
                                        "model": self.model,
                                        "prompt": prompt,
                                        "stream": self.stream,
                                        "think": False,
                                        "temperature": self.temperature
                                    },
                                    timeout=60.0
                                )
            response.raise_for_status()
            data = response.json()
        return data["response"]

    async def async_generate(self, prompt: str) -> str:
        """
        Gọi /api/generate với stream=false, trả về full text một lần.
        """
        client = await self._get_client()
        payload: Dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # non-stream để trả về trọn gói
            "think": False,
            "temperature": self.temperature,
            "options": {
                "num_ctx": 8192
            }
        }

        last_exc = None
        for _ in range(self.max_retries + 1):
            try:
                resp = await client.post(self._endpoint(), json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                continue
        if last_exc:
            raise last_exc