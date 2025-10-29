from pydantic import BaseModel
import httpx
import requests
from typing import List, Dict, Union, Iterable
from api.services import *

class LanguageModelOllama:
    def __init__(self, model: str, stream: bool = False, temperature: float = 0.7,
                 host: str = "http://localhost:11434"):
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

    def _endpoint(self) -> str:
        if self.stream:
            return f"{self.host}/api/generate?stream=true"
        else:
            return f"{self.host}/api/generate"

    def prompt_process(self, sentence: str):
        prompt = f"""
    Câu hội thoại sau có phải liên quan đến nội dung cuộc họp (kế hoạch, ý kiến, đề xuất, báo cáo...) không?
    Nếu có, hãy viết lại ngắn gọn và rõ nghĩa để dùng làm truy vấn tìm tài liệu, 
    đồng thời chuẩn hóa lại từ ngữ, nếu là số viết dưới dạng chữ hay tên riêng nước ngoài nhưng bị phiên âm sang tiếng Việt thì viết chuẩn lại. 
    Nếu không liên quan hoặc là câu nói linh tinh thì trả về "None" và không giải thích gì thêm.

    Câu: "{sentence}"
    """
        return prompt

    def normalize_text(self, sentence: str):
        prompt = NORMALIZE_PROMPT.format(text=sentence)
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
            response = client.post(
                self._endpoint(),
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
