from pydantic import BaseModel
import httpx
import requests
from typing import List, Dict, Union, Iterable


class LanguageModelOllama:
    def __init__(self, model: str, stream: bool = False, temperature: float = 0.7,
                 host: str = "http://localhost:11434"):
        """
        model       : tên mô hình đã pull trong Ollama, ví dụ "llama3.2"
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
    Nếu có, hãy viết lại ngắn gọn và rõ nghĩa để dùng làm truy vấn tìm tài liệu. 
    Nếu không liên quan hoặc là câu nói linh tinh thì trả về "None".

    Câu: "{sentence}"
    """
        return prompt

    async def generate(self, prompt: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(self._endpoint(),
                                         json={"model": self.model, "prompt": prompt, "stream": self.stream},
                                         timeout=60.0)
        return response.json()["response"]
