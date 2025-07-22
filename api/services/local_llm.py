from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    model: str = "llama1"  # Tên model mặc định
    prompt: str               # Bắt buộc có
    stream: bool = False      # Mặc định False
    context: Optional[list] = None  # Context cho chat liên tục
    temperature: float = 0.7  # Độ sáng tạo
    max_tokens: int = 512     # Giới hạn token
