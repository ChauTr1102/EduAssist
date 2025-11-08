from fastapi import UploadFile, File, Form, APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, EmailStr, validator, Field
from dotenv import load_dotenv
import tempfile
# from api.services.whisper import FasterWhisper
from api.services.chunkformer_stt import ChunkFormer
from api.services.llm import LLM
import logging
import time
from datetime import datetime
from typing import Optional
import sys
import os
import httpx
# from api.services.local_llm import ChatRequest
from api.services.local_llm import LanguageModelOllama
from api.services.video_to_audio_convert import *
import pynvml
# import bộ đo GPU
from api.utils.utils_gpu_mem import GPUMemSampler, detect_device_index

load_dotenv()

def get_gpu_memory_mb(device_index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 ** 2)  # MB


# Request and Response models for summarization
class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Vietnamese text to summarize")
    summary_length: Optional[int] = Field(50, ge=10, le=500, description="Desired summary length in words")

class SummarizeResponse(BaseModel):
    success: bool
    summary: str
    processing_time: float
    timestamp: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    timestamp: str

class APIInfo(BaseModel):
    service: str
    version: str
    status: str
    timestamp: str
    endpoint: dict
    usage: dict


class Script(BaseModel):
    script: str


class UserInput(BaseModel):
    user_input: str
    summarize_script: str
    history: str


class ChatRequest(BaseModel):
    prompt: str