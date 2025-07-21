from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from phobert_ollama_text_summarization import VietnameseSummarizationPipeline
import logging
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import tempfile
from fastapi.responses import JSONResponse

load_dotenv()

router = APIRouter()
