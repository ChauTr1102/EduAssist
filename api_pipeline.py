from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from EduAssist.phobert_ollama_text_summarization import VietnameseSummarizationPipeline
import logging
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Request and Response models
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

# Initialize the summarization pipeline
logger.info("Initializing Vietnamese Summarization Pipeline...")
try:
    pipeline = VietnameseSummarizationPipeline()
    logger.info("Pipeline initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None

@router.get("/", response_model=APIInfo)
async def home():
    """
    API information endpoint with usage documentation.
    """
    return APIInfo(
        service="Vietnamese Text Summarization API",
        version="1.0.0",
        status="running" if pipeline else "error",
        timestamp=datetime.now().isoformat(),
        endpoint={
            "path": "/summarize",
            "method": "POST",
            "description": "Summarize Vietnamese text using the complete pipeline"
        },
        usage={
            "request_format": {
                "text": "Vietnamese text to summarize (min 10 characters)",
                "summary_length": "Desired summary length in words (default: 50, range: 10-500)"
            },
            "response_format": {
                "success": "boolean",
                "summary": "Vietnamese summary text",
                "processing_time": "time in seconds",
                "timestamp": "ISO timestamp"
            }
        }
    )

@router.post("/summarize", response_model=SummarizeResponse, responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
async def summarize(request: SummarizeRequest):
    """
    Summarize Vietnamese text using the complete pipeline.
    
    **Pipeline Flow:**
    1. Vietnamese text → English translation
    2. English text → English summary (using Ollama LLM)  
    3. English summary → Vietnamese summary
    
    **Parameters:**
    - **text**: Vietnamese text to summarize (minimum 10 characters)
    - **summary_length**: Desired length of summary in words (10-500, default: 50)
    
    **Returns:**
    - **success**: Whether the operation was successful
    - **summary**: The Vietnamese summary
    - **processing_time**: Time taken to process in seconds
    - **timestamp**: ISO timestamp of completion
    """
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "Pipeline not initialized",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        start_time = time.time()
        
        # Process the text through the complete pipeline
        logger.info("Processing Vietnamese text through summarization pipeline...")
        results = pipeline.process(request.text.strip(), request.summary_length)
        
        processing_time = time.time() - start_time
        
        # Return only the Vietnamese summary
        response = SummarizeResponse(
            success=True,
            summary=results['vietnamese_summary'],
            processing_time=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Summarization completed successfully in {processing_time:.2f} seconds")
        return response

    except Exception as e:
        logger.error(f"Summarization pipeline failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
