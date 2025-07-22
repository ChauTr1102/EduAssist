import os

from api.routes import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize the summarization pipeline
logger.info("Initializing Vietnamese Summarization Pipeline...")
try:
    pipeline = VietnameseSummarizationPipeline()
    logger.info("Pipeline initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None


model_stt = FasterWhisper("large-v3")
model_llm = LLM(os.getenv("API_KEY"))

@router.get("/", response_model=APIInfo)
async def home():
    return "Hello hehe"


@router.post("/stt")
async def speech_to_text(audio_path: str = Form(...)):
    """
    Chuyển đổi audio thành văn bản từ file path
    Hỗ trợ định dạng: WAV, MP3, M4A
    """
    # Kiểm tra file có tồn tại không
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"File không tồn tại: {audio_path}"
        )

    # Kiểm tra định dạng file
    allowed_extensions = ['.wav', '.mp3', '.m4a']
    file_ext = os.path.splitext(audio_path)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ. Hỗ trợ: {', '.join(allowed_extensions)}"
        )

    try:
        # Xử lý audio trực tiếp từ file path
        result = model_stt.extract_text(audio_path)

        return JSONResponse(content={
            "success": True,
            "result": result,
            "file_path": audio_path  # Trả về path để debug
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý audio: {str(e)}"
        )

# Summarization endpoint
@router.post("/summarize", response_model=SummarizeResponse, responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
async def summarize(request: SummarizeRequest):
    """
    Summarize Vietnamese text using the complete pipeline.
    1. Vietnamese text → English translation
    2. English text → English summary (using Ollama LLM)
    3. English summary → Vietnamese summary
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
        logger.info("Processing Vietnamese text through summarization pipeline...")
        results = pipeline.process(request.text.strip(), request.summary_length)
        processing_time = time.time() - start_time
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


@router.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": request.model,
                    "prompt": request.prompt,
                    "stream": request.stream
                },
                timeout=60.0
            )
            return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize_gemini")
async def summarize_gemini(script: Script):
    prompt = model_llm.prompt_summarize(script.script)
    result = await model_llm.send_message_gemini(prompt)
    return result


@router.post("/chat_gemini")
async def summarize_gemini(user_input: UserInput):
    prompt = model_llm.prompt_qa_script(user_input.user_input, user_input.summarize_script)
    result = await model_llm.send_message_gemini(prompt)
    return result


@router.post("/extract-audio")
async def extract_audio_from_path(request: VideoPathRequest):
    try:
        audio_path = extract_audio(request.video_path, request.output_dir)

        return {
            "status": "success",
            "audio_path": audio_path,
            "filename": os.path.basename(audio_path)
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {str(e)}")