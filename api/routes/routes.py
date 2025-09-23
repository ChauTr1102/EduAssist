import os

from api.routes import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


faster_whisper = FasterWhisper("large-v3")
model_llm = LLM(os.getenv("API_KEY"))
chunkformer_stt = Chunkformer()

@router.get("/", response_model=APIInfo)
async def home():
    return "Hello hehe"


import time
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
import os


@router.post("/stt_chunkformer")
async def speech_to_text(audio_path: str = Form(...)):
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"File không tồn tại: {audio_path}"
        )

    allowed_extensions = ['.wav', '.mp3', '.m4a']
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ. Hỗ trợ: {', '.join(allowed_extensions)}"
        )

    try:
        start_time = time.time()

        # Lấy VRAM trước khi chạy
        vram_before = get_gpu_memory_mb()

        result = chunkformer_stt.run_chunkformer_stt(audio_path)

        # Lấy VRAM sau khi chạy
        vram_after = get_gpu_memory_mb()

        end_time = time.time()
        processing_time = round(end_time - start_time, 3)

        # Tính VRAM trung bình
        vram_avg = round((vram_before + vram_after) / 2, 2)

        return JSONResponse(content={
            "success": True,
            "result": result,
            "file_path": audio_path,
            "processing_time": processing_time,
            "gpu_vram_avg_mb": vram_avg
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý audio: {str(e)}"
        )



@router.post("/stt_faster-whisper")
async def speech_to_text(audio_path: str = Form(...)):
    """
    Chuyển đổi audio thành văn bản từ file path
    Hỗ trợ định dạng: WAV, MP3, M4A
    """
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"File không tồn tại: {audio_path}"
        )

    allowed_extensions = ['.wav', '.mp3', '.m4a']
    file_ext = os.path.splitext(audio_path)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ. Hỗ trợ: {', '.join(allowed_extensions)}"
        )

    try:
        start_time = time.time()
        # Đo VRAM trước khi chạy model
        vram_before = get_gpu_memory_mb()

        result = faster_whisper.extract_text(audio_path)

        # Đo VRAM sau khi chạy model
        vram_after = get_gpu_memory_mb()

        end_time = time.time()
        processing_time = round(end_time - start_time, 3)  # làm tròn 3 số lẻ

        vram_avg = round((vram_before + vram_after) / 2, 2)

        return JSONResponse(content={
            "success": True,
            "result": result,
            "file_path": audio_path,
            "processing_time": processing_time,
            "gpu_vram_avg_mb": vram_avg
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý audio: {str(e)}"
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
async def chat_gemini(user_input: UserInput):
    prompt = model_llm.prompt_qa_script(user_input.user_input, user_input.summarize_script, user_input.history)
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