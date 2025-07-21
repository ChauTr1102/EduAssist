from fastapi import UploadFile, File, APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
import tempfile
import os
from api.services.whisper import FasterWhisper

router = APIRouter()

# Khởi tạo model với thư mục tùy chỉnh
model_stt = FasterWhisper("large-v3", models_dir="models")


@router.get("/")
def read_root():
    return {"message": "EduAssist STT Service"}


@router.get("/model-info")
def get_model_info():
    """Lấy thông tin về model hiện tại"""
    return model_stt.get_model_info()


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
