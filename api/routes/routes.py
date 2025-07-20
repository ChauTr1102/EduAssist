from api.routes import *

model_stt = FasterWhisper("large-v3")

@router.get("/")
def read_root():
    return {
        "EduAssist Here!"
    }

@router.get("/test")
def hehe():
    import os, sys
    return {"Python:": sys.executable,
            "LD_LIBRARY_PATH:": os.environ.get("LD_LIBRARY_PATH")}


@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    if audio.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/x-m4a", "audio/m4a"]:
        raise HTTPException(status_code=400, detail="Invalid audio format")
    # Lưu file tạm ra đĩa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name
    result = model_stt.extract_text(temp_audio_path)
    return JSONResponse(content={"text": result})
