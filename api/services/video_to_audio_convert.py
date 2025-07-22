from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from pathlib import Path
import subprocess
import uuid
from typing import Optional

class VideoPathRequest(BaseModel):
    video_path: str
    output_dir: Optional[str] = "audio_output"


def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Trích xuất âm thanh từ video sử dụng ffmpeg và trả về đường dẫn tuyệt đối

    Args:
        video_path: Đường dẫn đến file video (có thể là relative hoặc absolute)
        output_dir: Thư mục đích (có thể là relative hoặc absolute)

    Returns:
        Đường dẫn tuyệt đối đến file audio đã trích xuất

    Raises:
        FileNotFoundError: Nếu video không tồn tại
        RuntimeError: Nếu có lỗi khi chạy ffmpeg
    """
    # Chuyển đổi đường dẫn video thành absolute path
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Chuyển đổi output_dir thành absolute path và tạo thư mục nếu cần
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Tạo tên file output
    video_name = Path(video_path).stem
    audio_filename = f"{video_name}_{uuid.uuid4().hex[:6]}.wav"
    audio_output_path = os.path.join(output_dir, audio_filename)

    # Trích xuất âm thanh
    try:
        subprocess.run([
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            audio_output_path
        ], check=True, capture_output=True)

        # Đảm bảo trả về absolute path
        return os.path.abspath(audio_output_path)

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else "Unknown ffmpeg error"
        raise RuntimeError(f"FFmpeg error: {error_msg}")