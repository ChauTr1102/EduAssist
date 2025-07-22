from api.services import *
from faster_whisper import WhisperModel
import os
from pathlib import Path


class FasterWhisper:
    def __init__(self, model_name="large-v3", models_dir="models", ):
        """
        Khởi tạo model Whisper với khả năng tự động tải model vào thư mục chỉ định

        Args:
            model_name (str): Tên model (tiny, base, small, medium, large-v3)
            models_dir (str): Thư mục lưu trữ model
        """
        self.model_name = model_name
        self.models_dir = models_dir

        # Tạo thư mục nếu chưa tồn tại
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        # Khởi tạo model
        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="int8_float16",
            download_root=self.models_dir,
            local_files_only=False  # Tự động tải nếu chưa có
        )

    def extract_text(self, audio_path, beam_size=5):
        """
        Chuyển đổi audio thành văn bản

        Args:
            audio_path (str): Đường dẫn file audio
            beam_size (int): Độ chính xác (từ 1-20)

        Returns:
            dict: Kết quả chứa text, ngôn ngữ và độ tin cậy
        """
        try:
            segments, info = self.model.transcribe(audio_path, beam_size=beam_size)
            transcribe_by_sentence = ""
            for segment in segments:
                transcribe_by_sentence += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"

            return {
                "transcribe_by_sentence": transcribe_by_sentence,
                "language": info.language,
                "language_probability": info.language_probability
            }
        except Exception as e:
            raise RuntimeError(f"Lỗi khi chuyển đổi audio: {str(e)}")

    def get_model_info(self):
        """Lấy thông tin về model hiện tại"""
        return {
            "model_name": self.model_name,
            "models_dir": os.path.abspath(self.models_dir),
            "model_path": os.path.join(os.path.abspath(self.models_dir), self.model_name)
        }