from faster_whisper import WhisperModel
import time
import os
import pyaudio
import wave
import warnings

# Ignore all warnings to clean up output
warnings.filterwarnings("ignore")

##############################################################################
# CẤU HÌNH MẶC ĐỊNH - NGƯỜI DÙNG CÓ THỂ THAY ĐỔI CÁC THAM SỐ SAU:
##############################################################################

# 1. CHỌN MODEL (tiny, base, small, medium, large-v2, large-v3)
# - tiny: nhỏ nhất, nhanh nhất, độ chính xác thấp
# - base: cân bằng giữa tốc độ và độ chính xác
# - small: tốt cho hầu hết trường hợp
# - medium: chính xác hơn nhưng chậm hơn
# - large-v2/large-v3: chính xác nhất nhưng chậm và yêu cầu nhiều tài nguyên
MODEL_SIZE = "large-v3"

# 2. CHỌN THIẾT BỊ ("cpu" hoặc "cuda" nếu có GPU)
DEVICE = "cuda"

# 3. CHỌN LOẠI TÍNH TOÁN
# Các lựa chọn cho GPU:
# - "int8_float16" (tốt nhất cho tốc độ/độ chính xác)
# - "float16" (an toàn nếu gặp lỗi với int8_float16)
# - "float32" (chính xác nhất nhưng chậm)
# Cho CPU chỉ nên dùng "int8" hoặc "float32"
COMPUTE_TYPE = "int8" if DEVICE == "cpu" else "int8_float16"

# 4. THỜI GIAN GHI ÂM TỪ MICRO (giây)
RECORD_DURATION = 10  # Có thể điều chỉnh theo nhu cầu

# 5. TÊN FILE ÂM THANH TẠM THỜI
TEMP_AUDIO_PATH = "temp_recording.wav"

# 6. CÀI ĐẶT BEAM SIZE (kích thước chùm tìm kiếm)
# Giá trị lớn hơn cho kết quả chính xác hơn nhưng tốn tài nguyên
BEAM_SIZE = 5

# 7. THƯ MỤC LƯU MODEL
MODEL_DIR = "models"

# 8. CẤU HÌNG GHI ÂM
SAMPLE_RATE = 16000  # Whisper hoạt động tốt nhất với 16kHz
CHANNELS = 1  # Mono
FORMAT = pyaudio.paInt16  # Định dạng 16-bit PCM


##############################################################################
# HÀM GHI ÂM TỪ MICRO
##############################################################################

def record_from_mic(duration=RECORD_DURATION, output_path=TEMP_AUDIO_PATH):
    """Ghi âm từ microphone và lưu ra file WAV tạm thời

    Args:
        duration (int): Thời gian ghi âm (giây)
        output_path (str): Đường dẫn file output

    Returns:
        str: Đường dẫn file đã ghi
    """
    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=1024)

        print(f"\n🎤 Đang ghi âm trong {duration} giây... (Nhấn Ctrl+C để dừng sớm)")
        frames = []

        try:
            for _ in range(0, int(SAMPLE_RATE / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
        except KeyboardInterrupt:
            print("\n⏹️ Dừng ghi âm sớm theo yêu cầu")

        print("✅ Hoàn thành ghi âm")

        # Lưu file âm thanh
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))

        return output_path

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


##############################################################################
# KHỞI TẠO MODEL VÀ XỬ LÝ AUDIO
##############################################################################

def load_model():
    """Khởi tạo model Whisper với cấu hình đã chọn

    Returns:
        WhisperModel: Model đã được khởi tạo
    """
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Khởi tạo model với cấu hình đã chọn
    try:
        if DEVICE == "cpu":
            model = WhisperModel(
                MODEL_SIZE,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=os.cpu_count() or 4,  # Sử dụng tất cả core CPU
                num_workers=4,
                download_root=MODEL_DIR
            )
        else:
            model = WhisperModel(
                MODEL_SIZE,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                download_root=MODEL_DIR
            )


        return model

    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo model: {str(e)}")
        print("🔄 Đang thử dùng float16 thay cho int8_float16...")
        # Fallback nếu int8_float16 không hoạt động

def main():
    """Hàm chính thực hiện ghi âm và chuyển đổi giọng nói"""
    try:
        # Load model
        print("\n🔍 Đang tải model...")
        model = load_model()

        # Ghi âm từ micro
        audio_path = record_from_mic()
        print(audio_path)
        # Xử lý chuyển đổi
        print("\n🔊 Đang chuyển đổi giọng nói thành văn bản...")
        start_time = time.time()

        segments, info = model.transcribe(
            audio_path,
            language="vi",
            beam_size=BEAM_SIZE,
            vad_filter=True,  # Tự động lọc khoảng lặng
            without_timestamps=True  # Bỏ thời gian nếu không cần
        )

        # In kết quả
        print("\n📝 Kết quả:")
        full_text = []
        for segment in segments:
            print(f"- {segment.text}")
            full_text.append(segment.text)

        # Tính thời gian xử lý
        end_time = time.time()
        execution_time = end_time - start_time
        audio_duration = info.duration
        real_time_factor = execution_time / audio_duration

        print(f"\n⏱️ Thời gian xử lý: {execution_time:.2f}s cho {audio_duration:.2f}s âm thanh")
        print(f"⚡ Tốc độ xử lý: {real_time_factor:.2f}x thời gian thực")

        # Xóa file tạm
        if os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {str(e)}")
    finally:
        print(f"\n💾 Model được lưu tại: {os.path.abspath(MODEL_DIR)}")


if __name__ == "__main__":
    main()