from faster_whisper import WhisperModel
import time
##############################################################################
# CẤU HÌNH MẶC ĐỊNH - NGƯỜI DÙNG CÓ THỂ THAY ĐỔI CÁC THAM SỐ SAU:
##############################################################################

# 1. CHỌN MODEL (tiny, base, small, medium, large-v2, large-v3)
# - Model càng nhỏ (tiny) → càng nhanh, càng ít chính xác
# - Model càng lớn (large-v3) → càng chậm, càng chính xác
MODEL_SIZE = "large-v2"

# 2. CHỌN THIẾT BỊ ("cpu" hoặc "cuda" nếu có GPU)
# - "cpu"     : Chạy trên CPU (phù hợp máy yếu)
# - "cuda"    : Chạy trên GPU NVIDIA (nhanh hơn 5-10 lần)
DEVICE = "cuda"

# 3. CHỌN LOẠI TÍNH TOÁN (tuỳ DEVICE đã chọn)
# - Nếu dùng GPU: "float16" (nhanh), "int8_float16" (tiết kiệm VRAM)
# - Nếu dùng CPU: "int8" (nhẹ nhất), "float32" (chính xác hơn)
COMPUTE_TYPE = "int8" if DEVICE == "cpu" else "int8_float16"

# 4. ĐƯỜNG DẪN FILE ÂM THANH CẦN XỬ LÝ
AUDIO_PATH = "data/kinh_te_chinh_tri_2m_47s.MP3"

# 5. CÀI ĐẶT BEAM SIZE (ảnh hưởng tốc độ/độ chính xác)
# - Giảm beam_size để tăng tốc (nhưng giảm độ chính xác)
# - Tăng beam_size nếu cần độ chính xác cao (mặc định: 5)
BEAM_SIZE = 5

##############################################################################
# KHỞI TẠO MODEL VÀ XỬ LÝ AUDIO - KHÔNG CẦN SỬA CODE PHẦN NÀY
##############################################################################

def main():
    # Khởi tạo model với cấu hình đã chọn
    if DEVICE == "cpu":
        model = WhisperModel(
            MODEL_SIZE,  # Hoặc "base" nếu CPU yếu hơn
            device=DEVICE,
            compute_type="int8",  # Nhẹ nhất (hoặc "float32" nếu cần độ chính xác cao)
            cpu_threads=12,  # ⭐ Sử dụng toàn bộ 12 luồng
            num_workers=6  # Số lượng processes xử lý song song (tối đa = số nhân vật lý = 6)
        )
    else:
        model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )

    # Transcribe với ngôn ngữ tiếng Việt
    segments, info = model.transcribe(
        AUDIO_PATH,
        language="vi",  # ⚠️ Bắt buộc nếu muốn nhận diện tiếng Việt
        beam_size=BEAM_SIZE
    )

    # In thông tin ngôn ngữ phát hiện
    # print(f"Phát hiện ngôn ngữ: '{info.language}' (độ tin cậy: {info.language_probability*100:.2f}%)")

    # In kết quả theo từng đoạn
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian thực thi: {execution_time:.6f} giây")