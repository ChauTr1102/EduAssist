import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import soundfile as sf
import pyloudnorm as pyln
import soundfile as sf
import pyloudnorm as pyln
import os
import tempfile
# Settings
samplerate = 16000
block_duration = 4  # seconds
chunk_duration = 8  # seconds
channels = 1
frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")  # Tạo thư mục tạm
os.makedirs(temp_dir, exist_ok=True)


"""
  Hướng dẫn sử dụng tất cả compute_type trong Faster-Whisper.

  Args:
      model_size (str): Kích thước model (e.g., "tiny", "base", "small", "medium", "large-v3").
      compute_type (str): Loại tính toán (xem mô tả bên dưới).
      device (str): "cuda" cho GPU hoặc "cpu" cho CPU.

  Compute_type hỗ trợ:
  1. "float32"  - Độ chính xác cao nhất, chậm, tốn bộ nhớ. Dùng để debug.
  2. "float16"  - Cân bằng tốt giữa tốc độ và độ chính xác (mặc định nên dùng trên GPU).
  3. "int8"     - Tốc độ nhanh nhất, tiết kiệm VRAM, độ chính xác giảm nhẹ.
  4. "int8_float16" - Kết hợp int8 và float16 (ít dùng, thử nghiệm).
  5. "int8_float32" - Dùng int8 nhưng kiểm tra bằng float32 (độ chính xác cao hơn int8 thuần).
  6. "bf16"     - Hỗ trợ GPU Ampere (RTX 30xx+), tiết kiệm bộ nhớ như float16.
  """
# Model setup: medium.en + float16 (optimized for 3x80)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")  # use model size "medium.en" for faster

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate, channels=channels,
                      callback=audio_callback, blocksize=frames_per_block):
        print("🔍 Listening... Press Ctrl+C to stop.")
        while True:
            sd.sleep(100)

def check_loud_ness(audio_data,chunk_number):
    # Lưu file WAV
    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_number}.wav")
    sf.write(chunk_path, audio_data, samplerate)

    data, rate = sf.read(chunk_path)
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    if loudness > -50:
        # print("speaking..."+str(loudness)+ "chunk number: "+str(chunk_number))
        return True
    return False

def transcriber():
    global audio_buffer
    chunk_counter = 0

    while True:
        block = audio_queue.get()
        if check_loud_ness(block, chunk_counter) == True:
            # print('okay hear you '+ str(chunk_counter))
            audio_buffer.append(block)

            total_frames = sum(len(b) for b in audio_buffer)

            if total_frames >= frames_per_chunk:
                audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
                audio_buffer = []  # Clear buffer
                audio_data = audio_data.flatten().astype(np.float32)


                # Xử lý với Whisper
                segments, _ = model.transcribe(
                    audio_data,
                    language="vi",
                    beam_size=5
                )

                for segment in segments:
                    print(f"{segment.text}")

                chunk_counter += 1


# Start threads
threading.Thread(target=recorder, daemon=True).start()
transcriber()