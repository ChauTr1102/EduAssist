from api.services import *
import subprocess, sys
import time
def run_chunkformer(audio_path):
    cmd = [
        sys.executable, "chunkformer/decode.py",
        "--model_checkpoint", "chunkformer-large-vie",
        "--long_form_audio", audio_path,
        "--total_batch_duration", "1800",
        "--chunk_size", "64",
        "--left_context_size", "128",
        "--right_context_size", "128",
        "--device", "cuda",
        "--autocast_dtype", "fp16",
    ]
    # capture_output=True để lấy transcript về
    result = subprocess.run(cmd, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

# chạy thử
start_time = time.time()
run_chunkformer("/data/kinh_te_chinh_tri_2m_47s.MP3")
end_time = time.time() # Record the end time
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")


