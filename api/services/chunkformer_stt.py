from api.services import *
import subprocess, sys
import time


class Chunkformer:
    def __init__(self, model_dir="api/services/chunkformer-large-vie", decode_py = "api/services/chunkformer/decode.py", compute_type = "cuda"):
        self.model_dir = model_dir
        self.decoce_py = decode_py
        self.compute_type = compute_type

    def run_chunkformer_stt(self,audio_path):
        cmd = [
            sys.executable, self.decoce_py ,
            "--model_checkpoint", self.model_dir ,
            "--long_form_audio", audio_path,
            "--total_batch_duration", "1800",
            "--chunk_size", "64",
            "--left_context_size", "128",
            "--right_context_size", "128",
            "--device", self.compute_type ,
            "--autocast_dtype", "fp16",
        ]
        # capture_output=True để lấy transcript về
        result = subprocess.run(cmd, text=True, capture_output=True)
        # print(result.stdout)
        stederr = None
        if result.stderr:
            stederr = result.stderr
        return {
            "transcribe_by_sentence": result.stdout,
            "stederr": stederr,
        }

