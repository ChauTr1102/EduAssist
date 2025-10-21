# Cài đặt các thư viện cần thiết:
# pip install torch torchaudio sounddevice numpy pyyaml onnxruntime-gpu punctuators

import os
import math
import argparse
import yaml
import re
import sys
from collections import deque
import threading
import queue
import time

# Punctuation model imports
import onnxruntime as ort
from punctuators.models import PunctCapSegModelONNX

# ASR model imports
import torch
import numpy as np
import sounddevice as sd
import torchaudio
import torchaudio.compliance.kaldi as kaldi

# Giả định các file utils của ChunkFormer nằm trong thư mục 'model'
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output


# ==================== Utils ====================
def _longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a.endswith(b[:L]):
            return L
    return 0


# ==================== ASR Worker Thread (Kiến trúc mới) ====================
@torch.no_grad()
def asr_worker(args, asr_model, char_dict, hypothesis_queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subsampling = asr_model.encoder.embed.subsampling_factor
    num_layers = asr_model.encoder.num_blocks
    conv_lorder = asr_model.encoder.cnn_module_kernel // 2
    enc_steps = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size = enc_steps
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    att_cache = torch.zeros((num_layers, left_context_size, asr_model.encoder.attention_heads,
                             asr_model.encoder._output_size * 2 // asr_model.encoder.attention_heads), device=device)
    cnn_cache = torch.zeros((num_layers, asr_model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)
    sr = args.mic_sr
    block_samples = int(args.stream_chunk_sec * sr)
    lookahead_blocks = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q = deque()
    full_hypothesis = ""

    print("ASR worker started (new architecture). Listening...")
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_samples) as stream:
            while not args.stop_event.is_set():
                audio_block, _ = stream.read(block_samples)
                q.append(np.squeeze(audio_block, axis=1).astype(np.float32))

                if len(q) < 1 + lookahead_blocks: continue

                # NỐI CHUỖI CÁC BLOCK TRONG HÀNG ĐỢI THAY VÌ CHỈ 1 BLOCK
                seg_np = np.concatenate(list(q))
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * (1 << 15)

                if seg.size(1) < int(0.025 * sr):
                    q.popleft()
                    continue

                x = kaldi.fbank(seg, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, energy_floor=0.0,
                                sample_frequency=16000).unsqueeze(0).to(device)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    enc_out, enc_len, _, att_cache, cnn_cache, offset = asr_model.encoder.forward_parallel_chunk(
                        xs=x, xs_origin_lens=x_len, chunk_size=chunk_size,
                        left_context_size=left_context_size, right_context_size=right_context_size,
                        att_cache=att_cache, cnn_cache=cnn_cache,
                        truncated_context_size=chunk_size, offset=offset
                    )
                    enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :enc_len]
                    if enc_out.size(1) > chunk_size: enc_out = enc_out[:, :chunk_size]
                    offset = offset - enc_len + enc_out.size(1)
                    hyp_step = asr_model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                current_chunk_text = get_output([hyp_step], char_dict)[0]
                overlap = _longest_suffix_prefix_overlap(full_hypothesis, current_chunk_text)
                full_hypothesis += current_chunk_text[overlap:]

                hypothesis_queue.put(full_hypothesis)
                q.popleft()

    except Exception as e:
        print(f"\nError in ASR worker: {e}")
    finally:
        hypothesis_queue.put(None)
        print("ASR worker finished.")


# ==================== Model init ====================
# (Không thay đổi)
@torch.no_grad()
def init_asr_model(args):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    config_path = os.path.join(args.model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(args.model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(args.model_checkpoint, "vocab.txt")
    with open(config_path, 'r') as fin: config = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(config, config_path)
    model.eval()
    load_checkpoint(model, checkpoint_path)
    model.encoder = model.encoder.cuda()
    model.ctc = model.ctc.cuda()
    symbol_table = read_symbol_table(symbol_table_path)
    char_dict = {v: k for k, v in symbol_table.items()}
    return model, char_dict


def init_punctuation_model(args):
    print(f"Loading punctuation model: {args.punc_model}")
    m = PunctCapSegModelONNX.from_pretrained(args.punc_model)
    available_providers = ort.get_available_providers()
    if args.punc_device == "cuda" and "CUDAExecutionProvider" in available_providers:
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Punctuation model will run on CUDA.")
    else:
        prov = ["CPUExecutionProvider"]
        if args.punc_device == "cuda":
            print("Warning: CUDAExecutionProvider not found. Punctuation model will run on CPU.")
        else:
            print("Punctuation model will run on CPU.")
    for name in dir(m):
        obj = getattr(m, name)
        if isinstance(obj, ort.InferenceSession): obj.set_providers(prov)
    print("Punctuation model loaded.")
    return m


# ==================== Main Thread: Punctuation and Display ====================
def main():
    parser = argparse.ArgumentParser(description="Real-time ASR with new robust streaming architecture")

    # SỬA LỖI: THÊM LẠI TOÀN BỘ PHẦN ĐỊNH NGHĨA THAM SỐ
    # --- Tham số cho model ASR ---
    asr_parser = parser.add_argument_group("ASR Model & Streaming Parameters")
    asr_parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to ASR model directory")
    asr_parser.add_argument("--mic_sr", type=int, default=16000)
    asr_parser.add_argument("--stream_chunk_sec", type=float, default=0.5)
    asr_parser.add_argument("--lookahead_sec", type=float, default=0.5)
    asr_parser.add_argument("--chunk_size", type=int, default=64,
                            help="Internal chunk size for the model, not for audio.")
    asr_parser.add_argument("--left_context_size", type=int, default=128)
    asr_parser.add_argument("--right_context_size", type=int, default=32)

    # --- Tham số cho Punctuation ---
    punc_parser = parser.add_argument_group("Streaming Punctuation Parameters")
    punc_parser.add_argument("--punc_model", type=str,
                             default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
    punc_parser.add_argument("--punc_device", type=str, default="cuda", choices=["cuda", "cpu"])
    punc_parser.add_argument("--use_sbd", action="store_true")
    punc_parser.add_argument("--punc_window_words", type=int, default=120,
                             help="Number of words in the active (yellow) window before committing.")
    punc_parser.add_argument("--punc_commit_margin_words", type=int, default=50,
                             help="Number of words to keep for context after a commit.")

    args = parser.parse_args()
    args.stop_event = threading.Event()
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    asr_model, char_dict = init_asr_model(args)
    punctuation_model = init_punctuation_model(args)
    hypothesis_queue = queue.Queue()

    asr_thread = threading.Thread(target=asr_worker, args=(args, asr_model, char_dict, hypothesis_queue))
    asr_thread.start()

    final_punctuated_text = ""
    raw_transcript = ""
    last_displayed_text = ""

    print("\nMic streaming. Nhấn Ctrl+C để dừng.")
    try:
        while True:
            latest_hypothesis = None
            while not hypothesis_queue.empty():
                item = hypothesis_queue.get()
                if item is None:
                    args.stop_event.set()
                    break
                latest_hypothesis = item
            if args.stop_event.is_set(): break

            if latest_hypothesis is None or latest_hypothesis == raw_transcript:
                time.sleep(0.05)
                continue

            raw_transcript = latest_hypothesis
            punctuated_transcript = punctuation_model.infer([raw_transcript], apply_sbd=args.use_sbd)[0]
            words = punctuated_transcript.split()

            committed_text = ""
            active_text = punctuated_transcript

            if len(words) > args.punc_window_words:
                commit_point = len(words) - args.punc_commit_margin_words
                committed_text = " ".join(words[:commit_point]) + " "
                active_text = " ".join(words[commit_point:])

            final_punctuated_text = committed_text

            display_text = f"\r{final_punctuated_text}{YELLOW}{active_text}{RESET} "
            if display_text != last_displayed_text:
                print(display_text.ljust(100), end="", flush=True)  # Dùng ljust để xóa dòng cũ
                last_displayed_text = display_text

    except KeyboardInterrupt:
        print("\nĐang dừng...")
        args.stop_event.set()
    finally:
        asr_thread.join()
        final_text = punctuation_model.infer([raw_transcript], apply_sbd=args.use_sbd)[0]
        print(f"\r{final_text.strip()} ")
        print("\nHoàn tất.")


if __name__ == "__main__":
    """
    Đây là khối cấu hình để chạy script trực tiếp, mô phỏng việc gõ lệnh từ terminal.
    Bạn có thể bỏ comment và thay đổi các giá trị dưới đây để thử nghiệm.
    Các tham số được chia thành 2 nhóm: ASR & Streaming và Punctuation & Display.
    """
    sys.argv = [
        "stream_mic_new_arch.py",


        # --- ASR & STREAMING ---
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--mic_sr", "16000",
        "--stream_chunk_sec", "0.5",
        "--lookahead_sec", "0.5",
        "--chunk_size", "64",
        "--left_context_size", "128",
        "--right_context_size", "32",

        # --- PUNCTUATION & DISPLAY ---
        "--punc_model", "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
        "--punc_device", "cuda",

        # Tổng số từ tối đa trong vùng "đang xử lý" (màu vàng).
        # Khi văn bản dài hơn ngưỡng này, phần đầu sẽ được "chốt" (chuyển sang màu trắng).
        # Giá trị lớn hơn (ví dụ: 30) sẽ giữ lại nhiều văn bản hơn để sửa đổi.
        "--punc_window_words", "24",

        # Số từ được giữ lại trong vùng màu vàng sau khi chốt để làm ngữ cảnh.
        # Giá trị này nên bằng khoảng 1/3 đến 1/2 của punc_window_words.
        "--punc_commit_margin_words", "8",

        # Bật tính năng tự động ngắt câu (Sentence Boundary Detection) của model dấu câu.
        # Để bật, hãy bỏ comment dòng dưới đây.
        # "--use_sbd",
    ]

    main()