# Cài đặt các thư viện cần thiết:
# pip install torch torchaudio sounddevice numpy pyyaml onnxruntime-gpu punctuators colorama

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
import colorama

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

# Khởi tạo colorama để tự động hỗ trợ màu trên Windows
colorama.init(autoreset=True)


# ==================== Utils ====================
def _longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a.endswith(b[:L]):
            return L
    return 0


# ==================== ASR Worker Thread ====================
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

                # CHỈ GỬI ĐI PHẦN VĂN BẢN MỚI
                new_text = current_chunk_text[overlap:]
                if new_text:
                    full_hypothesis += new_text
                    hypothesis_queue.put(new_text)

                q.popleft()

    except Exception as e:
        print(f"\nError in ASR worker: {e}")
    finally:
        hypothesis_queue.put(None)
        print("ASR worker finished.")


# ==================== Model init ====================
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
    parser = argparse.ArgumentParser(description="Real-time ASR with True Sliding Window Processing")

    asr_parser = parser.add_argument_group("ASR Model & Streaming Parameters")
    # ... (giữ nguyên các parser.add_argument)
    asr_parser.add_argument("--model_checkpoint", type=str, required=True)
    asr_parser.add_argument("--mic_sr", type=int, default=16000)
    asr_parser.add_argument("--stream_chunk_sec", type=float, default=0.5)
    asr_parser.add_argument("--lookahead_sec", type=float, default=0.5)
    asr_parser.add_argument("--chunk_size", type=int, default=64)
    asr_parser.add_argument("--left_context_size", type=int, default=128)
    asr_parser.add_argument("--right_context_size", type=int, default=32)

    punc_parser = parser.add_argument_group("Streaming Punctuation Parameters")
    punc_parser.add_argument("--punc_model", type=str,
                             default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
    punc_parser.add_argument("--punc_device", type=str, default="cuda", choices=["cuda", "cpu"])
    punc_parser.add_argument("--use_sbd", action="store_true")
    punc_parser.add_argument("--punc_window_words", type=int, default=20)
    punc_parser.add_argument("--punc_commit_margin_words", type=int, default=8)

    args = parser.parse_args()
    args.stop_event = threading.Event()

    YELLOW = colorama.Fore.YELLOW
    BLUE = colorama.Fore.BLUE
    RESET = colorama.Style.RESET_ALL

    asr_model, char_dict = init_asr_model(args)
    punctuation_model = init_punctuation_model(args)
    hypothesis_queue = queue.Queue()

    asr_thread = threading.Thread(target=asr_worker, args=(args, asr_model, char_dict, hypothesis_queue))
    asr_thread.start()

    # THAY ĐỔI LỚN: Quản lý state theo logic mới
    final_punctuated_text = ""
    active_raw_buffer = ""  # Buffer thô cho vùng active
    last_displayed_text = ""

    print("\nMic streaming. Nhấn Ctrl+C để dừng.")
    try:
        while True:
            # Lấy các chunk text MỚI từ ASR worker
            new_text_chunk = None
            try:
                # Lấy item mới nhất, không block
                new_text_chunk = hypothesis_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.05)
                continue

            if new_text_chunk is None:
                args.stop_event.set()
                break

            # Nối chunk mới vào buffer
            active_raw_buffer += new_text_chunk

            # THAY ĐỔI LỚN: Model chỉ xử lý trên buffer active, không xử lý toàn bộ transcript
            if not active_raw_buffer.strip():
                continue

            punctuated_active_buffer = punctuation_model.infer([active_raw_buffer], apply_sbd=args.use_sbd)[0]

            words_in_buffer = punctuated_active_buffer.split()

            # Logic chốt và cắt buffer
            if len(words_in_buffer) > args.punc_window_words:
                commit_point_words = len(words_in_buffer) - args.punc_commit_margin_words

                # Phần văn bản đã thêm dấu câu để chốt
                text_to_commit = " ".join(words_in_buffer[:commit_point_words]) + " "
                final_punctuated_text += text_to_commit

                # Phần buffer đã thêm dấu câu còn lại để hiển thị
                punctuated_active_buffer = " ".join(words_in_buffer[commit_point_words:])

                # Cắt bớt buffer thô tương ứng
                # Ước tính số từ thô tương ứng với số từ đã chốt
                raw_words = active_raw_buffer.split()
                # Đây là một cách ước lượng, có thể cải tiến thêm nếu cần độ chính xác tuyệt đối
                if len(raw_words) > commit_point_words:
                    active_raw_buffer = " ".join(raw_words[commit_point_words - 2:])  # Giữ lại 2 từ làm ngữ cảnh

            # Logic tô màu hiển thị (giữ nguyên)
            active_words = punctuated_active_buffer.split()
            num_active_words = len(active_words)

            blue_text, yellow_text = "", ""
            if num_active_words > args.punc_commit_margin_words:
                margin_point = num_active_words - args.punc_commit_margin_words
                blue_text = " ".join(active_words[:margin_point])
                yellow_text = " ".join(active_words[margin_point:])
            else:
                yellow_text = " ".join(active_words)

            colored_parts = []
            if blue_text: colored_parts.append(f"{BLUE}{blue_text}")
            if yellow_text: colored_parts.append(f"{YELLOW}{yellow_text}")

            display_text = f"\r{final_punctuated_text}{' '.join(colored_parts)}{RESET} "

            if display_text != last_displayed_text:
                print(display_text.ljust(100), end="", flush=True)
                last_displayed_text = display_text

    except KeyboardInterrupt:
        print("\nĐang dừng...")
        args.stop_event.set()
    finally:
        asr_thread.join()
        if active_raw_buffer.strip():
            final_punctuated_text += punctuation_model.infer([active_raw_buffer])[0]
        print(f"\r{final_punctuated_text.strip()} ")
        print("\nHoàn tất.")


if __name__ == "__main__":
    sys.argv = [
        "stream_mic_optimized.py",
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--stream_chunk_sec", "0.5",
        "--lookahead_sec", "0.5",
        "--punc_window_words", "60",
        "--punc_commit_margin_words", "20",
        "--punc_device", "cuda",
    ]

    main()