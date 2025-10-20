# Cài đặt các thư viện cần thiết:
# pip install torch torchaudio sounddevice numpy pyyaml onnxruntime onnxruntime-gpu punctuators

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
        if a[-L:] == b[:L]:
            return L
    return 0

_WORD_BOUNDARY_RE = re.compile(r"[ \t\n\r\f\v.,!?;:…，。！？；：]")

def _split_commit_tail(text: str, reserve_last_k_words: int = 1):
    parts = re.split(r"(\s+|[.,!?;:…])", text)
    words = []
    buf = ""
    for p in parts:
        buf += p
        if _WORD_BOUNDARY_RE.fullmatch(p or ""):
            words.append(buf)
            buf = ""
    if buf:
        words.append(buf)
    if not words:
        return "", text
    if len(words) <= reserve_last_k_words:
        return "", "".join(words)
    commit = "".join(words[:-reserve_last_k_words])
    tail = "".join(words[-reserve_last_k_words:])
    return commit, tail

# ==================== ASR Worker Thread ====================

@torch.no_grad()
def asr_worker(args, asr_model, char_dict, raw_text_queue):
    """
    Luồng chạy ngầm: ASR -> đẩy text thô vào queue.
    Cơ chế flush: nếu không có tiến triển giải mã trong N chunk liên tiếp => flush tail.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subsampling = asr_model.encoder.embed.subsampling_factor
    num_layers = asr_model.encoder.num_blocks
    conv_lorder = asr_model.encoder.cnn_module_kernel // 2

    # quy đổi stream_chunk_sec thành số bước encoder
    enc_steps = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size = enc_steps
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size

    att_cache = torch.zeros(
        (num_layers, left_context_size,
         asr_model.encoder.attention_heads,
         asr_model.encoder._output_size * 2 // asr_model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros(
        (num_layers, asr_model.encoder._output_size, conv_lorder),
        device=device
    )
    offset = torch.zeros(1, dtype=torch.int, device=device)

    sr = args.mic_sr
    block_samples = int(args.stream_chunk_sec * sr)
    lookahead_blocks = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q = deque()
    carry_text = ""
    stall_runs = 0  # đếm số chunk liên tiếp không tiến triển
    last_tail = ""  # tail của vòng trước để so sánh

    print("ASR worker started. Listening...")
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_samples) as stream:
            while not args.stop_event.is_set():
                audio_block, _ = stream.read(block_samples)
                a = np.squeeze(audio_block, axis=1).astype(np.float32)
                q.append(a)

                # đủ lookahead để suy luận
                if len(q) < 1 + lookahead_blocks:
                    continue

                # tạo segment: block hiện tại + lookahead
                seg_np = np.concatenate([q[0]] + list(list(q)[1:1 + lookahead_blocks]))
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * (1 << 15)

                # bảo vệ đoạn quá ngắn
                if seg.size(1) < int(0.025 * sr):
                    q.popleft()
                    continue

                # FBANK
                x = kaldi.fbank(
                    seg,
                    num_mel_bins=80,
                    frame_length=25,
                    frame_shift=10,
                    dither=0.0,
                    energy_floor=0.0,
                    sample_frequency=16000
                ).unsqueeze(0).to(device)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                # Encoder + CTC step
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    enc_out, enc_len, _, att_cache, cnn_cache, offset = asr_model.encoder.forward_parallel_chunk(
                        xs=x,
                        xs_origin_lens=x_len,
                        chunk_size=chunk_size,
                        left_context_size=left_context_size,
                        right_context_size=right_context_size,
                        att_cache=att_cache,
                        cnn_cache=cnn_cache,
                        truncated_context_size=chunk_size,
                        offset=offset
                    )
                    enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :enc_len]
                    if enc_out.size(1) > chunk_size:
                        enc_out = enc_out[:, :chunk_size]
                    offset = offset - enc_len + enc_out.size(1)
                    hyp_step = asr_model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                chunk_text = get_output([hyp_step], char_dict)[0]

                # hợp nhất với carry bằng LPS overlap
                ov = _longest_suffix_prefix_overlap(carry_text, chunk_text, max_k=32)
                merged = carry_text + chunk_text[ov:]

                # tách commit/tail theo số từ dự trữ
                commit, new_tail = _split_commit_tail(
                    merged,
                    reserve_last_k_words=max(1, args.stable_reserve_words)
                )

                # đẩy phần commit ổn định
                if commit:
                    raw_text_queue.put(commit)

                # đánh giá tiến triển: nếu không có commit mới và tail không đổi => stall
                progressed = bool(commit) or (new_tail != last_tail)

                if progressed:
                    stall_runs = 0
                else:
                    stall_runs += 1

                # nếu stall đủ N chunk => flush tail để hiển thị
                if stall_runs >= args.no_progress_patience and new_tail.strip():
                    raw_text_queue.put(new_tail)
                    new_tail = ""
                    stall_runs = 0

                # cập nhật carry và last_tail
                carry_text = new_tail
                last_tail = new_tail

                # trượt cửa sổ
                q.popleft()

    except Exception as e:
        print(f"Error in ASR worker: {e}")
    finally:
        # flush phần tail còn lại khi dừng
        if carry_text.strip():
            raw_text_queue.put(carry_text)
        raw_text_queue.put(None)
        print("ASR worker finished.")

# ==================== Model init ====================

@torch.no_grad()
def init_asr_model(args):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    config_path = os.path.join(args.model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(args.model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(args.model_checkpoint, "vocab.txt")
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
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
    if args.punc_device == "cuda":
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Punctuation model will run on CUDA (with CPU fallback).")
    else:
        prov = ["CPUExecutionProvider"]
        print("Punctuation model will run on CPU.")
    for name in dir(m):
        obj = getattr(m, name)
        if isinstance(obj, ort.InferenceSession):
            obj.set_providers(prov)
    print("Punctuation model loaded.")
    return m

# ==================== Main Thread: Punctuation and Display ====================

def main():
    parser = argparse.ArgumentParser(description="Real-time ASR with Rewritable Punctuation")

    # --- Tham số cho model ASR ---
    asr_parser = parser.add_argument_group("ASR Model & Streaming Parameters")
    asr_parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to ASR model directory")
    asr_parser.add_argument("--chunk_size", type=int, default=64)
    asr_parser.add_argument("--left_context_size", type=int, default=128)
    asr_parser.add_argument("--right_context_size", type=int, default=16)
    asr_parser.add_argument("--mic_sr", type=int, default=16000)
    asr_parser.add_argument("--stream_chunk_sec", type=float, default=0.5)
    asr_parser.add_argument("--lookahead_sec", type=float, default=0.2)
    asr_parser.add_argument("--stable_reserve_words", type=int, default=2)
    # MỚI: patience theo số chunk không tiến triển
    asr_parser.add_argument("--no_progress_patience", type=int, default=3,
                            help="Flush tail nếu không có tiến triển giải mã trong N chunk liên tiếp.")

    # Punctuation
    punc_parser = parser.add_argument_group("Punctuation Model Parameters")
    punc_parser.add_argument("--punc_model", type=str, default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
                              help="Name of the punctuation model from Hugging Face.")
    punc_parser.add_argument("--punc_device", type=str, default="cuda", choices=["cuda", "cpu"],
                              help="Device to run the punctuation model on.")
    punc_parser.add_argument("--use_sbd", action="store_true", help="Enable Sentence Boundary Detection in the punctuation model.")

    args = parser.parse_args()
    args.stop_event = threading.Event()

    # --- Khởi tạo ---
    asr_model, char_dict = init_asr_model(args)
    punctuation_model = init_punctuation_model(args)
    raw_text_queue = queue.Queue()

    # --- Khởi chạy luồng ASR ngầm ---
    asr_thread = threading.Thread(
        target=asr_worker,
        args=(args, asr_model, char_dict, raw_text_queue)
    )
    asr_thread.start()

    # --- Vòng lặp chính: Chuẩn hóa và Hiển thị ---
    raw_transcript = ""
    last_displayed_text = ""

    print("\nMic streaming. Nhấn Ctrl+C để dừng.")
    try:
        while True:
            new_text_received = False
            while not raw_text_queue.empty():
                chunk = raw_text_queue.get()
                if chunk is None:
                    args.stop_event.set()
                    break
                raw_transcript += chunk
                new_text_received = True

            if args.stop_event.is_set() and raw_text_queue.empty():
                break

            if new_text_received and raw_transcript.strip():
                punctuated_hypothesis = punctuation_model.infer(
                    [raw_transcript], apply_sbd=args.use_sbd
                )[0]
                last_displayed_text = punctuated_hypothesis

            print(f"\r{last_displayed_text.strip()} ", end="", flush=True)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nĐang dừng...")
        args.stop_event.set()


if __name__ == "__main__":
    # Gán các tham số mặc định để chạy trực tiếp script
    sys.argv = [
        "stream_mic_configurable.py",
        # --- ASR PARAMS ---
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--mic_sr", "16000",
        "--left_context_size", "128",
        "--right_context_size", "32",
        "--stream_chunk_sec", "0.5",
        "--lookahead_sec", "0.5",
        "--stable_reserve_words", "2",
        # MỚI: patience không tiến triển
        "--no_progress_patience", "1",
        # Punctuation
        "--punc_model", "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
        "--punc_device", "cuda",
        # "--use_sbd",
    ]
    main()