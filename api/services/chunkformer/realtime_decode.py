import os
import math
import argparse
import yaml
import re
import sys  # Đã thêm sys
from collections import deque

import torch
import numpy as np
import sounddevice as sd
import torchaudio
import torchaudio.compliance.kaldi as kaldi

# Các hàm utility không cần thiết đã được loại bỏ
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output


# ==================== Utils for stable streaming ====================

def _longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    """Độ dài trùng khớp dài nhất giữa suffix(a) và prefix(b) để khử trùng lặp chồng lấn."""
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a[-L:] == b[:L]:
            return L
    return 0


_WORD_BOUNDARY_RE = re.compile(r"[ \t\n\r\f\v.,!?;:…，。！？；：]")


def _split_commit_tail(text: str, reserve_last_k_words: int = 1):
    """
    Chỉ commit đến ranh giới từ chắc chắn.
    Giữ lại k từ cuối làm 'tail' để tránh cắt nửa từ ở biên.
    """
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


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


# ==================== Model init ====================

@torch.no_grad()
def init(model_checkpoint):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

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


# ==================== Microphone streaming with fixed lookahead ====================

@torch.no_grad()
def stream_mic(args, model, char_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subsampling = model.encoder.embed.subsampling_factor
    num_layers = model.encoder.num_blocks
    conv_lorder = model.encoder.cnn_module_kernel // 2

    enc_steps = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size = enc_steps
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size

    att_cache = torch.zeros(
        (num_layers, left_context_size, model.encoder.attention_heads,
         model.encoder._output_size * 2 // model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros((num_layers, model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    sr = args.mic_sr
    assert sr == 16000, "Mic nên 16 kHz để khớp feature"
    block_samples = int(args.stream_chunk_sec * sr)
    lookahead_blocks = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q = deque()
    committed_text = ""
    carry_text = ""
    silence_run = 0
    SIL_THRESH = args.silence_rms
    SIL_RUN_TO_FLUSH = args.silence_runs

    print("Mic streaming. Nhấn Ctrl+C để dừng.")
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_samples) as stream:
            while True:
                audio_block, _ = stream.read(block_samples)
                a = np.squeeze(audio_block, axis=1).astype(np.float32)
                q.append(a)

                if _rms(a) < SIL_THRESH:
                    silence_run += 1
                else:
                    silence_run = 0

                if len(q) < 1 + lookahead_blocks:
                    continue

                seg_np = np.concatenate([q[0]] + list(list(q)[1:1 + lookahead_blocks]))
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * (1 << 15)

                if seg.size(1) < int(0.025 * sr):
                    q.popleft()
                    continue

                x = kaldi.fbank(
                    seg, num_mel_bins=80, frame_length=25, frame_shift=10,
                    dither=0.0, energy_floor=0.0, sample_frequency=16000
                ).unsqueeze(0).to(device)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                truncated_context_size = chunk_size
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    enc_out, enc_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
                        xs=x, xs_origin_lens=x_len, chunk_size=chunk_size,
                        left_context_size=left_context_size, right_context_size=right_context_size,
                        att_cache=att_cache, cnn_cache=cnn_cache,
                        truncated_context_size=truncated_context_size, offset=offset
                    )
                    enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :enc_len]
                    if enc_out.size(1) > truncated_context_size:
                        enc_out = enc_out[:, :truncated_context_size]
                    offset = offset - enc_len + enc_out.size(1)
                    hyp_step = model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                chunk_text = get_output([hyp_step], char_dict)[0]

                ov = _longest_suffix_prefix_overlap(carry_text, chunk_text, max_k=32)
                merged = carry_text + chunk_text[ov:]
                commit, new_tail = _split_commit_tail(merged, reserve_last_k_words=max(1, args.stable_reserve_words))

                if commit:
                    committed_text += commit

                if silence_run >= SIL_RUN_TO_FLUSH and new_tail.strip():
                    committed_text += new_tail
                    new_tail = ""

                print(f"\r{committed_text.strip()} {new_tail.strip()}", end="", flush=True)

                carry_text = new_tail
                q.popleft()
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\nĐã dừng.")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="ChunkFormer Microphone Streaming")

    parser.add_argument("--model_checkpoint", type=str, required=True, help="Đường dẫn đến thư mục chứa model")
    parser.add_argument("--chunk_size", type=int, default=64, help="Kích thước chunk của encoder (sau subsampling)")
    parser.add_argument("--left_context_size", type=int, default=128, help="Kích thước context trái (encoder steps)")
    parser.add_argument("--right_context_size", type=int, default=16, help="Kích thước context phải (encoder steps)")
    parser.add_argument("--mic_sr", type=int, default=16000, help="Tần số lấy mẫu của micro")
    parser.add_argument("--stream_chunk_sec", type=float, default=0.5, help="Độ dài mỗi lát audio (giây)")
    parser.add_argument("--lookahead_sec", type=float, default=0.2, help="Audio bổ sung cho context phải (giây)")
    parser.add_argument("--stable_reserve_words", type=int, default=1, help="Giữ lại k từ cuối để tránh cắt nửa từ")
    parser.add_argument("--silence_rms", type=float, default=0.005, help="Ngưỡng RMS coi là im lặng")
    parser.add_argument("--silence_runs", type=int, default=3, help="Số block im lặng liên tiếp để flush đuôi")

    args = parser.parse_args()

    print(f"Model Checkpoint: {args.model_checkpoint}")
    print(f"Chunk Size (encoder steps): {args.chunk_size}")
    print(f"Left Context Size: {args.left_context_size}")
    print(f"Right Context Size: {args.right_context_size}")

    model, char_dict = init(args.model_checkpoint)
    stream_mic(args, model, char_dict)


if __name__ == "__main__":

    sys.argv = [
        "stream_mic.py",
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--mic_sr", "16000",
        "--left_context_size", "128",
        "--right_context_size", "32",
        "--stream_chunk_sec", "0.5",
        "--lookahead_sec", "0.5",
        "--stable_reserve_words", "2",
        "--silence_runs", "1"
    ]

    main()