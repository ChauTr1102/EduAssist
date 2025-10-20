import os
import math
import argparse
import yaml
import re
import sys
from collections import deque

import torch
import numpy as np
import sounddevice as sd
import torchaudio
import torchaudio.compliance.kaldi as kaldi

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

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

def _take_last_k_words(s: str, k: int):
    """Cắt k từ cuối của s. Trả về (remain, last_k)."""
    if k <= 0 or not s:
        return s, ""
    tokens = re.split(r"(\s+|[.,!?;:…])", s)
    # gom lại theo ranh giới
    words = []
    buf = ""
    for t in tokens:
        buf += t
        if _WORD_BOUNDARY_RE.fullmatch(t or ""):
            words.append(buf)
            buf = ""
    if buf:
        words.append(buf)
    if not words:
        return "", s
    if k >= len(words):
        return "", "".join(words)
    keep = "".join(words[:-k])
    lastk = "".join(words[-k:])
    return keep, lastk

def _first_k_words(s: str, k: int):
    if k <= 0 or not s:
        return "", s
    tokens = re.split(r"(\s+|[.,!?;:…])", s)
    words = []
    buf = ""
    for t in tokens:
        buf += t
        if _WORD_BOUNDARY_RE.fullmatch(t or ""):
            words.append(buf)
            buf = ""
    if buf:
        words.append(buf)
    if not words:
        return "", s
    if k >= len(words):
        return s, ""
    first = "".join(words[:k])
    remain = "".join(words[k:])
    return first, remain


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


# ==================== Streaming with stall-based soft commit ====================

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
    assert sr == 16000, "Mic nên 16 kHz"
    block_samples = int(args.stream_chunk_sec * sr)
    lookahead_blocks = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q = deque()
    hard_text = ""        # khóa cứng, không hồi tố
    soft_text = ""        # khóa mềm, có thể hồi tố trong cửa sổ
    carry_text = ""       # đuôi chưa commit
    prev_display = ""
    no_new_char_chunks = 0

    # Tham số cơ chế stall
    STALL_N = args.stall_n_chunks           # số chunk liên tiếp không có ký tự mới
    SOFT_COMMIT_WORDS = args.soft_commit_words_per_stall  # số từ chuyển từ carry -> soft khi stall
    SOFT_WINDOW = args.soft_window_words    # tối đa số từ có thể hồi tố trong soft
    ROLLBACK_WORDS = args.soft_rollback_words  # khi có nói tiếp, hồi tối đa bấy nhiêu từ từ soft

    SIL_THRESH = args.silence_rms
    silence_run = 0

    print("Mic streaming. Ctrl+C để dừng.")
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

                # Hợp nhất đuôi trước đó với decode mới
                ov = _longest_suffix_prefix_overlap(carry_text, chunk_text, max_k=32)
                merged_tail = carry_text + chunk_text[ov:]

                # Chia commit/tail theo ranh giới từ ổn định
                commit_part, new_tail = _split_commit_tail(merged_tail, reserve_last_k_words=max(1, args.stable_reserve_words))

                # Áp commit tạm thời vào soft (không vượt quá SOFT_WINDOW)
                if commit_part:
                    soft_text += commit_part

                # Giới hạn cửa sổ hồi tố: đẩy phần đầu soft sang hard nếu soft quá dài
                # đo theo số từ
                _, soft_excess = _first_k_words(soft_text, max(0, len(re.split(r"(\s+|[.,!?;:…])", soft_text))))  # dummy call
                # tính nhanh số từ bằng regex split theo ranh giới
                def _count_words(s: str) -> int:
                    parts = re.split(r"(\s+|[.,!?;:…])", s)
                    words = []
                    buf = ""
                    for p in parts:
                        buf += p
                        if _WORD_BOUNDARY_RE.fullmatch(p or ""):
                            words.append(buf)
                            buf = ""
                    if buf:
                        words.append(buf)
                    return len(words)

                while _count_words(soft_text) > SOFT_WINDOW:
                    first, remain = _first_k_words(soft_text, _count_words(soft_text) - SOFT_WINDOW)
                    hard_text += first
                    soft_text = remain

                # Tính hiển thị hiện tại
                display_now = f"{hard_text}{soft_text}{new_tail}"

                # Phát hiện stall: không có ký tự mới xuất hiện
                if display_now.strip() == prev_display.strip():
                    no_new_char_chunks += 1
                else:
                    no_new_char_chunks = 0

                # Nếu stall đủ lâu, chuyển một số từ từ tail sang soft để "chốt" mà vẫn hồi tố được
                if no_new_char_chunks >= STALL_N and new_tail:
                    move, remain = _first_k_words(new_tail, SOFT_COMMIT_WORDS)
                    soft_text += move
                    new_tail = remain
                    no_new_char_chunks = 0  # reset sau khi chốt mềm

                # Nếu người nói tiếp tục và có thay đổi đáng kể, rollback một phần soft để cho phép sửa
                # Hàm kích hoạt: có ký tự mới và trước đó từng stall hoặc có im lặng kết thúc
                resumed_speech = (_rms(a) >= SIL_THRESH) and (display_now.strip() != prev_display.strip())
                if resumed_speech and ROLLBACK_WORDS > 0:
                    # kéo lại ROLLBACK_WORDS từ soft quay về tail để tái đánh giá cùng decode mới
                    soft_text, rollback_chunk = _take_last_k_words(soft_text, ROLLBACK_WORDS)
                    if rollback_chunk:
                        # gắn trước tail rồi hợp nhất lại với decode hiện tại ở vòng sau
                        new_tail = rollback_chunk + new_tail

                # Flush khi im lặng dài (bảo toàn)
                if silence_run >= args.silence_runs and new_tail.strip():
                    soft_text += new_tail
                    new_tail = ""

                # In ra
                print(f"\r{hard_text.strip()} {soft_text.strip()} {new_tail.strip()}",
                      end="", flush=True)

                # Cập nhật trạng thái
                carry_text = new_tail
                prev_display = f"{hard_text}{soft_text}{new_tail}"
                q.popleft()
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\nĐã dừng.")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="ChunkFormer Microphone Streaming")

    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--left_context_size", type=int, default=128)
    parser.add_argument("--right_context_size", type=int, default=16)
    parser.add_argument("--mic_sr", type=int, default=16000)
    parser.add_argument("--stream_chunk_sec", type=float, default=0.5)
    parser.add_argument("--lookahead_sec", type=float, default=0.2)
    parser.add_argument("--stable_reserve_words", type=int, default=1)
    parser.add_argument("--silence_rms", type=float, default=0.005)
    parser.add_argument("--silence_runs", type=int, default=3)

    # Tham số cơ chế stall + soft-commit
    parser.add_argument("--stall_n_chunks", type=int, default=3,
                        help="Số chunk liên tiếp không thay đổi để chốt mềm")
    parser.add_argument("--soft_commit_words_per_stall", type=int, default=1,
                        help="Số từ chuyển từ tail sang soft mỗi lần stall")
    parser.add_argument("--soft_window_words", type=int, default=8,
                        help="Cửa sổ tối đa (số từ) có thể hồi tố trong soft")
    parser.add_argument("--soft_rollback_words", type=int, default=3,
                        help="Khi nói tiếp, rollback bấy nhiêu từ từ soft về tail để cho phép sửa")

    args = parser.parse_args()

    print(f"Model Checkpoint: {args.model_checkpoint}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"L/R Context: {args.left_context_size}/{args.right_context_size}")

    model, char_dict = init(args.model_checkpoint)
    stream_mic(args, model, char_dict)


if __name__ == "__main__":
    # ví dụ chạy nhanh
    sys.argv = [
        "stream_mic.py",
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--mic_sr", "16000",
        "--left_context_size", "128",
        "--right_context_size", "16",
        "--stream_chunk_sec", "0.5",
        "--lookahead_sec", "0.5",
        "--stable_reserve_words", "2",
        "--silence_runs", "1",
        "--stall_n_chunks", "2",
        "--soft_commit_words_per_stall", "1",
        "--soft_window_words", "6",
        "--soft_rollback_words", "3",
        "--silence_rms", "0.005"
    ]
    main()
