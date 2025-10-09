import os, math, argparse, yaml, re
import torch, torchaudio
import numpy as np
from collections import deque
import torchaudio.compliance.kaldi as kaldi

from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output, milliseconds_to_hhmmssms


# ==================== Text merge & stability ====================

def _longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 64) -> int:
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a[-L:] == b[:L]:
            return L
    return 0

def _longest_common_substring(a: str, b: str):
    """LCSubstr trên cửa sổ nhỏ, trả (len, end_a, end_b). O(mn) với m,n<=64."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    best = (0, 0, 0)
    for i in range(1, m+1):
        ai = a[i-1]
        for j in range(1, n+1):
            if ai == b[j-1]:
                v = dp[i-1][j-1] + 1
                dp[i][j] = v
                if v > best[0]:
                    best = (v, i, j)
    return best  # length, end_a_idx, end_b_idx

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
    return "".join(words[:-reserve_last_k_words]), "".join(words[-reserve_last_k_words:])

def _smart_merge(prev_tail: str, cur_text: str, lcs_window=48, lcs_min=12):
    ov = _longest_suffix_prefix_overlap(prev_tail, cur_text, max_k=lcs_window)
    if ov >= lcs_min:
        return prev_tail + cur_text[ov:]
    a = prev_tail[-lcs_window:]
    b = cur_text[:lcs_window]
    L, _ea, eb = _longest_common_substring(a, b)
    if L >= lcs_min:
        return prev_tail + cur_text[eb:]  # bỏ phần trùng
    return prev_tail + cur_text

def _adaptive_reserve_words(overlap_ratio: float, base=1, hard=2, thresh=0.35):
    return hard if overlap_ratio < thresh else base

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


# ==================== Feature extractors ====================

class GPUMel80:
    def __init__(self, device, sr=16000):
        self.device = device
        self.sr = sr
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=400, win_length=400, hop_length=160,
            f_min=0.0, f_max=8000.0,
            n_mels=80, power=2.0,
            mel_scale="htk", norm=None, center=True,
        ).to(device)

    @torch.no_grad()
    def __call__(self, wav_1xT: torch.Tensor) -> torch.Tensor:
        # wav in 1xT, range like Kaldi (scaled), we just compute log-mel
        mel = self.mel(wav_1xT.to(self.device))  # (1, 80, Tm)
        mel = torch.clamp(mel, min=1e-10).log().transpose(1, 2).contiguous()  # (1, Tm, 80)
        return mel

@torch.no_grad()
def _kaldi_fbank_cpu(wav_1xT: torch.Tensor) -> torch.Tensor:
    return kaldi.fbank(
        wav_1xT.cpu(),
        num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000
    ).unsqueeze(0)  # (1, T, 80)


# ==================== Model init ====================

@torch.no_grad()
def init(model_checkpoint):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass

    cfg = os.path.join(model_checkpoint, "config.yaml")
    ckpt = os.path.join(model_checkpoint, "pytorch_model.bin")
    vocab = os.path.join(model_checkpoint, "vocab.txt")

    with open(cfg, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(config, cfg)
    model.eval()
    load_checkpoint(model, ckpt)

    model.encoder = model.encoder.cuda()
    model.ctc = model.ctc.cuda()

    symbol_table = read_symbol_table(vocab)
    char_dict = {v: k for k, v in symbol_table.items()}
    return model, char_dict


# ==================== Microphone streaming ====================

@torch.no_grad()
def stream_mic(args, model, char_dict):
    import sounddevice as sd

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
    block = int(args.stream_chunk_sec * sr)
    look_blocks = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q = deque()
    produced_steps = 0
    carry_text = ""
    silence_run = 0

    fe = GPUMel80(device, sr) if args.use_gpu_mel else None

    idle_flush_chunks = args.idle_flush_chunks
    max_tail_chars = args.max_tail_chars
    punct_flush = args.punct_flush
    idle_counter = 0  # đếm số chunk không tiến triển

    print("Mic streaming. Ctrl+C để dừng.")
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block) as stream:
        while True:
            audio_block, _ = stream.read(block)
            a = np.squeeze(audio_block, axis=1).astype(np.float32)
            q.append(a)

            if len(q) < 1 + look_blocks:
                continue

            seg_np = np.concatenate([q[0]] + list(list(q)[1:1 + look_blocks]))
            seg = torch.from_numpy(seg_np).unsqueeze(0).to(device)
            seg = seg * (1 << 15)
            if seg.size(1) < int(0.025 * sr):
                q.popleft()
                continue

            x = _kaldi_fbank_cpu(seg).to(device) if fe is None else fe(seg)
            x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                enc_out, enc_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
                    xs=x, xs_origin_lens=x_len,
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                    right_context_size=right_context_size,
                    att_cache=att_cache, cnn_cache=cnn_cache,
                    truncated_context_size=chunk_size, offset=offset
                )
                enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :enc_len]
                if enc_out.size(1) > chunk_size:
                    enc_out = enc_out[:, :chunk_size]
                offset = offset - enc_len + enc_out.size(1)
                hyp_step = model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

            seg_start_ms = int(produced_steps * subsampling * 10)
            seg_end_ms = int((produced_steps + hyp_step.numel()) * subsampling * 10)
            produced_steps += hyp_step.numel()

            chunk_text = get_output([hyp_step], char_dict)[0]

            # merge
            merged = _smart_merge(carry_text, chunk_text, lcs_window=args.lcs_window, lcs_min=args.lcs_min)
            overlap_used = len(carry_text) + len(chunk_text) - len(merged)
            overlap_ratio = 0.0 if len(chunk_text) == 0 else max(0.0, min(1.0, overlap_used / max(1, len(chunk_text))))
            reserve_words = _adaptive_reserve_words(overlap_ratio, base=args.stable_reserve_words,
                                                    hard=args.stable_reserve_words + 1,
                                                    thresh=args.adaptive_overlap_thresh)
            commit, new_tail = _split_commit_tail(merged, reserve_last_k_words=reserve_words)

            progressed = bool(commit) or (new_tail != carry_text)  # có thay đổi thực sự?
            if progressed:
                idle_counter = 0
            else:
                idle_counter += 1

            # in phần commit nếu có
            if commit:
                print(
                    f"{milliseconds_to_hhmmssms(seg_start_ms)} - {milliseconds_to_hhmmssms(seg_end_ms)}: {commit.strip()}")

            # các điều kiện ép commit đuôi
            do_force = False
            if punct_flush and re.search(r"[.!?…。，、！？；：]\s*$", new_tail):
                do_force = True
            if len(new_tail) >= max_tail_chars:
                do_force = True
            if idle_counter >= idle_flush_chunks and new_tail.strip():
                do_force = True

            if do_force and new_tail.strip():
                print(
                    f"{milliseconds_to_hhmmssms(seg_start_ms)} - {milliseconds_to_hhmmssms(seg_end_ms)}: {new_tail.strip()}")
                new_tail = ""
                idle_counter = 0

            carry_text = new_tail
            q.popleft()

# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="ChunkFormer streaming without CIF (stabilized)")

    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--max_duration", type=int, default=1800)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--left_context_size", type=int, default=128)
    parser.add_argument("--right_context_size", type=int, default=16)

    parser.add_argument("--long_form_audio", type=str)
    parser.add_argument("--audio_list", type=str, default=None)

    parser.add_argument("--mic", action="store_true")
    parser.add_argument("--mic_sr", type=int, default=16000)
    parser.add_argument("--stream_chunk_sec", type=float, default=0.5)
    parser.add_argument("--lookahead_sec", type=float, default=0.2)
    parser.add_argument("--print_final", action="store_true")
    parser.add_argument("--stream", action="store_true")

    # ổn định văn bản
    parser.add_argument("--stable_reserve_words", type=int, default=1)
    parser.add_argument("--adaptive_overlap_thresh", type=float, default=0.35)
    parser.add_argument("--lcs_window", type=int, default=48)
    parser.add_argument("--lcs_min", type=int, default=12)

    # im lặng để flush tail
    parser.add_argument("--silence_rms", type=float, default=0.005)
    parser.add_argument("--silence_runs", type=int, default=3)

    parser.add_argument("--idle_flush_chunks", type=int, default=5)
    parser.add_argument("--max_tail_chars", type=int, default=40)
    parser.add_argument("--punct_flush", action="store_true")

    # GPU mel
    parser.add_argument("--use_gpu_mel", action="store_true")

    args = parser.parse_args()

    print(f"Model: {args.model_checkpoint}")
    print(f"Stream chunk: {args.stream_chunk_sec}s | lookahead: {args.lookahead_sec}s | L/R ctx: {args.left_context_size}/{args.right_context_size}")
    print(f"GPU Mel: {args.use_gpu_mel}")

    assert any([getattr(args, "mic", False), args.long_form_audio, args.audio_list]), "Cần --mic hoặc --long_form_audio hoặc --audio_list"

    model, char_dict = init(args.model_checkpoint)

    if getattr(args, "mic", False):
        stream_mic(args, model, char_dict); return


if __name__ == "__main__":
    import sys
    # ví dụ chạy mic
    import sys

    sys.argv = [
        "realtime_decode.py",
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--mic",
        "--mic_sr", "16000",
        "--left_context_size", "16",
        "--right_context_size", "8",
        "--stream_chunk_sec", "0.5",
        "--lookahead_sec", "0.2",
        "--stable_reserve_words", "0",
        "--lcs_window", "64",
        "--lcs_min", "16",
        "--use_gpu_mel",
        "--idle_flush_chunks", "2",  # ép commit nếu 5 chunk (~1s) không tiến triển
        "--max_tail_chars", "40",  # commit nếu đuôi vượt 40 ký tự
        "--punct_flush"  # commit khi kết thúc bằng dấu câu
    ]
main()