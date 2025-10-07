import os, math, argparse, yaml, re
import torch, torchaudio, pandas as pd, jiwer
import numpy as np
from collections import deque
from tqdm import tqdm
from colorama import Fore, Style
import torchaudio.compliance.kaldi as kaldi

from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output_with_timestamps, get_output, milliseconds_to_hhmmssms


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


# ==================== Streaming from file ====================

@torch.no_grad()
def stream_audio(args, model, char_dict):
    device = torch.device("cuda")
    subsampling = model.encoder.embed.subsampling_factor
    num_layers = model.encoder.num_blocks
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # encoder steps từ thời lượng khối
    enc_steps = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size = enc_steps

    left_context_size = args.left_context_size
    right_context_size = args.right_context_size

    # cache
    att_cache = torch.zeros(
        (num_layers, left_context_size, model.encoder.attention_heads,
         model.encoder._output_size * 2 // model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros((num_layers, model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    # audio
    wav_path = args.long_form_audio
    waveform, sr = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    waveform = waveform * (1 << 15)

    hop = int(args.stream_chunk_sec * sr)
    look = int(args.lookahead_sec * sr)

    produced_steps = 0
    carry_text = ""
    last_chunk_len = 1

    # feature
    fe = GPUMel80(device, sr) if args.use_gpu_mel else None

    cur = 0
    while cur < waveform.size(1):
        end = min(cur + hop, waveform.size(1))
        end_la = min(end + look, waveform.size(1))
        seg = waveform[:, cur:end_la]

        if seg.size(1) < int(0.025 * sr):
            break

        if fe is None:
            x = _kaldi_fbank_cpu(seg).to(device)
        else:
            x = fe(seg.to(device))
        x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

        truncated_context_size = chunk_size
        with torch.cuda.amp.autocast(dtype=torch.float16):
            enc_out, enc_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
                xs=x,
                xs_origin_lens=x_len,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                truncated_context_size=truncated_context_size,
                offset=offset
            )
            enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :enc_len]
            if enc_out.size(1) > truncated_context_size:
                enc_out = enc_out[:, :truncated_context_size]
            offset = offset - enc_len + enc_out.size(1)
            hyp_step = model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

        seg_start_ms = int(produced_steps * subsampling * 10)
        seg_end_ms = int((produced_steps + hyp_step.numel()) * subsampling * 10)
        produced_steps += hyp_step.numel()

        chunk_text = get_output([hyp_step], char_dict)[0]

        # merge thông minh
        merged = _smart_merge(carry_text, chunk_text, lcs_window=args.lcs_window, lcs_min=args.lcs_min)

        # dự trữ đuôi thích ứng
        overlap_used = len(carry_text) + len(chunk_text) - len(merged)
        overlap_ratio = 0.0 if len(chunk_text) == 0 else max(0.0, min(1.0, overlap_used / max(1, len(chunk_text))))
        reserve_words = _adaptive_reserve_words(overlap_ratio, base=args.stable_reserve_words, hard=args.stable_reserve_words+1, thresh=args.adaptive_overlap_thresh)

        commit, new_tail = _split_commit_tail(merged, reserve_last_k_words=reserve_words)

        if commit:
            print(f"{Fore.CYAN}{milliseconds_to_hhmmssms(seg_start_ms)}{Style.RESET_ALL} - {Fore.CYAN}{milliseconds_to_hhmmssms(seg_end_ms)}{Style.RESET_ALL}: {commit.strip()}")

        carry_text = new_tail
        last_chunk_len = max(1, len(chunk_text))
        cur = end

    if args.print_final:
        print(carry_text.strip())


# ==================== Endless decode (giữ để so sánh) ====================

@torch.no_grad()
def endless_decode(args, model, char_dict):
    wav_path = args.long_form_audio
    subsampling_factor = model.encoder.embed.subsampling_factor
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    conv_lorder = model.encoder.cnn_module_kernel // 2

    max_length_limited_context = args.max_duration
    max_length_limited_context = int((max_length_limited_context // 0.01)) // 2  # 10ms frame

    multiply_n = max_length_limited_context // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n

    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n - 1)

    rel_right_context_size = get_max_input_context(chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks)
    rel_right_context_size = rel_right_context_size * subsampling_factor

    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform * (1 << 15)
    offset = torch.zeros(1, dtype=torch.int, device="cuda")

    xs = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000
    ).unsqueeze(0)

    hyps = []
    att_cache = torch.zeros((model.encoder.num_blocks, left_context_size, model.encoder.attention_heads, model.encoder._output_size * 2 // model.encoder.attention_heads)).cuda()
    cnn_cache = torch.zeros((model.encoder.num_blocks, model.encoder._output_size, conv_lorder)).cuda()

    for idx, _ in enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)):
        start = max(truncated_context_size * subsampling_factor * idx, 0)
        end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs.shape[1])

        x = xs[:, start:end + rel_right_context_size]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).cuda()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            enc_out, enc_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
                xs=x,
                xs_origin_lens=x_len,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                truncated_context_size=truncated_context_size,
                offset=offset
            )
        enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :enc_len]
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
            enc_out = enc_out[:, :truncated_context_size]
        offset = offset - enc_len + enc_out.size(1)

        hyp = model.encoder.ctc_forward(enc_out).squeeze(0)
        hyps.append(hyp)

        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
            break

    hyps = torch.cat(hyps)
    decode = get_output_with_timestamps([hyps], char_dict)[0]
    for item in decode:
        start = f"{Fore.RED}{item['start']}{Style.RESET_ALL}"
        end = f"{Fore.RED}{item['end']}{Style.RESET_ALL}"
        print(f"{start} - {end}: {item['decode']}")


# ==================== Batch decode (gần như giữ nguyên) ====================

@torch.no_grad()
def batch_decode(args, model, char_dict):
    df = pd.read_csv(args.audio_list, sep="\t")
    max_length_limited_context = args.max_duration
    max_length_limited_context = int((max_length_limited_context // 0.01)) // 2
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size

    decodes, xs, xs_origin_lens = [], [], []
    max_frames = max_length_limited_context

    for idx, wav_path in tqdm(enumerate(df['wav'].to_list())):
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform * (1 << 15)
        x = kaldi.fbank(
            waveform,
            num_mel_bins=80, frame_length=25, frame_shift=10,
            dither=0.0, energy_floor=0.0, sample_frequency=16000
        )
        xs.append(x)
        xs_origin_lens.append(x.shape[0])
        max_frames -= xs_origin_lens[-1]

        if (max_frames <= 0) or (idx == len(df) - 1):
            xs_origin_lens = torch.tensor(xs_origin_lens, dtype=torch.int, device="cuda")
            offset = torch.zeros(len(xs), dtype=torch.int, device="cuda")
            with torch.cuda.amp.autocast(dtype=torch.float16):
                enc_out, enc_len, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(
                    xs=xs,
                    xs_origin_lens=xs_origin_lens,
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                    right_context_size=right_context_size,
                    offset=offset
                )
            hyps = model.encoder.ctc_forward(enc_out, enc_len, n_chunks)
            decodes += get_output(hyps, char_dict)

            xs, xs_origin_lens, max_frames = [], [], max_length_limited_context

    df['decode'] = decodes
    if "txt" in df:
        print("WER:", jiwer.wer(df["txt"].to_list(), decodes))
    df.to_csv(args.audio_list, sep="\t", index=False)


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

    print("Mic streaming. Ctrl+C để dừng.")
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block) as stream:
        while True:
            audio_block, _ = stream.read(block)
            a = np.squeeze(audio_block, axis=1).astype(np.float32)
            q.append(a)

            # im lặng để flush
            silence_run = silence_run + 1 if _rms(a) < args.silence_rms else 0

            if len(q) < 1 + look_blocks:
                continue

            # ghép block + lookahead
            seg_np = np.concatenate([q[0]] + list(list(q)[1:1+look_blocks]))
            seg = torch.from_numpy(seg_np).unsqueeze(0).to(device)
            seg = seg * (1 << 15)
            if seg.size(1) < int(0.025 * sr):
                q.popleft()
                continue

            if fe is None:
                x = _kaldi_fbank_cpu(seg).to(device)
            else:
                x = fe(seg)
            x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                truncated_context_size = chunk_size
                enc_out, enc_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
                    xs=x,
                    xs_origin_lens=x_len,
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                    right_context_size=right_context_size,
                    att_cache=att_cache,
                    cnn_cache=cnn_cache,
                    truncated_context_size=truncated_context_size,
                    offset=offset
                )
                enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :enc_len]
                if enc_out.size(1) > truncated_context_size:
                    enc_out = enc_out[:, :truncated_context_size]
                offset = offset - enc_len + enc_out.size(1)

                hyp_step = model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

            seg_start_ms = int(produced_steps * subsampling * 10)
            seg_end_ms = int((produced_steps + hyp_step.numel()) * subsampling * 10)
            produced_steps += hyp_step.numel()

            chunk_text = get_output([hyp_step], char_dict)[0]

            merged = _smart_merge(carry_text, chunk_text, lcs_window=args.lcs_window, lcs_min=args.lcs_min)
            overlap_used = len(carry_text) + len(chunk_text) - len(merged)
            overlap_ratio = 0.0 if len(chunk_text) == 0 else max(0.0, min(1.0, overlap_used / max(1, len(chunk_text))))
            reserve_words = _adaptive_reserve_words(overlap_ratio, base=args.stable_reserve_words, hard=args.stable_reserve_words+1, thresh=args.adaptive_overlap_thresh)

            commit, new_tail = _split_commit_tail(merged, reserve_last_k_words=reserve_words)

            if commit:
                print(f"{milliseconds_to_hhmmssms(seg_start_ms)} - {milliseconds_to_hhmmssms(seg_end_ms)}: {commit.strip()}")

            if silence_run >= args.silence_runs and new_tail.strip():
                print(f"{milliseconds_to_hhmmssms(seg_start_ms)} - {milliseconds_to_hhmmssms(seg_end_ms)}: {new_tail.strip()}")
                new_tail = ""

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
    if args.stream and args.long_form_audio:
        stream_audio(args, model, char_dict)
    elif args.long_form_audio:
        endless_decode(args, model, char_dict)
    else:
        batch_decode(args, model, char_dict)


if __name__ == "__main__":
    import sys
    # ví dụ chạy mic
    sys.argv = [
        "realtime_decode.py",
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--mic",
        "--mic_sr", "16000",
        "--left_context_size", "64",
        "--right_context_size", "16",
        "--stream_chunk_sec", "0.5",
        "--lookahead_sec", "0.25",
        "--stable_reserve_words", "1",
        "--lcs_window", "48",
        "--lcs_min", "12",
        "--use_gpu_mel"
    ]
    main()
