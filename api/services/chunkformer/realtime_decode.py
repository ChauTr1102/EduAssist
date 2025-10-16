import os, math, argparse, yaml, re, sys, threading, queue
import torch, torchaudio
import numpy as np
from collections import deque
import torchaudio.compliance.kaldi as kaldi

# ========== optional punctuator ==========
try:
    import onnxruntime as ort
    from punctuators.models import PunctCapSegModelONNX
except Exception:
    ort = None
    PunctCapSegModelONNX = None

from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output
from api.private_config import *

# ==================== ANSI ====================
ANSI_RESET  = "\033[0m"
ANSI_DIM    = "\033[2m"
ANSI_REV    = "\033[7m"
ANSI_BLINK  = "\033[5m"

def _supports_ansi():
    return sys.stdout.isatty() and os.environ.get("TERM", "") not in {"dumb", ""}

# ==================== Text merge & stability ====================

def _longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 64) -> int:
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a[-L:] == b[:L]:
            return L
    return 0

def _longest_common_substring(a: str, b: str):
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
    return best

_WORD_BOUNDARY_RE = re.compile(r"[ \t\n\r\f\v.,!?;:…，。！？；：]")

def _split_commit_tail(text: str, reserve_last_k_words: int = 1):
    parts = re.split(r"(\s+|[.,!?;:…])", text)
    words, buf = [], ""
    for p in parts:
        buf += p
        if _WORD_BOUNDARY_RE.fullmatch(p or ""):
            words.append(buf); buf = ""
    if buf: words.append(buf)
    if not words: return "", text
    if len(words) <= reserve_last_k_words:
        return "", "".join(words)
    return "".join(words[:-reserve_last_k_words]), "".join(words[-reserve_last_k_words:])

def _smart_merge(prev_tail: str, cur_text: str, lcs_window=48, lcs_min=12):
    ov = _longest_suffix_prefix_overlap(prev_tail, cur_text, max_k=lcs_window)
    if ov >= lcs_min: return prev_tail + cur_text[ov:]
    a = prev_tail[-lcs_window:]; b = cur_text[:lcs_window]
    L, _ea, eb = _longest_common_substring(a, b)
    if L >= lcs_min: return prev_tail + cur_text[eb:]
    return prev_tail + cur_text

def _adaptive_reserve_words(overlap_ratio: float, base=1, hard=2, thresh=0.35):
    return hard if overlap_ratio < thresh else base

def _split_prefix_last_k_words(text: str, k: int):
    if k <= 0: return text, ""
    parts = re.split(r"(\s+|[.,!?;:…])", text)
    words, buf = [], ""
    for p in parts:
        buf += p
        if _WORD_BOUNDARY_RE.fullmatch(p or ""):
            words.append(buf); buf = ""
    if buf: words.append(buf)
    if len(words) <= k: return "", "".join(words)
    return "".join(words[:-k]), "".join(words[-k:])

def _count_words(s: str) -> int:
    s = s.strip()
    return 0 if not s else len([t for t in re.split(r"\s+", s) if t])

# ==================== Feature extractors ====================

class GPUMel80:
    def __init__(self, device, sr=16000):
        self.device = device
        self.sr = sr
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=400, win_length=400, hop_length=160,
            f_min=0.0, f_max=8000.0, n_mels=80, power=2.0,
            mel_scale="htk", norm=None, center=True,
        ).to(device)

    @torch.no_grad()
    def __call__(self, wav_1xT: torch.Tensor) -> torch.Tensor:
        mel = self.mel(wav_1xT.to(self.device))
        mel = torch.clamp(mel, min=1e-10).log().transpose(1, 2).contiguous()
        return mel

@torch.no_grad()
def _kaldi_fbank_cpu(wav_1xT: torch.Tensor) -> torch.Tensor:
    return kaldi.fbank(
        wav_1xT.cpu(), num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000
    ).unsqueeze(0)

# ==================== Punctuator worker (async, lookahead confirm) ====================

_EOS_RE = re.compile(r"[.!?…。，、！？；：]\s*$")

class _PunctWorker:
    """Non-blocking ONNX punctuator. Last-write-wins."""
    def __init__(self, enabled: bool, model_name: str, providers: str, min_chars: int):
        self.enabled = enabled and (PunctCapSegModelONNX is not None)
        self.min_chars = min_chars
        self._q = queue.Queue(maxsize=1)
        self._res_q = queue.Queue(maxsize=1)
        self._alive = False
        self._model = None
        self._prov = providers
        if not self.enabled: return

        def _boot():
            try:
                self._model = PunctCapSegModelONNX.from_pretrained(model_name)
                prov = ["CPUExecutionProvider"] if self._prov == "cpu" else ["CUDAExecutionProvider","CPUExecutionProvider"]
                for name in dir(self._model):
                    obj = getattr(self._model, name)
                    if ort is not None and isinstance(obj, ort.InferenceSession):
                        obj.set_providers(prov)
            except Exception:
                self.enabled = False
                return
            self._alive = True
            while self._alive:
                try:
                    job = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                job_id, prefix, lastk = job
                seg = lastk
                try:
                    fixed = seg if len(seg.strip()) < self.min_chars else self._model.infer([seg], apply_sbd=False)[0]
                except Exception:
                    fixed = seg
                # drop stale result if queue already has newer
                if not self._res_q.empty():
                    try: self._res_q.get_nowait()
                    except queue.Empty: pass
                self._res_q.put((job_id, prefix, lastk, fixed))

        self._th = threading.Thread(target=_boot, daemon=True)
        self._th.start()

    def stop(self): self._alive = False

    def submit(self, job_id: int, prefix: str, lastk: str):
        if not self.enabled: return
        if not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: pass
        self._q.put((job_id, prefix, lastk))

    def poll(self):
        if not self.enabled: return None
        try: return self._res_q.get_nowait()
        except queue.Empty: return None

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
    model = init_model(config, cfg); model.eval()
    load_checkpoint(model, ckpt)
    model.encoder = model.encoder.cuda()
    model.ctc = model.ctc.cuda()
    symbol_table = read_symbol_table(vocab)
    char_dict = {v: k for k, v in symbol_table.items()}
    return model, char_dict

# ==================== One-line writer ====================

class _LineWriter:
    def __init__(self): self.prev_len = 0
    def write(self, s: str):
        s = s.replace("\n", " ")
        sys.stdout.write("\r" + s)
        extra = self.prev_len - len(s)
        if extra > 0:
            sys.stdout.write(" " * extra)
            sys.stdout.write("\r" + s)
        sys.stdout.flush()
        self.prev_len = len(s)

# ==================== Mic streaming ====================

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
    full_text, carry_text = "", ""
    last_snapshot = ""
    idle_counter = 0

    fe = GPUMel80(device, sr) if args.use_gpu_mel else None
    line = _LineWriter()
    blink = False
    ansi_ok = _supports_ansi()

    # async punctuator
    punct = _PunctWorker(
        enabled=args.punct,
        model_name=args.punct_model,
        providers=args.punct_prov,
        min_chars=args.punct_min_chars
    )
    job_seq = 0
    pending = None  # {'job_id', 'prefix','lastk','fixed','model_ready', 'base_len', 'chunks_waited','words_after','added_eos'}

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block) as stream:
        while True:
            audio_block, _ = stream.read(block)
            a = np.squeeze(audio_block, axis=1).astype(np.float32)
            q.append(a)

            # always render ASAP for ASR realtime feel
            render_now = True

            if len(q) >= 1 + look_blocks:
                seg_np = np.concatenate([q[0]] + list(list(q)[1:1 + look_blocks]))
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * (1 << 15)
                if seg.size(1) >= int(0.025 * sr):
                    x = _kaldi_fbank_cpu(seg).to(device) if fe is None else fe(seg)
                    x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        enc_out, enc_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
                            xs=x, xs_origin_lens=x_len, chunk_size=chunk_size,
                            left_context_size=left_context_size, right_context_size=right_context_size,
                            att_cache=att_cache, cnn_cache=cnn_cache,
                            truncated_context_size=chunk_size, offset=offset
                        )
                        enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :enc_len]
                        if enc_out.size(1) > chunk_size: enc_out = enc_out[:, :chunk_size]
                        offset = offset - enc_len + enc_out.size(1)
                        hyp_step = model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                    chunk_text = get_output([hyp_step], char_dict)[0]

                    merged = _smart_merge(carry_text, chunk_text, lcs_window=args.lcs_window, lcs_min=args.lcs_min)
                    overlap_used = len(carry_text) + len(chunk_text) - len(merged)
                    overlap_ratio = 0.0 if len(chunk_text) == 0 else max(0.0, min(1.0, overlap_used / max(1, len(chunk_text))))
                    reserve_words = _adaptive_reserve_words(overlap_ratio,
                                                            base=args.stable_reserve_words,
                                                            hard=args.stable_reserve_words + 1,
                                                            thresh=args.adaptive_overlap_thresh)
                    commit, new_tail = _split_commit_tail(merged, reserve_last_k_words=reserve_words)

                    progressed = bool(commit) or (new_tail != last_snapshot)
                    idle_counter = 0 if progressed else (idle_counter + 1)
                    last_snapshot = new_tail

                    if commit:
                        full_text += commit

                    punct_char_hit = bool(_EOS_RE.search(new_tail))
                    max_tail_hit = len(new_tail) >= args.max_tail_chars
                    idle_hit = (idle_counter >= args.idle_flush_chunks) and new_tail.strip()

                    commit_tail_now = punct_char_hit or max_tail_hit or idle_hit

                    if commit_tail_now and new_tail.strip():
                        full_text += new_tail
                        carry_text = ""
                        last_snapshot = ""
                        # submit punctuation only on idle
                        if args.punct and idle_hit and punct.enabled:
                            prefix, lastk = _split_prefix_last_k_words(full_text, args.punct_window_words)
                            job_seq += 1
                            pending = {
                                "job_id": job_seq, "prefix": prefix, "lastk": lastk,
                                "fixed": None, "model_ready": False,
                                "base_len": len(prefix) + len(lastk),
                                "chunks_waited": 0, "words_after": 0,
                                "added_eos": False
                            }
                            punct.submit(job_seq, prefix, lastk)
                        idle_counter = 0
                    else:
                        carry_text = new_tail

                q.popleft()

            # poll async punctuator (non-blocking)
            res = punct.poll()
            if res is not None:
                job_id, prefix, lastk, fixed = res
                # ignore stale if no pending or job mismatch
                if pending is not None and job_id == pending["job_id"] and prefix == pending["prefix"] and lastk == pending["lastk"]:
                    pending["fixed"] = fixed
                    pending["model_ready"] = True
                    pending["added_eos"] = (not _EOS_RE.search(lastk)) and bool(_EOS_RE.search(fixed))
                    # if only truecase/comma without EOS, overlay immediately
                    if not pending["added_eos"]:
                        anchor = pending["prefix"] + pending["lastk"]
                        if full_text.startswith(anchor):
                            full_text = pending["prefix"] + fixed + full_text[len(anchor):]
                        else:
                            idx = full_text.rfind(pending["lastk"])
                            if idx != -1:
                                full_text = full_text[:idx] + fixed + full_text[idx+len(pending["lastk"]):]
                            else:
                                full_text = pending["prefix"] + fixed
                        pending = None

            # confirm pending with lookahead while printing continues
            if pending is not None:
                pending["chunks_waited"] += 1
                all_text = full_text + carry_text
                extra = all_text[pending["base_len"]:]
                pending["words_after"] = _count_words(extra)
                # soft triggers
                force_by_words  = pending["words_after"] >= args.punct_lookahead_words
                force_by_chunks = pending["chunks_waited"] >= args.punct_lookahead_max_chunks
                # hard trigger: another idle detected (safe end)
                another_idle = (idle_counter >= args.idle_flush_chunks) and carry_text.strip()
                if pending["model_ready"] and (force_by_words or force_by_chunks or another_idle):
                    anchor = pending["prefix"] + pending["lastk"]
                    fixed = pending["fixed"]
                    if full_text.startswith(anchor):
                        full_text = pending["prefix"] + fixed + full_text[len(anchor):]
                    else:
                        idx = full_text.rfind(pending["lastk"])
                        if idx != -1:
                            full_text = full_text[:idx] + fixed + full_text[idx+len(pending["lastk"]):]
                        else:
                            full_text = pending["prefix"] + fixed
                    pending = None

            # render ASAP, never blocked by punctuation
            if True:
                if args.show_tail:
                    if ansi_ok and args.twinkle_tail and carry_text:
                        deco = (ANSI_BLINK if blink else ANSI_REV) + ANSI_DIM
                        disp = (full_text + f"{deco}{carry_text}{ANSI_RESET}").strip()
                    else:
                        disp = (full_text + carry_text).strip()
                else:
                    disp = full_text.strip()
                line.write(disp)
                blink = not blink

# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="ChunkFormer streaming: realtime ASR print, async punctuation with lookahead overlay")

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

    # text stability
    parser.add_argument("--stable_reserve_words", type=int, default=1)
    parser.add_argument("--adaptive_overlap_thresh", type=float, default=0.35)
    parser.add_argument("--lcs_window", type=int, default=48)
    parser.add_argument("--lcs_min", type=int, default=12)

    parser.add_argument("--idle_flush_chunks", type=int, default=5)
    parser.add_argument("--max_tail_chars", type=int, default=40)

    # GPU mel
    parser.add_argument("--use_gpu_mel", action="store_true")

    # display
    parser.add_argument("--show_tail", action="store_true")
    parser.add_argument("--twinkle_tail", action="store_true")

    # punctuator options (async)
    parser.add_argument("--punct", action="store_true")
    parser.add_argument("--punct_model", type=str, default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
    parser.add_argument("--punct_prov", type=str, choices=["cpu","cuda"], default="cpu")
    parser.add_argument("--punct_window_words", type=int, default=60)
    parser.add_argument("--punct_min_chars", type=int, default=40)

    # lookahead confirmation for EOS
    parser.add_argument("--punct_lookahead_words", type=int, default=3,
                        help="Số từ mới cần thấy sau điểm chấm câu dự kiến để xác nhận.")
    parser.add_argument("--punct_lookahead_max_chunks", type=int, default=6,
                        help="Số vòng xử lý tối đa để xác nhận nếu người nói ngắt quãng dài.")

    args = parser.parse_args()
    assert any([getattr(args, "mic", False), args.long_form_audio, args.audio_list]), "Cần --mic hoặc --long_form_audio hoặc --audio_list"

    model, char_dict = init(args.model_checkpoint)
    if getattr(args, "mic", False):
        stream_mic(args, model, char_dict); return

if __name__ == "__main__":
    sys.argv = [
        "realtime_decode.py",
        "--model_checkpoint", CHUNKFORMER_CHECKPOINT,
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
        "--idle_flush_chunks", "1",
        "--max_tail_chars", "40",
        "--punct",
        "--punct_prov", "cpu",
        "--punct_lookahead_words", "3",
        "--punct_lookahead_max_chunks", "6",
        # "--show_tail",
        "--twinkle_tail",
    ]
    main()
