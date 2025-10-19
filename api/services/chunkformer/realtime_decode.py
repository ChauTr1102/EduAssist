# realtime_asr_punct.py
# ASR ChunkFormer streaming + ONNX punctuation truecasing overlay (song song, nhấp-nháy)

import os, re, sys, math, yaml, time, threading, queue
import numpy as np
from collections import deque

import torch, torchaudio
import torchaudio.compliance.kaldi as kaldi

# ====== your project imports (giữ nguyên) ======
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output
from api.private_config import CHUNKFORMER_CHECKPOINT

# ====== optional punctuation ONNX ======
try:
    import onnxruntime as ort
    from punctuators.models import PunctCapSegModelONNX

    _HAS_PUNCT = True
except Exception:
    _HAS_PUNCT = False

# ==================== ANSI helpers (nhấp-nháy/nhạt màu phần đuôi) ====================
ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_REV = "\033[7m"
ANSI_BLINK = "\033[5m"


def _supports_ansi():
    return sys.stdout.isatty()


# ==================== Text merge & stability (giữ của bạn) ====================
_WORD_BOUNDARY_RE = re.compile(r"[ \t\n\r\f\v.,!?;:…，。！？；：]")


def _longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 64) -> int:
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a[-L:] == b[:L]:
            return L
    return 0


def _longest_common_substring(a: str, b: str):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = (0, 0, 0)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                v = dp[i - 1][j - 1] + 1
                dp[i][j] = v
                if v > best[0]:
                    best = (v, i, j)
    return best  # length, end_a_idx, end_b_idx


def _split_commit_tail(text: str, reserve_last_k_words: int = 1):
    parts = re.split(r"(\s+|[.,!?;:…])", text)
    words = []
    buf = ""
    for p in parts:
        buf += p
        if _WORD_BOUNDARY_RE.fullmatch(p or ""):
            words.append(buf);
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
        return prev_tail + cur_text[eb:]
    return prev_tail + cur_text


def _adaptive_reserve_words(overlap_ratio: float, base=1, hard=2, thresh=0.35):
    return hard if overlap_ratio < thresh else base


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
        mel = self.mel(wav_1xT.to(self.device))
        mel = torch.clamp(mel, min=1e-10).log().transpose(1, 2).contiguous()
        return mel  # (1, Tm, 80)


@torch.no_grad()
def _kaldi_fbank_cpu(wav_1xT: torch.Tensor) -> torch.Tensor:
    return kaldi.fbank(
        wav_1xT.cpu(),
        num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000
    ).unsqueeze(0)


# ==================== Model init ====================
@torch.no_grad()
def init_asr(model_checkpoint):
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


# ==================== Punctuation worker (song song) ====================
class PunctWorker(threading.Thread):
    def __init__(self,
                 model_id="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
                 device="cuda", providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                 rate_hz=6.0,
                 window_chars=640,  # Tham số này không còn tác dụng trực tiếp nhưng vẫn giữ để tương thích
                 reserve_words=1,
                 apply_sbd=False,
                 ansi_dim_tail=True,
                 ansi_blink=False):
        super().__init__(daemon=True)
        self.enabled = _HAS_PUNCT
        self.queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.rate_hz = rate_hz
        self.window_chars = window_chars  # Giữ lại để không phá vỡ CLI
        self.reserve_words = max(0, reserve_words)
        self.apply_sbd = apply_sbd
        self.ansi_dim_tail = ansi_dim_tail
        self.ansi_blink = ansi_blink and _supports_ansi()
        self._last_emit = 0.0
        self._overlay = ""  # chuỗi đã gắn dấu để hiển thị
        self._have_overlay = False

        if self.enabled:
            # load model
            self.m = PunctCapSegModelONNX.from_pretrained(model_id)
            # đặt providers ưu tiên GPU rồi CPU
            prov = list(providers) if device == "cuda" else ["CPUExecutionProvider"]
            for name in dir(self.m):
                obj = getattr(self.m, name)
                if isinstance(obj, ort.InferenceSession):
                    try:
                        obj.set_providers(prov)
                    except Exception:
                        pass
        else:
            self.m = None

    def submit(self, raw_text: str):
        if not self.enabled:
            return

        # drop older items in queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        try:
            # put the full raw text
            self.queue.put_nowait(raw_text)
        except queue.Full:
            pass

    def get_display(self, fallback: str) -> str:
        if self._have_overlay:
            # Tách phần đã commit và phần đuôi từ chuỗi đã được xử lý
            commit, tail = _split_commit_tail(self._overlay, self.reserve_words)
            decorated_tail = self._decorate_tail(tail)
            return commit, decorated_tail

        commit, tail = _split_commit_tail(fallback, self.reserve_words)
        return commit, self._decorate_tail(tail)  # Thêm decorate cho fallback

    def _decorate_tail(self, text: str) -> str:
        # Làm nhạt/nhấp-nháy phần đuôi chưa commit
        if not text or not _supports_ansi() or not self.ansi_dim_tail:
            return text
        deco = text
        if self.ansi_blink:
            deco = ANSI_BLINK + deco + ANSI_RESET
        else:
            deco = ANSI_DIM + deco + ANSI_RESET
        return deco

    def run(self):
        if not self.enabled:
            return
        min_interval = 1.0 / max(1e-6, self.rate_hz)
        while not self.stop_event.is_set():
            try:
                raw = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            now = time.time()
            if now - self._last_emit < min_interval:
                time.sleep(min_interval - (now - self._last_emit))
            self._last_emit = time.time()

            # ONNX infer
            try:
                # <<< SỬA LỖI: Luôn xử lý toàn bộ văn bản `raw` >>>
                # Bỏ việc cắt bớt `infer_text` để giữ toàn bộ ngữ cảnh.
                out_list = self.m.infer([raw], apply_sbd=self.apply_sbd)

                if isinstance(out_list, list) and out_list:
                    pred = out_list[0]
                    if isinstance(pred, dict) and "text" in pred:
                        pred = pred["text"]
                    elif not isinstance(pred, str):
                        pred = str(pred)
                else:
                    pred = str(out_list)

                # <<< SỬA LỖI: Gán trực tiếp kết quả đã xử lý >>>
                # Không cần ghép nối với prefix thô nữa vì `pred` đã là kết quả
                # của toàn bộ văn bản.
                self._overlay = pred

            except Exception:
                self._overlay = raw  # fallback to raw text on error

            self._have_overlay = True

    def stop(self):
        self.stop_event.set()


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
    full_text = ""  # đã commit
    carry_text = ""  # đuôi chưa commit
    last_snapshot = ""
    idle_counter = 0

    fe = GPUMel80(device, sr) if args.use_gpu_mel else None

    # khởi động worker dấu câu nếu bật
    punct = None
    if args.enable_punct and _HAS_PUNCT:
        punct = PunctWorker(
            model_id=args.punct_model_id,
            device=("cuda" if args.punct_device == "gpu" else "cpu"),
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            rate_hz=args.punct_rate_hz,
            window_chars=args.punct_window_chars,
            reserve_words=args.punct_reserve_words,
            apply_sbd=args.punct_apply_sbd,
            ansi_dim_tail=args.ansi_dim_tail,
            ansi_blink=args.ansi_blink_tail
        )
        punct.start()

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block) as stream:
            while True:
                audio_block, _ = stream.read(block)
                a = np.squeeze(audio_block, axis=1).astype(np.float32)
                q.append(a)

                snapshot_raw = full_text + carry_text

                if len(q) < 1 + look_blocks:
                    if punct: punct.submit(snapshot_raw)
                    # Cập nhật hiển thị ngay cả khi chưa có audio mới để hiệu ứng nhấp nháy hoạt động
                    commit_display, tail_display = punct.get_display(snapshot_raw)

                    # Xóa dòng hiện tại và in lại
                    sys.stdout.write("\r\033[K")
                    display_text = commit_display + tail_display if args.show_tail else commit_display
                    sys.stdout.write(display_text.strip())
                    sys.stdout.flush()
                    continue

                seg_np = np.concatenate([q[0]] + list(list(q)[1:1 + look_blocks]))
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device)
                seg = seg * (1 << 15)
                if seg.size(1) < int(0.025 * sr):
                    q.popleft();
                    continue

                x = _kaldi_fbank_cpu(seg).to(device) if fe is None else fe(seg)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
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

                chunk_text = get_output([hyp_step], char_dict)[0]

                merged = _smart_merge(carry_text, chunk_text, lcs_window=args.lcs_window, lcs_min=args.lcs_min)
                overlap_used = len(carry_text) + len(chunk_text) - len(merged)
                overlap_ratio = 0.0 if len(chunk_text) == 0 else max(0.0,
                                                                     min(1.0, overlap_used / max(1, len(chunk_text))))
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

                do_force = False
                if args.punct_flush and re.search(r"[.!?…。，、！？；：]\s*$", new_tail):
                    do_force = True
                if len(new_tail) >= args.max_tail_chars:
                    do_force = True
                if idle_counter >= args.idle_flush_chunks and new_tail.strip():
                    do_force = True

                if do_force and new_tail.strip():
                    full_text += new_tail
                    new_tail = ""
                    idle_counter = 0

                carry_text = new_tail

                snapshot_raw = full_text + carry_text
                if punct: punct.submit(snapshot_raw)

                commit_display_punct, tail_display_punct = punct.get_display(snapshot_raw)

                # Xóa toàn bộ dòng cũ để tránh ký tự rác
                sys.stdout.write("\r\033[K")

                # In ra văn bản hoàn chỉnh, terminal sẽ tự động xuống dòng khi cần
                final_display = commit_display_punct + tail_display_punct if args.show_tail else commit_display_punct
                sys.stdout.write(final_display.strip())
                sys.stdout.flush()

                q.popleft()
    finally:
        if punct:
            punct.stop()
        print("\nStreaming finished.")  # In ra dòng mới khi kết thúc


# ==================== CLI ====================
def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="ChunkFormer streaming + realtime punctuation overlay")
    # ASR
    p.add_argument("--model_checkpoint", type=str, required=True)
    p.add_argument("--left_context_size", type=int, default=16)
    p.add_argument("--right_context_size", type=int, default=8)
    p.add_argument("--mic", action="store_true")
    p.add_argument("--mic_sr", type=int, default=16000)
    p.add_argument("--stream_chunk_sec", type=float, default=0.5)
    p.add_argument("--lookahead_sec", type=float, default=0.2)
    # ổn định văn bản
    p.add_argument("--stable_reserve_words", type=int, default=0)
    p.add_argument("--adaptive_overlap_thresh", type=float, default=0.35)
    p.add_argument("--lcs_window", type=int, default=64)
    p.add_argument("--lcs_min", type=int, default=16)
    p.add_argument("--idle_flush_chunks", type=int, default=1)
    p.add_argument("--max_tail_chars", type=int, default=40)
    p.add_argument("--punct_flush", action="store_true")
    p.add_argument("--use_gpu_mel", action="store_true")
    p.add_argument("--show_tail", action="store_true", default=True)
    # punctuation worker
    p.add_argument("--enable_punct", action="store_true")
    p.add_argument("--punct_model_id", type=str,
                   default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
    p.add_argument("--punct_device", choices=["gpu", "cpu"], default="gpu",
                   help="Nếu 1 GPU chung với ASR bị tranh tài nguyên, đặt cpu.")
    p.add_argument("--punct_rate_hz", type=float, default=6.0)
    p.add_argument("--punct_window_chars", type=int, default=640)
    p.add_argument("--punct_reserve_words", type=int, default=1)
    p.add_argument("--punct_apply_sbd", action="store_true",
                   help="Bật nếu muốn worker tự ngắt câu theo mô hình.")
    # ANSI hiển thị
    p.add_argument("--ansi_dim_tail", action="store_true")
    p.add_argument("--ansi_blink_tail", action="store_true")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    assert getattr(args, "mic", False), "Chế độ này cần --mic"
    model, char_dict = init_asr(args.model_checkpoint)
    stream_mic(args, model, char_dict)


if __name__ == "__main__":
    # Cấu hình ưu tiên độ ổn định và chính xác
    sys.argv = [
        "realtime_asr_punct.py",
        "--model_checkpoint", CHUNKFORMER_CHECKPOINT,
        "--mic",
        "--mic_sr", "16000",

        # Cân bằng giữa độ trễ và độ chính xác
        "--stream_chunk_sec", "0.5",  # Tăng nhẹ để có thêm ngữ cảnh
        "--lookahead_sec", "0.2",
        "--left_context_size", "32",
        "--right_context_size", "16",

        # Cài đặt để văn bản "chốt" nhanh và ít thay đổi
        "--stable_reserve_words", "0",
        "--idle_flush_chunks", "0",  # Chốt khi ngừng nói
        "--max_tail_chars", "40",
        # "--punct_flush",  # Chốt khi có dấu câu cuối câu

        # Tham số kỹ thuật
        "--lcs_window", "64",
        "--lcs_min", "16",
        "--use_gpu_mel",

        # Bật và cấu hình mô hình dấu câu
        "--enable_punct",
        "--punct_device", "gpu",  # An toàn, tránh tranh chấp tài nguyên GPU
        "--punct_rate_hz", "5.0",  # Tần suất cập nhật hợp lý
        "--punct_window_chars", "2048",
        "--punct_reserve_words", "0",  # Hiển thị 2 từ cuối ở dạng "đang chờ"

        # Hiệu ứng giao diện
        "--ansi_dim_tail",  # Dùng hiệu ứng mờ (dễ nhìn hơn nhấp nháy)
        "--show_tail",
    ]
    main()