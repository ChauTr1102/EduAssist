import os
import math
import argparse
import yaml
import re
import sys
import threading
import queue
import contextlib
from collections import deque
import traceback

from api.private_config import *  # CHUNKFORMER_CHECKPOINT, ITN_REPO, ...

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import sounddevice as sd
import onnxruntime as ort
from punctuators.models import PunctCapSegModelONNX
import colorama

import pynini
from pynini.lib.rewrite import top_rewrite
from pynini.lib import rewrite

# ---------- Bootstrap import path (EduAssist as Sources Root) ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.services.chunkformer.model.utils.init_model import init_model
from api.services.chunkformer.model.utils.checkpoint import load_checkpoint
from api.services.chunkformer.model.utils.file_utils import read_symbol_table
from api.services.chunkformer.model.utils.ctc_utils import get_output

# ----------------------------------------------------------------------

# ANSI
colorama.init(autoreset=True)
YELLOW = colorama.Fore.YELLOW
BLUE   = colorama.Fore.BLUE

# ===== Regex & helpers =====
WORD_RE     = re.compile(r"[0-9A-Za-zÀ-Ỵà-ỵ]+")
_TOKEN_RE   = re.compile(r"\S+")
END_SENT_RE = re.compile(r"[\.!\?…]\s*$", re.UNICODE)

# numeric heuristics để bắt các pattern ITN sai kiểu "Ba nghìn 721"
NUM_TOKEN_RE = re.compile(r"^[0-9]+([.,][0-9]+)*$")

VI_NUM_WORDS = {
    "không", "một", "mốt", "hai", "ba", "bốn", "tư",
    "năm", "lăm", "sáu", "bảy", "tám", "chín",
    "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu",
    "tỷ", "tỉ", "linh", "lẻ", "phẩy"
}

PUNCT_CHARS = ".,;:!?…"


def strip_punct(tok: str) -> str:
    return tok.strip(PUNCT_CHARS)


def is_digit_token(tok: str) -> bool:
    t = strip_punct(tok)
    return bool(NUM_TOKEN_RE.match(t))


def is_vi_num_word(tok: str) -> bool:
    t = strip_punct(tok).lower()
    return t in VI_NUM_WORDS


def has_mixed_num_span(text: str) -> bool:
    """
    True nếu trong text có span liên tiếp vừa có digit vừa có từ số Việt.
    Bắt cả case có dấu phẩy: 'Ba nghìn, 721 ha' → span ['Ba', 'nghìn,', '721'].
    """
    tokens = _TOKEN_RE.findall(text)
    spans = []
    cur = []
    for tok in tokens:
        if is_digit_token(tok) or is_vi_num_word(tok):
            cur.append(tok)
        else:
            if cur:
                spans.append(cur)
                cur = []
    if cur:
        spans.append(cur)

    for span in spans:
        has_digits = any(is_digit_token(t) for t in span)
        has_words  = any(is_vi_num_word(t) for t in span)
        if has_digits and has_words:
            return True
    return False


def advance_pointer_by_words(full_text: str, start_idx: int, n_words: int) -> int:
    """
    Di chuyển con trỏ char trên 'full_text' đi đúng 'n_words' (đếm bằng WORD_RE).
    Dùng để sync pointer raw_text với số từ đã commit.
    """
    cnt = 0
    for m in WORD_RE.finditer(full_text, start_idx):
        cnt += 1
        if cnt == n_words:
            return m.end()
    return len(full_text)


def longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    """
    Ghép các chunk CTC không bị lặp.
    """
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a.endswith(b[:L]):
            return L
    return 0


# ===== ASR worker =====
@torch.no_grad()
def asr_worker(args, asr_model, char_dict, hypothesis_queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subsampling = asr_model.encoder.embed.subsampling_factor
    num_layers  = asr_model.encoder.num_blocks
    conv_lorder = asr_model.encoder.cnn_module_kernel // 2
    enc_steps   = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size  = enc_steps

    left_ctx, right_ctx = args.left_context_size, args.right_context_size

    att_cache = torch.zeros(
        (num_layers, left_ctx, asr_model.encoder.attention_heads,
         asr_model.encoder._output_size * 2 // asr_model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros(
        (num_layers, asr_model.encoder._output_size, conv_lorder),
        device=device
    )
    offset    = torch.zeros(1, dtype=torch.int, device=device)

    sr            = args.mic_sr
    block_samples = int(args.stream_chunk_sec * sr)
    lookahead_blk = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q_audio  = deque(maxlen=1 + lookahead_blk)
    full_hyp = ""
    last_sent = ""

    print("ASR worker started. Listening...")
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                            blocksize=block_samples) as stream:
            while not args.stop_event.is_set():
                audio_block, _ = stream.read(block_samples)

                q_audio.append(np.squeeze(audio_block, axis=1).astype(np.float32, copy=True))
                if len(q_audio) < 1 + lookahead_blk:
                    continue

                seg_np = np.concatenate(q_audio, dtype=np.float32)
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * 32768.0
                if seg.size(1) < int(0.025 * sr):
                    continue

                x = kaldi.fbank(
                    seg,
                    num_mel_bins=80,
                    frame_length=25,
                    frame_shift=10,
                    dither=0.0,
                    energy_floor=0.0,
                    sample_frequency=sr
                ).unsqueeze(0).to(device)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                use_cuda = device.type == "cuda"
                ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) \
                    if use_cuda else contextlib.nullcontext()

                with ctx:
                    enc_out, enc_len, _, att_cache, cnn_cache, offset = \
                        asr_model.encoder.forward_parallel_chunk(
                            xs=x,
                            xs_origin_lens=x_len,
                            chunk_size=chunk_size,
                            left_context_size=left_ctx,
                            right_context_size=right_ctx,
                            att_cache=att_cache,
                            cnn_cache=cnn_cache,
                            truncated_context_size=chunk_size,
                            offset=offset
                        )
                    T = int(enc_len.item())
                    enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :T]
                    if enc_out.size(1) > chunk_size:
                        enc_out = enc_out[:, :chunk_size]
                    offset = offset - T + enc_out.size(1)

                    hyp_step = asr_model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                chunk_text = get_output([hyp_step], char_dict)[0]
                ovl = longest_suffix_prefix_overlap(full_hyp, chunk_text)
                if ovl < len(chunk_text):
                    full_hyp += chunk_text[ovl:]

                if full_hyp != last_sent:
                    hypothesis_queue.put(full_hyp)
                    last_sent = full_hyp

    except Exception as e:
        print("\n!!!!!!!!!! CRASH TRONG ASR WORKER !!!!!!!!!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"\nError in ASR worker: {e}", file=sys.stderr)
    finally:
        hypothesis_queue.put(None)
        print("ASR worker finished.")


# ===== Model init =====
@torch.no_grad()
def init_asr_model(args):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, args.cpu_threads))

    cfg   = os.path.join(args.model_checkpoint, "config.yaml")
    ckpt  = os.path.join(args.model_checkpoint, "pytorch_model.bin")
    vocab = os.path.join(args.model_checkpoint, "vocab.txt")

    with open(cfg, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(config, cfg)
    model.eval()
    load_checkpoint(model, ckpt)

    if torch.cuda.is_available():
        model.encoder = model.encoder.cuda()
        model.ctc     = model.ctc.cuda()

    symtab = read_symbol_table(vocab)
    char_dict = {v: k for k, v in symtab.items()}
    return model, char_dict


def init_punctuation_model(args):
    print(f"Loading punctuation model: {args.punc_model}")
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["ORT_NUM_THREADS"] = str(args.cpu_threads)

    m = PunctCapSegModelONNX.from_pretrained(args.punc_model)

    prov = ["CPUExecutionProvider"]
    if args.punc_device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.punc_device == "cuda":
        print("Warning: CUDAExecutionProvider not found, falling back to CPU for punctuation.",
              file=sys.stderr)

    print("Punctuation model ready on",
          "CUDA" if prov[0].startswith("CUDA") else "CPU")
    return m


# ===== ITN =====
def init_itn_model(itn_model_dir):
    print(f"Loading ITN model from: {itn_model_dir}")
    far_dir = os.path.join(itn_model_dir, "far")
    classifier_far = os.path.join(far_dir, "classify", "tokenize_and_classify.far")
    verbalizer_far = os.path.join(far_dir, "verbalize", "verbalize.far")

    if not os.path.exists(classifier_far) or not os.path.exists(verbalizer_far):
        print(f"LỖI: Không tìm thấy file .far trong {far_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        reader_classifier = pynini.Far(classifier_far)
        reader_verbalizer = pynini.Far(verbalizer_far)
        classifier = reader_classifier.get_fst()
        verbalizer = reader_verbalizer.get_fst()
        print("ITN model ready.")
        return classifier, verbalizer
    except Exception as e:
        print(f"Lỗi khi tải ITN model: {e}", file=sys.stderr)
        sys.exit(1)


def inverse_normalize(s: str, classifier, verbalizer) -> str:
    if not s.strip():
        return s
    try:
        token = top_rewrite(s, classifier)
        return top_rewrite(token, verbalizer)
    except rewrite.Error:
        return s
    except Exception as e:
        print(f"\nError during ITN: {e}", file=sys.stderr)
        return s


def safe_itn_commit(text: str, classifier, verbalizer) -> str:
    """
    ITN an toàn cho vùng đã commit:
    - Nếu output chứa span số bẩn (mixed words+digits) -> fallback về text gốc.
    """
    norm = inverse_normalize(text, classifier, verbalizer)
    if has_mixed_num_span(norm):
        return text
    return norm


# ===== main =====
def main():
    parser = argparse.ArgumentParser(description="Realtime ASR + Punctuation + SAFE ITN (chunk-wise)")

    # ASR
    ap = parser.add_argument_group("ASR")
    ap.add_argument("--model_checkpoint", type=str, required=True)
    ap.add_argument("--mic_sr", type=int, default=16000)
    ap.add_argument("--stream_chunk_sec", type=float, default=0.5)
    ap.add_argument("--lookahead_sec", type=float, default=0.5)
    ap.add_argument("--left_context_size", type=int, default=128)
    ap.add_argument("--right_context_size", type=int, default=32)
    ap.add_argument("--cpu_threads", type=int, default=1)

    # Punctuation (giữ gần logic gốc)
    pp = parser.add_argument_group("Punctuation")
    pp.add_argument("--punc_model", type=str,
                    default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
    pp.add_argument("--punc_device", type=str, default="cuda", choices=["cuda", "cpu"])
    pp.add_argument("--use_sbd", action="store_true")
    pp.add_argument("--punc_window_words", type=int, default=240)
    pp.add_argument("--punc_commit_margin_words", type=int, default=120)
    pp.add_argument("--punc_processing_window_words", type=int, default=400)
    pp.add_argument("--punc_context_overlap_words", type=int, default=3)

    # ITN
    ip = parser.add_argument_group("ITN")
    ip.add_argument("--itn_model_dir", type=str, required=True,
                    help="Đường dẫn tới repo Vietnamese-Inverse-Text-Normalization")

    args = parser.parse_args()
    args.stop_event = threading.Event()

    # init models
    asr_model, char_dict = init_asr_model(args)
    punc_model = init_punctuation_model(args)
    itn_classifier, itn_verbalizer = init_itn_model(args.itn_model_dir)

    hyp_q = queue.Queue(maxsize=8)
    t = threading.Thread(
        target=asr_worker,
        args=(args, asr_model, char_dict, hyp_q),
        daemon=True
    )
    t.start()

    # STATE
    raw_text        = ""
    committed_ptr   = 0              # pointer trên raw_text
    committed_raw   = ""             # text đã commit sau punctuation
    committed_norm  = ""             # text đã commit (overlay ITN-safe)
    last_render     = ""

    print("\nMic streaming. Ctrl+C to stop.")
    try:
        while True:
            # lấy hypothesis mới
            try:
                item = hyp_q.get(timeout=0.1)
                if item is None:
                    args.stop_event.set()
                    break
                latest = item
            except queue.Empty:
                continue

            if latest == raw_text:
                continue
            raw_text = latest

            # tail chưa commit
            tail_raw = raw_text[committed_ptr:]
            if not tail_raw.strip():
                continue

            tokens_tail = _TOKEN_RE.findall(tail_raw)
            if not tokens_tail:
                continue

            # ====== PUNCT LOGIC: giống bản gốc ======
            N = args.punc_processing_window_words
            if committed_ptr == 0:
                K = 0
            else:
                K = args.punc_context_overlap_words

            context_tokens = _TOKEN_RE.findall(committed_raw.strip())[-K:] \
                if K > 0 and committed_raw.strip() else []
            window_tokens = context_tokens + tokens_tail[-N:]
            processing_window_raw = " ".join(window_tokens)

            punct_window = punc_model.infer([processing_window_raw],
                                            apply_sbd=args.use_sbd)[0]
            punct_tokens_full = _TOKEN_RE.findall(punct_window)

            actual_k = len(context_tokens)
            if actual_k > 0 and len(punct_tokens_full) > actual_k:
                punct_tokens = punct_tokens_full[actual_k:]
            else:
                punct_tokens = punct_tokens_full

            # commit theo window_words & commit_margin_words
            if len(punct_tokens) > args.punc_window_words:
                commit_k = len(punct_tokens) - args.punc_commit_margin_words
                commit_chunk_raw = " ".join(punct_tokens[:commit_k]) + " "
                active_raw = " ".join(punct_tokens[commit_k:])

                # 1) update committed_raw
                committed_raw += commit_chunk_raw

                # 2) ITN-safe cho chunk mới
                commit_chunk_raw_stripped = commit_chunk_raw.strip()
                commit_chunk_norm = safe_itn_commit(
                    commit_chunk_raw_stripped,
                    itn_classifier,
                    itn_verbalizer
                )
                if committed_norm and not committed_norm.endswith(" "):
                    committed_norm += " "
                committed_norm += commit_chunk_norm
                if not committed_norm.endswith(" "):
                    committed_norm += " "

                # 3) update pointer theo số từ THÔ trong commit_chunk_raw
                commit_word_count = len(WORD_RE.findall(commit_chunk_raw))
                committed_ptr = advance_pointer_by_words(
                    raw_text, committed_ptr, commit_word_count
                )
            else:
                active_raw = " ".join(punct_tokens)

            # ====== ITN mềm cho vùng active ======
            if active_raw.strip():
                active_norm = inverse_normalize(active_raw, itn_classifier, itn_verbalizer)
            else:
                active_norm = ""

            # ====== TÔ MÀU ======
            prefix_display = committed_norm
            if prefix_display and not prefix_display.endswith(" "):
                prefix_display += " "

            active_tokens_norm = _TOKEN_RE.findall(active_norm)
            # dùng commit_margin_words làm độ dài tail màu vàng
            tail_color = max(1, int(args.punc_commit_margin_words))

            if len(active_tokens_norm) > tail_color:
                head_norm = " ".join(active_tokens_norm[:-tail_color])
                tail_norm = " ".join(active_tokens_norm[-tail_color:])
                display = f"\r{prefix_display}{BLUE}{head_norm} {YELLOW}{tail_norm} "
            else:
                display = f"\r{prefix_display}{YELLOW}{' '.join(active_tokens_norm)} "

            if display != last_render:
                print(display.ljust(120), end="", flush=True)
                last_render = display

    except KeyboardInterrupt:
        print("\nStopping...")
        args.stop_event.set()
    except Exception as e:
        print("\n!!!!!!!!!! CRASH TRONG MAIN THREAD !!!!!!!!!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        args.stop_event.set()
    finally:
        t.join()

        # ====== FINALIZE: punc+ITN cho phần đuôi ======
        tail_raw = raw_text[committed_ptr:]
        final_punc_tail = ""

        if tail_raw.strip():
            try:
                tokens_tail = _TOKEN_RE.findall(tail_raw)
                if tokens_tail:
                    N = args.punc_processing_window_words
                    if committed_ptr == 0:
                        K = 0
                    else:
                        K = args.punc_context_overlap_words

                    context_tokens = _TOKEN_RE.findall(committed_raw.strip())[-K:] \
                        if K > 0 and committed_raw.strip() else []
                    window_tokens = context_tokens + tokens_tail[-N:]
                    processing_window_raw = " ".join(window_tokens)

                    final_punc_window = punc_model.infer(
                        [processing_window_raw],
                        apply_sbd=args.use_sbd
                    )[0]

                    punct_tokens_full = _TOKEN_RE.findall(final_punc_window)
                    actual_k = len(context_tokens)
                    if actual_k > 0 and len(punct_tokens_full) > actual_k:
                        punct_tokens_tail = punct_tokens_full[actual_k:]
                    else:
                        punct_tokens_tail = punct_tokens_full

                    final_punc_tail = " ".join(punct_tokens_tail)
            except Exception as e:
                print(f"\nError final punctuation tail: {e}", file=sys.stderr)

        full_punc_text = (committed_raw + " " + final_punc_tail).strip()
        try:
            final_norm = safe_itn_commit(full_punc_text, itn_classifier, itn_verbalizer)
        except Exception as e:
            print(f"\nError final ITN: {e}", file=sys.stderr)
            final_norm = full_punc_text

        print(f"\r{final_norm.strip()} ")
        print("\nDone.")


# ===== entry =====
if __name__ == "__main__":
    if len(sys.argv) == 1:
        itn_repo_path = ITN_REPO

        sys.argv = [
            "realtime_decode.py",
            "--model_checkpoint", CHUNKFORMER_CHECKPOINT,
            "--itn_model_dir", itn_repo_path,
            "--punc_device", "cuda",
            "--cpu_threads", "1",

            "--stream_chunk_sec", "0.5",
            "--lookahead_sec", "0.2",
            "--left_context_size", "128",
            "--right_context_size", "32",

            "--punc_processing_window_words", "400",
            "--punc_window_words", "240",
            "--punc_commit_margin_words", "120",
            "--punc_context_overlap_words", "3",
        ]

        if not os.path.isdir(os.path.join(itn_repo_path, "far")):
            print("=" * 50, file=sys.stderr)
            print(f"LỖI: Không tìm thấy thư mục 'far' tại: {itn_repo_path}", file=sys.stderr)
            sys.exit(1)

    main()
