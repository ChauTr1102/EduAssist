import os, math, argparse, yaml, re, sys, threading, queue, contextlib
from collections import deque
import traceback

import numpy as np
import torch, torchaudio
import torchaudio.compliance.kaldi as kaldi
import sounddevice as sd
import onnxruntime as ort
from punctuators.models import PunctCapSegModelONNX
import colorama

import pynini
# --- SỬA LỖI 1: Import pynini chính xác ---
from pynini.lib.rewrite import top_rewrite
from pynini.lib import rewrite

# ------------------------------------

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
BLUE = colorama.Fore.BLUE

# ===== token & regex =====
_TOKEN_RE = re.compile(r"\S+")
END_SENT_RE = re.compile(r"[\.!\?…]\s*$", re.UNICODE)


# ===== helpers =====
def advance_pointer_by_words(full_text: str, start_idx: int, n_words: int) -> int:
    cnt = 0
    for m in _TOKEN_RE.finditer(full_text, start_idx):
        cnt += 1
        if cnt == n_words:
            return m.end()
    return len(full_text)


def longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a.endswith(b[:L]): return L
    return 0


def decap_first_token(s: str) -> str:
    if not s: return s
    i = s.find(" ")
    first = s if i == -1 else s[:i]
    rest = "" if i == -1 else s[i + 1:]
    first = first[:1].lower() + first[1:]
    return first if i == -1 else f"{first} {rest}"


# ===== ASR worker =====
@torch.no_grad()
def asr_worker(args, asr_model, char_dict, hypothesis_queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subsampling = asr_model.encoder.embed.subsampling_factor
    num_layers = asr_model.encoder.num_blocks
    conv_lorder = asr_model.encoder.cnn_module_kernel // 2
    enc_steps = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size = enc_steps

    left_ctx, right_ctx = args.left_context_size, args.right_context_size

    att_cache = torch.zeros(
        (num_layers, left_ctx, asr_model.encoder.attention_heads,
         asr_model.encoder._output_size * 2 // asr_model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros((num_layers, asr_model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    sr = args.mic_sr
    block_samples = int(args.stream_chunk_sec * sr)
    lookahead_blk = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q_audio = deque(maxlen=1 + lookahead_blk)
    full_hyp = ""
    last_sent = ""

    print("ASR worker started. Listening...")

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_samples) as stream:
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
                    seg, num_mel_bins=80, frame_length=25, frame_shift=10,
                    dither=0.0, energy_floor=0.0, sample_frequency=sr
                ).unsqueeze(0).to(device)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                use_cuda = device.type == "cuda"
                ctx = torch.amp.autocast(device_type='cuda',
                                         dtype=torch.float16) if use_cuda else contextlib.nullcontext()
                with ctx:
                    enc_out, enc_len, _, att_cache, cnn_cache, offset = asr_model.encoder.forward_parallel_chunk(
                        xs=x, xs_origin_lens=x_len, chunk_size=chunk_size,
                        left_context_size=left_ctx, right_context_size=right_ctx,
                        att_cache=att_cache, cnn_cache=cnn_cache,
                        truncated_context_size=chunk_size, offset=offset
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
        print(f"\n!!!!!!!!!! CRASH TRONG ASR WORKER !!!!!!!!!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"\nError in ASR worker: {e}", file=sys.stderr)

    finally:
        hypothesis_queue.put(None)
        print("ASR worker finished.")


# ===== model init =====
@torch.no_grad()
def init_asr_model(args):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, args.cpu_threads))

    cfg = os.path.join(args.model_checkpoint, "config.yaml")
    ckpt = os.path.join(args.model_checkpoint, "pytorch_model.bin")
    vocab = os.path.join(args.model_checkpoint, "vocab.txt")

    with open(cfg, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(config, cfg)
    model.eval()
    load_checkpoint(model, ckpt)
    if torch.cuda.is_available():
        model.encoder = model.encoder.cuda()
        model.ctc = model.ctc.cuda()

    symtab = read_symbol_table(vocab)
    char_dict = {v: k for k, v in symtab.items()}
    return model, char_dict


def init_punctuation_model(args):
    print(f"Loading punctuation model: {args.punc_model}")
    m = PunctCapSegModelONNX.from_pretrained(args.punc_model)

    prov = ["CPUExecutionProvider"]
    if args.punc_device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.punc_device == "cuda":
        print("Warning: CUDAExecutionProvider not found, falling back to CPU for punctuation.", file=sys.stderr)

    for name in dir(m):
        obj = getattr(m, name)
        if isinstance(obj, ort.InferenceSession):
            so = obj.get_session_options()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            obj.set_providers(prov, [{"arena_extend_strategy": "kNextPowerOfTwo"}])

    print("Punctuation model ready on", "CUDA" if prov[0].startswith("CUDA") else "CPU")
    return m


# ===== ITN (Inverse Text Normalization) functions =====
def init_itn_model(itn_model_dir):
    """Tải các FST .far của Pynini một lần duy nhất."""
    print(f"Loading ITN model from: {itn_model_dir}")
    far_dir = os.path.join(itn_model_dir, "far")
    classifier_far = os.path.join(far_dir, "classify/tokenize_and_classify.far")
    verbalizer_far = os.path.join(far_dir, "verbalize/verbalize.far")

    if not os.path.exists(classifier_far) or not os.path.exists(verbalizer_far):
        print(f"LỖI: Không tìm thấy file .far trong {far_dir}", file=sys.stderr)
        print(f"Hãy chắc chắn --itn_model_dir trỏ đến thư mục gốc của repo Vietnamese-Inverse-Text-Normalization",
              file=sys.stderr)
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


# --- SỬA LỖI 1: Sửa hàm ITN ---
def inverse_normalize(s: str, classifier, verbalizer) -> str:
    """Thực thi ITN trên một chuỗi bằng các FST đã được tải."""
    if not s.strip():
        return s
    try:
        token = top_rewrite(s, classifier)
        return top_rewrite(token, verbalizer)
    except rewrite.Error:
        print(f"\nWarning: ITN rewrite failed for: '{s}'", file=sys.stderr)
        return s
    except Exception as e:
        print(f"\nError during ITN: {e}", file=sys.stderr)
        return s


# -----------------------------

# ===== main =====
def main():
    parser = argparse.ArgumentParser(description="Realtime ASR + punctuation + ITN")
    ap = parser.add_argument_group("ASR")
    ap.add_argument("--model_checkpoint", type=str, required=True)
    ap.add_argument("--mic_sr", type=int, default=16000)
    ap.add_argument("--stream_chunk_sec", type=float, default=0.5)
    ap.add_argument("--lookahead_sec", type=float, default=0.5)
    ap.add_argument("--left_context_size", type=int, default=128)
    ap.add_argument("--right_context_size", type=int, default=32)
    ap.add_argument("--cpu_threads", type=int, default=1)

    pp = parser.add_argument_group("Punctuation")
    pp.add_argument("--punc_model", type=str,
                    default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
    pp.add_argument("--punc_device", type=str, default="cuda", choices=["cuda", "cpu"])
    pp.add_argument("--use_sbd", action="store_true")
    pp.add_argument("--punc_window_words", type=int, default=24)
    pp.add_argument("--punc_commit_margin_words", type=int, default=8)
    pp.add_argument("--punc_processing_window_words", type=int, default=40)
    pp.add_argument("--punc_context_overlap_words", type=int, default=3,
                    help="Số từ từ phần đã commit đưa vào đầu cửa sổ punctuation làm ngữ cảnh.")

    ip = parser.add_argument_group("Inverse Text Normalization")
    ip.add_argument("--itn_model_dir", type=str, required=True,
                    help="Đường dẫn đến thư mục gốc của 'Vietnamese-Inverse-Text-Normalization'")

    args = parser.parse_args()
    args.stop_event = threading.Event()

    # --- Khởi tạo 3 mô hình ---
    asr_model, char_dict = init_asr_model(args)
    punc_model = init_punctuation_model(args)
    itn_classifier, itn_verbalizer = init_itn_model(args.itn_model_dir)

    hyp_q = queue.Queue(maxsize=8)

    t = threading.Thread(target=asr_worker, args=(args, asr_model, char_dict, hyp_q), daemon=True)
    t.start()

    committed_text_punctuated = ""
    committed_text_normalized = ""
    raw_text = ""
    committed_ptr = 0
    last_render = ""

    print("\nMic streaming. Ctrl+C to stop.")
    try:
        while True:
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

            tail_raw = raw_text[committed_ptr:]
            if not tail_raw.strip():
                continue

            tokens_tail = _TOKEN_RE.findall(tail_raw)
            if not tokens_tail:
                continue

            # ---- context-overlap an toàn ----
            N = args.punc_processing_window_words
            K_cfg = max(0, int(args.punc_context_overlap_words))
            K = 0 if committed_ptr == 0 else K_cfg
            context_tokens = _TOKEN_RE.findall(committed_text_punctuated.strip())[
                -K:] if K > 0 and committed_text_punctuated.strip() else []
            window_tokens = context_tokens + tokens_tail[-N:]
            processing_window_raw = " ".join(window_tokens)

            punct_window = punc_model.infer([processing_window_raw], apply_sbd=args.use_sbd)[0]

            punct_tokens_full = _TOKEN_RE.findall(punct_window)
            actual_k = len(context_tokens)
            if actual_k > 0 and len(punct_tokens_full) > actual_k:
                punct_tokens = punct_tokens_full[actual_k:]
            else:
                punct_tokens = punct_tokens_full

            if len(punct_tokens) > args.punc_window_words:
                commit_k = len(punct_tokens) - args.punc_commit_margin_words

                commit_text_punc = " ".join(punct_tokens[:commit_k]) + " "

                commit_text_itn = inverse_normalize(commit_text_punc, itn_classifier, itn_verbalizer)

                committed_text_punctuated += commit_text_punc
                committed_text_normalized += commit_text_itn

                active_tokens = punct_tokens[commit_k:]
                active_text = " ".join(active_tokens)
                committed_ptr = advance_pointer_by_words(raw_text, committed_ptr, commit_k)
            else:
                active_text = " ".join(punct_tokens)

            if active_text and not END_SENT_RE.search(committed_text_punctuated.strip()):
                active_text = decap_first_token(active_text)

            # --- SỬA LỖI 2: Đảm bảo khoảng trắng khi render ---
            prefix_display = committed_text_normalized.strip()
            if prefix_display:
                prefix_display += " "  # Luôn thêm 1 khoảng trắng nếu không rỗng

            active_words = _TOKEN_RE.findall(active_text)
            if len(active_words) > args.punc_commit_margin_words:
                head = " ".join(active_words[:-args.punc_commit_margin_words])
                tail = " ".join(active_words[-args.punc_commit_margin_words:])
                display = f"\r{prefix_display}{BLUE}{head} {YELLOW}{tail} "
            else:
                display = f"\r{prefix_display}{YELLOW}{' '.join(active_words)} "
            # -----------------------------------------------

            if display != last_render:
                print(display.ljust(120), end="", flush=True)
                last_render = display

    except KeyboardInterrupt:
        print("\nStopping...")
        args.stop_event.set()
    except Exception as e:
        print(f"\n!!!!!!!!!! CRASH TRONG MAIN THREAD !!!!!!!!!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        args.stop_event.set()

    finally:
        t.join()

        tail_raw = raw_text[committed_ptr:]
        if tail_raw.strip():
            try:
                final_punc_text = punc_model.infer([tail_raw])[0]
                final_itn_text = inverse_normalize(final_punc_text, itn_classifier, itn_verbalizer)

                # SỬA LỖI 2 (cho phần final): Đảm bảo khoảng trắng
                current_committed = committed_text_normalized.strip()
                final_itn_text_stripped = final_itn_text.strip()

                if current_committed and final_itn_text_stripped:
                    committed_text_normalized = current_committed + " " + final_itn_text_stripped
                elif final_itn_text_stripped:
                    committed_text_normalized = final_itn_text_stripped
                else:
                    committed_text_normalized = current_committed

            except Exception as e:
                print(f"\nError processing final tail: {e}", file=sys.stderr)
                current_committed = committed_text_normalized.strip()
                if current_committed and tail_raw:
                    committed_text_normalized = current_committed + " " + tail_raw
                else:
                    committed_text_normalized = current_committed + tail_raw

        print(f"\r{committed_text_normalized.strip()} ")
        print("\nDone.")


# ===== entry =====
if __name__ == "__main__":
    if len(sys.argv) == 1:
        itn_repo_path = "/home/trinhchau/code/EduAssist/api/services/Vietnamese-Inverse-Text-Normalization"

        sys.argv = [
            "realtime_decode.py",
            "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
            "--itn_model_dir", itn_repo_path,
            "--punc_device", "cuda",
            "--stream_chunk_sec", "0.5",
            "--lookahead_sec", "0.5",
            "--punc_processing_window_words", "200",
            "--punc_window_words", "120",
            "--punc_commit_margin_words", "60",
            "--punc_context_overlap_words", "3",
            "--left_context_size", "128",
            "--right_context_size", "32",
            "--cpu_threads", "1",
        ]

        if not os.path.isdir(os.path.join(itn_repo_path, "far")):
            print("=" * 50, file=sys.stderr)
            print(f"LỖI: Không tìm thấy thư mục 'far' tại: {itn_repo_path}", file=sys.stderr)
            print(f"Hãy chắc chắn 'itn_repo_path' trỏ đến thư mục '{'Vietnamese-Inverse-Text-Normalization'}'",
                  file=sys.stderr)
            print(f"Và bạn đã chạy pynini_export.py để tạo file .far", file=sys.stderr)
            print("=" * 50, file=sys.stderr)
            sys.exit(1)

    main()