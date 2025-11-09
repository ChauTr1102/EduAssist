import os, math, argparse, yaml, re, sys, threading, queue, contextlib
from collections import deque
import traceback
from api.private_config import *
import numpy as np
import torch, torchaudio
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
BLUE = colorama.Fore.BLUE

# ===== Regex (Cải thiện) =====
# Dùng để *đếm* từ, phân tách logic, chống sai lệch (drift)
WORD_RE = re.compile(r"[0-9A-Za-zÀ-Ỵà-ỵ]+")
# Dùng để *hiển thị* và xử lý token punctuation
_TOKEN_RE = re.compile(r"\S+")
# Dùng để tìm ranh giới commit
SENT_END_RE = re.compile(r"[\.!\?…]$")


# -----------------------------

# ===== helpers =====
def advance_pointer_by_words(full_text: str, start_idx: int, n_words: int) -> int:
    """
    Di chuyển con trỏ (char index) trên 'full_text' đi đúng 'n_words' (đếm bằng WORD_RE).
    Đây là hàm then chốt để chống lại sai lệch (drift).
    """
    cnt = 0
    # Tìm từ bằng WORD_RE
    for m in WORD_RE.finditer(full_text, start_idx):
        cnt += 1
        if cnt == n_words:
            # Trả về vị trí *kết thúc* của từ thứ n
            return m.end()
    # Nếu không đủ từ, trả về cuối chuỗi
    return len(full_text)


def longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a.endswith(b[:L]): return L
    return 0


# (Loại bỏ `decap_first_token` - tin tưởng vào mô hình true-casing
#  để tránh làm hỏng tên riêng, như bạn đã đề xuất)

# ===== ASR worker (Cải thiện với VAD) =====
@torch.no_grad()
def asr_worker(args, asr_model, char_dict, hypothesis_queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ... (Các tham số model ASR giữ nguyên) ...
    subsampling = asr_model.encoder.embed.subsampling_factor
    num_layers = asr_model.encoder.num_blocks
    conv_lorder = asr_model.encoder.cnn_module_kernel // 2
    enc_steps = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size = enc_steps
    left_ctx, right_ctx = args.left_context_size, args.right_context_size
    att_cache = torch.zeros((num_layers, left_ctx, asr_model.encoder.attention_heads,
                             asr_model.encoder._output_size * 2 // asr_model.encoder.attention_heads), device=device)
    cnn_cache = torch.zeros((num_layers, asr_model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)
    # ---------------------------------------------------

    sr = args.mic_sr
    block_samples = int(args.stream_chunk_sec * sr)
    lookahead_blk = max(0, int(math.ceil(args.lookahead_sec / args.stream_chunk_sec)))

    q_audio = deque(maxlen=1 + lookahead_blk)
    full_hyp = ""
    last_sent = ""

    # --- Biến VAD ---
    silence_blocks = 0
    was_speaking = False
    # ----------------

    print("ASR worker started. Listening...")

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_samples) as stream:
            while not args.stop_event.is_set():
                audio_block, _ = stream.read(block_samples)

                # --- LOGIC VAD (Voice Activity Detection) ĐƠN GIẢN ---
                # Tính năng lượng (RMS) của khối audio
                rms = np.sqrt(np.mean(np.square(audio_block)))

                if rms < args.vad_threshold:
                    silence_blocks += 1
                    if silence_blocks > args.vad_min_silence_blocks:
                        # Đã vào trạng thái im lặng
                        if was_speaking:
                            # Vừa nói xong -> Gửi tín hiệu "commit"
                            # Chúng ta thêm một token đặc biệt để main thread biết
                            hypothesis_queue.put(full_hyp + " <COMMIT_SILENCE>")
                            was_speaking = False

                        # Bỏ qua không xử lý khối im lặng này
                        continue
                else:
                    # Đang nói
                    silence_blocks = 0
                    was_speaking = True
                # --- KẾT THÚC VAD ---

                q_audio.append(np.squeeze(audio_block, axis=1).astype(np.float32, copy=True))
                if len(q_audio) < 1 + lookahead_blk:
                    continue

                seg_np = np.concatenate(q_audio, dtype=np.float32)
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * 32768.0
                if seg.size(1) < int(0.025 * sr):
                    continue

                # --- Trích xuất đặc trưng (Nút cổ chai như bạn nói) ---
                # TODO: Chuyển sang torchaudio.transforms.MelSpectrogram trên GPU
                # để tối ưu hơn, nhưng cần kiểm tra WER cẩn thận.
                # Tạm giữ kaldi.fbank (CPU) để đảm bảo độ chính xác.
                x = kaldi.fbank(
                    seg, num_mel_bins=80, frame_length=25, frame_shift=10,
                    dither=0.0, energy_floor=0.0, sample_frequency=sr
                ).unsqueeze(0).to(device)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                # --- Chạy model ASR ---
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
    # Đặt biến môi trường TRƯỚC KHI load model (an toàn hơn)
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["ORT_NUM_THREADS"] = str(args.cpu_threads)

    m = PunctCapSegModelONNX.from_pretrained(args.punc_model)

    prov = ["CPUExecutionProvider"]
    if args.punc_device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.punc_device == "cuda":
        print("Warning: CUDAExecutionProvider not found, falling back to CPU for punctuation.", file=sys.stderr)

    # --- Cải thiện: Không can thiệp vào session sau khi đã tạo ---
    # Đoạn code `set_providers` sau khi load có thể không ổn định.
    # Thay vào đó, PunctCapSegModelONNX nên tự xử lý việc này.
    # (Nếu model này không hỗ trợ, chúng ta phải tạo session thủ công,
    # nhưng hiện tại giả định thư viện đã làm đúng)
    # -----------------------------------------------------------

    print("Punctuation model ready on", "CUDA" if prov[0].startswith("CUDA") else "CPU")
    return m


# ===== ITN (Inverse Text Normalization) functions =====
def init_itn_model(itn_model_dir):
    print(f"Loading ITN model from: {itn_model_dir}")
    far_dir = os.path.join(itn_model_dir, "far")
    classifier_far = os.path.join(far_dir, "classify/tokenize_and_classify.far")
    verbalizer_far = os.path.join(far_dir, "verbalize/verbalize.far")

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
        # print(f"\nWarning: ITN rewrite failed for: '{s}'", file=sys.stderr) # Bỏ log ồn ào
        return s
    except Exception as e:
        print(f"\nError during ITN: {e}", file=sys.stderr)
        return s


# -----------------------------

# ===== main =====
def main():
    parser = argparse.ArgumentParser(description="Realtime ASR + Punctuation + ITN (Improved)")

    # --- ASR ---
    ap = parser.add_argument_group("ASR")
    ap.add_argument("--model_checkpoint", type=str, required=True)
    ap.add_argument("--mic_sr", type=int, default=16000)
    # (Tham số mới)
    ap.add_argument("--stream_chunk_sec", type=float, default=0.36)
    ap.add_argument("--lookahead_sec", type=float, default=0.36)
    ap.add_argument("--left_context_size", type=int, default=128)
    ap.add_argument("--right_context_size", type=int, default=32)
    ap.add_argument("--cpu_threads", type=int, default=2)  # Tăng nhẹ

    # --- VAD (Mới) ---
    vp = parser.add_argument_group("VAD")
    vp.add_argument("--vad_threshold", type=float, default=0.01, help="Ngưỡng năng lượng RMS để kích hoạt VAD")
    vp.add_argument("--vad_min_silence_blocks", type=int, default=5,
                    help="Số khối im lặng liên tiếp để kích hoạt trạng thái 'im lặng'")

    # --- Punctuation (Tham số mới) ---
    pp = parser.add_argument_group("Punctuation")
    pp.add_argument("--punc_model", type=str,
                    default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase")
    pp.add_argument("--punc_device", type=str, default="cuda", choices=["cuda", "cpu"])
    pp.add_argument("--use_sbd", action="store_true")
    pp.add_argument("--punc_window_words", type=int, default=48)
    pp.add_argument("--punc_commit_margin_words", type=int, default=10)
    pp.add_argument("--punc_processing_window_words", type=int, default=72)
    pp.add_argument("--punc_context_overlap_words", type=int, default=5)

    # --- ITN ---
    ip = parser.add_argument_group("Inverse Text Normalization")
    ip.add_argument("--itn_model_dir", type=str, required=True,
                    help="Đường dẫn đến thư mục gốc của 'Vietnamese-Inverse-Text-Normalization'")

    # --- Logic (Mới) ---
    lp = parser.add_argument_group("Logic")
    lp.add_argument("--rate_limit_words", type=int, default=4, help="Chỉ chạy Punc/ITN khi có ít nhất N từ mới")
    lp.add_argument("--context_buffer_size", type=int, default=300, help="Kích thước bộ đệm ngữ cảnh (số ký tự)")

    args = parser.parse_args()
    args.stop_event = threading.Event()

    # --- Khởi tạo 3 mô hình ---
    asr_model, char_dict = init_asr_model(args)
    punc_model = init_punctuation_model(args)
    itn_classifier, itn_verbalizer = init_itn_model(args.itn_model_dir)

    hyp_q = queue.Queue(maxsize=8)

    t = threading.Thread(target=asr_worker, args=(args, asr_model, char_dict, hyp_q), daemon=True)
    t.start()

    # --- Biến trạng thái (Cải thiện) ---
    raw_text = ""  # Văn bản thô đầy đủ từ ASR
    committed_ptr = 0  # Con trỏ (char index) trên raw_text, đánh dấu phần đã commit
    last_render = ""

    # Bộ đệm cho ngữ cảnh Punctuation (chỉ lưu phần punc đã commit)
    committed_text_punctuated_context = ""

    # Bộ đệm cho hiển thị (phần ITN đã commit)
    committed_text_normalized_display = ""

    # Cache cho ITN (punc_head, itn_head)
    cached_itn_head = ("", "")

    # Cho Rate Limiting
    last_punc_call_word_count = 0
    # ------------------------------------

    print("\nMic streaming. Ctrl+C to stop.")
    try:
        while True:
            try:
                item = hyp_q.get(timeout=0.1)
                if item is None:
                    args.stop_event.set()
                    break

                force_commit = False
                if isinstance(item, str) and item.endswith(" <COMMIT_SILENCE>"):
                    latest = item.replace(" <COMMIT_SILENCE>", "").strip()
                    force_commit = True  # VAD kích hoạt "commit cưỡng bức"
                else:
                    latest = item

            except queue.Empty:
                continue

            if latest == raw_text and not force_commit:
                continue
            raw_text = latest

            # --- 0. TỐI ƯU: RATE LIMITING ---
            current_raw_word_count = len(WORD_RE.findall(raw_text))
            if (current_raw_word_count - last_punc_call_word_count < args.rate_limit_words) and not force_commit:
                continue  # Đợi thêm từ mới
            last_punc_call_word_count = current_raw_word_count
            # -------------------------------

            # 1. Chuẩn bị đầu vào Punctuation
            tail_raw = raw_text[committed_ptr:]
            if not tail_raw.strip():
                continue

            # Đếm số từ *thô* trong phần đuôi
            tail_raw_words = WORD_RE.findall(tail_raw)
            if not tail_raw_words:
                continue

            # Lấy ngữ cảnh (đã qua punc) từ bộ đệm
            context_text = committed_text_punctuated_context
            processing_window_raw = context_text + " " + tail_raw

            # 2. Chạy Punctuation
            punct_window = punc_model.infer([processing_window_raw], apply_sbd=args.use_sbd)[0]

            # 3. Tách phần Active (loại bỏ context)
            if context_text and punct_window.startswith(context_text):
                active_punc_text = punct_window[len(context_text):].strip()
            else:
                # Fallback (nếu Punc làm thay đổi cả context)
                # Chúng ta chỉ lấy N token cuối
                punc_tokens_full = _TOKEN_RE.findall(punct_window)
                raw_context_word_count = len(WORD_RE.findall(context_text))
                # Ước lượng
                active_punc_text = " ".join(punc_tokens_full[raw_context_word_count:])

            if not active_punc_text.strip():
                continue

            punct_tokens_active = _TOKEN_RE.findall(active_punc_text)

            # 4. LOGIC COMMIT (Cải thiện: Ưu tiên ranh giới câu)
            commit_k_punc_tokens = 0  # Số token (punc) sẽ commit
            found_sentence_end = False

            # Mốc kiểm tra (tính theo số *từ thô* để an toàn)
            margin_check_word_idx = len(tail_raw_words) - args.punc_commit_margin_words

            temp_word_count = 0
            for i, tok in enumerate(punct_tokens_active):
                if WORD_RE.fullmatch(tok):
                    temp_word_count += 1

                # Chỉ commit nếu tìm thấy dấu câu TRƯỚC VÙNG MARGIN
                if temp_word_count < margin_check_word_idx and SENT_END_RE.search(tok):
                    commit_k_punc_tokens = i + 1  # Commit đến (và bao gồm) token này
                    found_sentence_end = True

            # Nếu VAD kích hoạt commit, commit tất cả
            if force_commit:
                commit_k_punc_tokens = len(punct_tokens_active)
            # Nếu không tìm thấy dấu câu, dùng logic cũ (ngưỡng từ)
            elif not found_sentence_end and len(tail_raw_words) > args.punc_window_words:
                # Cắt theo punc_commit_margin_words
                # (Logic này cần map ngược lại, rất phức tạp)
                # Đơn giản hóa: Commit N - margin
                commit_k_raw_words_fallback = len(tail_raw_words) - args.punc_commit_margin_words

                # Tìm xem `commit_k_raw_words_fallback` tương ứng bao nhiêu token Punc
                temp_word_count = 0
                for i, tok in enumerate(punct_tokens_active):
                    if WORD_RE.fullmatch(tok):
                        temp_word_count += 1
                    if temp_word_count >= commit_k_raw_words_fallback:
                        commit_k_punc_tokens = i + 1
                        break

            # 5. Xử lý Commit
            if commit_k_punc_tokens > 0:
                commit_tokens_punc = punct_tokens_active[:commit_k_punc_tokens]
                commit_text_punc = " ".join(commit_tokens_punc) + " "

                # --- FIX SAI LỆCH (DRIFT) ---
                # Đếm xem `commit_text_punc` chứa bao nhiêu *từ thô*
                commit_k_raw_words = len(WORD_RE.findall(commit_text_punc))
                # -----------------------------

                # --- LUỒNG "COMMIT" ---
                commit_text_itn = inverse_normalize(commit_text_punc, itn_classifier, itn_verbalizer)
                committed_text_normalized_display += commit_text_itn.strip() + " "

                # Thêm vào bộ đệm ngữ cảnh (và giới hạn kích thước)
                committed_text_punctuated_context += commit_text_punc
                if len(committed_text_punctuated_context) > args.context_buffer_size:
                    committed_text_punctuated_context = committed_text_punctuated_context[-args.context_buffer_size:]

                # --- Di chuyển con trỏ (chống sai lệch) ---
                committed_ptr = advance_pointer_by_words(raw_text, committed_ptr, commit_k_raw_words)

                # --- Active Text (mới) ---
                active_tokens_punc = punct_tokens_active[commit_k_punc_tokens:]
                active_text_punc = " ".join(active_tokens_punc)
            else:
                # Không commit, tất cả đều là active
                active_text_punc = " ".join(punct_tokens_active)

            # 6. LUỒNG "ACTIVE" (HIỂN THỊ - có Cache)
            prefix_display = committed_text_normalized_display

            active_words_punc_toks = _TOKEN_RE.findall(active_text_punc)
            margin_tok_count = args.punc_commit_margin_words  # Ước lượng margin

            if len(active_words_punc_toks) > margin_tok_count:
                head_punc_toks = active_words_punc_toks[:-margin_tok_count]
                tail_punc_toks = active_words_punc_toks[-margin_tok_count:]

                head_punc = " ".join(head_punc_toks)
                tail_punc = " ".join(tail_punc_toks)

                # --- TỐI ƯU: CACHE ITN ---
                if cached_itn_head[0] == head_punc:
                    head_itn = cached_itn_head[1]
                else:
                    head_itn = inverse_normalize(head_punc, itn_classifier, itn_verbalizer)
                    cached_itn_head = (head_punc, head_itn)
                # -------------------------

                tail_itn = inverse_normalize(tail_punc, itn_classifier, itn_verbalizer)

                display = f"\r{prefix_display}{BLUE}{head_itn} {YELLOW}{tail_itn} "
            else:
                active_text_itn = inverse_normalize(active_text_punc, itn_classifier, itn_verbalizer)
                display = f"\r{prefix_display}{YELLOW}{active_text_itn} "

            # 7. Render
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

        # --- Xử lý phần đuôi (Cải thiện: Có ngữ cảnh) ---
        tail_raw = raw_text[committed_ptr:]
        if tail_raw.strip():
            try:
                # Thêm ngữ cảnh vào lần punc cuối cùng
                final_context = committed_text_punctuated_context
                final_raw_with_context = final_context + " " + tail_raw

                final_punc_full = punc_model.infer([final_raw_with_context])[0]

                # Tách context
                if final_context and final_punc_full.startswith(final_context):
                    final_punc_text = final_punc_full[len(final_context):].strip()
                else:
                    final_punc_text = tail_raw  # Fallback: không punc được

                final_itn_text = inverse_normalize(final_punc_text, itn_classifier, itn_verbalizer)

                current_committed = committed_text_normalized_display.strip()
                final_itn_text_stripped = final_itn_text.strip()

                if current_committed and final_itn_text_stripped:
                    committed_text_normalized_display = current_committed + " " + final_itn_text_stripped
                elif final_itn_text_stripped:
                    committed_text_normalized_display = final_itn_text_stripped

            except Exception as e:
                print(f"\nError processing final tail: {e}", file=sys.stderr)
                committed_text_normalized_display += " " + tail_raw  # Fallback

        print(f"\r{committed_text_normalized_display.strip()} ")
        print("\nDone.")


# ===== entry =====
if __name__ == "__main__":
    # Dùng các tham số mới được đề xuất
    if len(sys.argv) == 1:
        itn_repo_path = ITN_REPO

        sys.argv = [
            "realtime_decode.py",
            "--model_checkpoint", CHUNKFORMER_CHECKPOINT,
            "--itn_model_dir", itn_repo_path,
            "--punc_device", "cuda",
            "--cpu_threads", "2",

            # Tham số ASR (mới)
            "--stream_chunk_sec", "0.5",
            "--lookahead_sec", "0.5",
            "--left_context_size", "128",
            "--right_context_size", "32",

            # Tham số VAD (mới)
            "--vad_threshold", "0.01",
            "--vad_min_silence_blocks", "2",

            # Tham số Punc (mới)
            "--punc_processing_window_words", "400",
            "--punc_window_words", "240",
            "--punc_commit_margin_words", "120",
            "--punc_context_overlap_words", "3",

            # Tham số Logic (mới)
            "--rate_limit_words", "4",
            "--context_buffer_size", "300"
        ]

        if not os.path.isdir(os.path.join(itn_repo_path, "far")):
            print("=" * 50, file=sys.stderr)
            print(f"LỖI: Không tìm thấy thư mục 'far' tại: {itn_repo_path}", file=sys.stderr)
            sys.exit(1)

    main()