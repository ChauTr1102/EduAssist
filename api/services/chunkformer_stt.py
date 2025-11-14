from api.services import *
import subprocess, sys
import time
from api.private_config import *
from pathlib import Path
import sys
import sounddevice as sd


# đường dẫn tới thư mục 'chunkformer' (nằm cùng cấp với file này)
HERE = Path(__file__).resolve().parent
CHUNKFORMER_DIR = HERE / "chunkformer"

# thêm 'services/chunkformer' vào sys.path để import 'model.*' hoạt động
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(CHUNKFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(CHUNKFORMER_DIR))

import  contextlib
import os
import math
import argparse
import yaml
import torch
import torchaudio
import pandas as pd
import jiwer
import re
from collections import deque

from tqdm import tqdm
from colorama import Fore, Style
import torchaudio.compliance.kaldi as kaldi
import sounddevice as sd
import numpy as np
from chunkformer.model.utils.init_model import init_model
from chunkformer.model.utils.checkpoint import load_checkpoint
from chunkformer.model.utils.file_utils import read_symbol_table
from chunkformer.model.utils.ctc_utils import get_output_with_timestamps, get_output, milliseconds_to_hhmmssms
from pynini.lib.rewrite import top_rewrite
from pynini.lib import rewrite
# ==================== Utils for stable streaming without CIF ====================

def _longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    """Độ dài trùng khớp dài nhất giữa suffix(a) và prefix(b) để khử trùng lặp chồng lấn."""
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a[-L:] == b[:L]:
            return L
    return 0

def longest_suffix_prefix_overlap(a: str, b: str, max_k: int = 32) -> int:
    """Độ dài lớn nhất L để a.endswith(b[:L]) với L<=max_k."""
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a.endswith(b[:L]):
            return L
    return 0

_WORD_BOUNDARY_RE = re.compile(r"[ \t\n\r\f\v.,!?;:…，。！？；：]")

def _split_commit_tail(text: str, reserve_last_k_words: int = 1):
    """
    Chỉ commit đến ranh giới từ chắc chắn.
    Giữ lại k từ cuối làm 'tail' để tránh cắt nửa từ ở biên.
    """
    # Tách theo khoảng trắng và dấu câu. Giữ dấu trong output.
    parts = re.split(r"(\s+|[.,!?;:…])", text)
    words = []
    buf = ""
    for p in parts:
        buf += p
        # Ranh giới từ khi gặp khoảng trắng hoặc dấu câu
        if _WORD_BOUNDARY_RE.fullmatch(p or ""):
            words.append(buf)
            buf = ""
    if buf:
        words.append(buf)

    if not words:
        return "", text

    # Ghép thành từng đơn vị "token từ" theo ranh giới
    # Sau đó giữ lại k phần tử cuối làm tail
    if len(words) <= reserve_last_k_words:
        return "", "".join(words)
    commit = "".join(words[:-reserve_last_k_words])
    tail = "".join(words[-reserve_last_k_words:])
    return commit, tail

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


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

def _compute_rel_right_context_frames(chunk_size_enc, right_context_size, conv_lorder, num_layers, subsampling):
    r_enc = max(right_context_size, conv_lorder)
    rrel_enc = r_enc + max(chunk_size_enc, r_enc) * (num_layers - 1)
    return rrel_enc * subsampling  # đổi sang số frame 10ms trước subsampling


# ==================== Model init ====================
class ChunkFormer:
    def __init__(self, model_checkpoint):
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        config_path = os.path.join(model_checkpoint, "config.yaml")
        checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
        symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

        with open(config_path, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        self.model = init_model(config, config_path)
        self.model.eval()
        load_checkpoint(self.model, checkpoint_path)

        self.model.encoder = self.model.encoder.cuda()
        self.model.ctc = self.model.ctc.cuda()

        symbol_table = read_symbol_table(symbol_table_path)
        self.char_dict = {v: k for k, v in symbol_table.items()}



    # ==================== Streaming from file with lookahead ====================
    def stream_audio(self, stream_chunk_sec, left_context_size, right_context_size, long_form_audio, lookahead_sec,
                     stable_reserve_words, print_final):
        """
        Giả lập streaming: cắt 0.5s, cộng lookahead cố định, chồng lấn văn bản.
        """
        device = torch.device("cuda")
        subsampling = self.model.encoder.embed.subsampling_factor  # thường = 8
        num_layers = self.model.encoder.num_blocks
        conv_lorder = self.model.encoder.cnn_module_kernel // 2

        # chunk_size theo encoder-steps (sau subsampling)
        enc_steps = max(1, int(round((stream_chunk_sec / 0.01) / subsampling)))  # ví dụ 0.5s -> ~6 steps
        chunk_size = enc_steps

        left_context_size = left_context_size
        right_context_size = right_context_size

        # cache
        att_cache = torch.zeros(
            (num_layers, left_context_size, self.model.encoder.attention_heads,
             self.model.encoder._output_size * 2 // self.model.encoder.attention_heads),
            device=device
        )
        cnn_cache = torch.zeros((num_layers, self.model.encoder._output_size, conv_lorder), device=device)
        offset = torch.zeros(1, dtype=torch.int, device=device)

        # audio
        wav_path = long_form_audio
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        assert sample_rate == 16000, "Yêu cầu audio 16 kHz"
        waveform = waveform * (1 << 15)

        hop_samples = int(stream_chunk_sec * sample_rate)  # ví dụ 0.5s
        lookahead_samples = int(lookahead_sec * sample_rate)

        produced_steps = 0
        carry_text = ""        # đuôi chưa commit
        committed_text = ""    # đã phát
        all_tokens = []

        cur = 0
        while cur < waveform.size(1):
            seg_end = min(cur + hop_samples, waveform.size(1))
            seg_end_with_look = min(seg_end + lookahead_samples, waveform.size(1))
            seg = waveform[:, cur:seg_end_with_look]

            # đảm bảo tối thiểu 25ms cho fbank
            if seg.size(1) < int(0.025 * sample_rate):
                break

            x = kaldi.fbank(
                seg,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=16000
            ).unsqueeze(0).to(device)  # (1, T_fbank, 80)

            x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

            truncated_context_size = chunk_size  # chỉ xuất đúng số bước hữu ích
            with torch.cuda.amp.autocast(dtype=torch.float16):
                encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = self.model.encoder.forward_parallel_chunk(
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
                encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
                if encoder_outs.shape[1] > truncated_context_size:
                    encoder_outs = encoder_outs[:, :truncated_context_size]

                offset = offset - encoder_lens + encoder_outs.shape[1]
                hyp_step = self.model.encoder.ctc_forward(encoder_outs).squeeze(0)

            all_tokens.append(hyp_step.cpu())

            seg_start_ms = int(produced_steps * subsampling * 10)
            seg_end_ms = int((produced_steps + hyp_step.numel()) * subsampling * 10)
            produced_steps += hyp_step.numel()

            # Văn bản chunk hiện tại
            chunk_text = get_output([hyp_step.cpu()], self.char_dict)[0]

            # Ghép chồng lấn: xóa phần trùng giữa carry và chunk_text
            ov = _longest_suffix_prefix_overlap(carry_text, chunk_text, max_k=32)
            merged = carry_text + chunk_text[ov:]

            # Chỉ phát đến ranh giới từ, giữ lại đuôi
            commit, new_tail = _split_commit_tail(merged, reserve_last_k_words=max(1, stable_reserve_words))
            if commit:
                committed_text += commit
                print(
                    f"{Fore.CYAN}{milliseconds_to_hhmmssms(seg_start_ms)}{Style.RESET_ALL}"
                    f" - "
                    f"{Fore.CYAN}{milliseconds_to_hhmmssms(seg_end_ms)}{Style.RESET_ALL}"
                    f": {commit.strip()}"
                )
            carry_text = new_tail

            cur = seg_end
            torch.cuda.empty_cache()

        # flush phần còn lại
        if print_final:
            # Gộp theo token cho kết quả cuối có timestamp
            hyps = torch.cat(all_tokens) if all_tokens else torch.tensor([], dtype=torch.long)
            final_decode = get_output_with_timestamps([hyps], self.char_dict)[0]
            print("\n=== Final (merged) ===")
            for item in final_decode:
                start = f"{Fore.RED}{item['start']}{Style.RESET_ALL}"
                end = f"{Fore.RED}{item['end']}{Style.RESET_ALL}"
                print(f"{start} - {end}: {item['decode']}")
        else:
            if carry_text.strip():
                print(carry_text.strip())


    # ==================== Microphone streaming with fixed lookahead ====================

    @torch.no_grad()
    def stream_mic_silent_rms(self, stream_chunk_sec: float,
                   left_context_size, right_context_size,
                   mic_sr, lookahead_sec: float,
                   silence_rms, silence_runs,
                   stable_reserve_words: int,
                   max_duration_sec: float | None = None,
                   on_update=None):
        """
            Streaming với callback: mỗi lần có commit/flush sẽ gọi on_update(event, text, full).
            Kết thúc trả về full transcript.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subsampling = self.model.encoder.embed.subsampling_factor
        num_layers = self.model.encoder.num_blocks
        conv_lorder = self.model.encoder.cnn_module_kernel // 2

        # chunk_size theo encoder-steps từ stream_chunk_sec
        enc_steps = max(1, int(round((stream_chunk_sec / 0.01) / subsampling)))
        chunk_size = enc_steps

        # cache để giữ LEFT CONTEXT giữa các bước
        att_cache = torch.zeros(
            (num_layers, left_context_size, self.model.encoder.attention_heads,
             self.model.encoder._output_size * 2 // self.model.encoder.attention_heads),
             device=device
        )
        cnn_cache = torch.zeros((num_layers, self.model.encoder._output_size, conv_lorder), device=device)
        offset = torch.zeros(1, dtype=torch.int, device=device)

        # cấu hình thu âm
        sr = mic_sr
        assert sr == 16000, "Mic nên 16 kHz để khớp feature"
        block_samples = int(stream_chunk_sec * sr)
        lookahead_blocks = max(0, int(math.ceil(lookahead_sec / stream_chunk_sec)))

        # hàng đợi khối audio để tạo lookahead cố định
        q = deque()
        produced_steps = 0
        carry_text = ""
        silence_run = 0
        SIL_THRESH = silence_rms
        SIL_RUN_TO_FLUSH = silence_runs

        transcript_parts: list[str] = []
        start_time = time.time()

        def _emit(event_type: str, payload):
            if on_update is not None:
                on_update(event_type, payload, " ".join(transcript_parts).strip())
        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_samples) as stream:
                prev_end_time = 0
                while True:
                    if max_duration_sec is not None and (time.time() - start_time) >= max_duration_sec:
                        if carry_text.strip():
                            transcript_parts.append(carry_text.strip())
                            _emit("final_flush", carry_text.strip())
                        break
                    audio_block, _ = stream.read(block_samples)  # (N, 1)
                    a = np.squeeze(audio_block, axis=1).astype(np.float32)  # [-1,1]
                    q.append(a)

                    # Kiểm tra im lặng
                    silence_run = silence_run + 1 if _rms(a) < SIL_THRESH else 0

                    # Đợi đủ lookahead
                    if len(q) < 1 + lookahead_blocks:
                        continue

                    # Ghép block cũ nhất + lookahead
                    seg_np = np.concatenate([q[0]] + list(list(q)[1:1 + lookahead_blocks]))  # không làm thay đổi q
                    seg = torch.from_numpy(seg_np).unsqueeze(0).to(device)  # (1, T)
                    seg = seg * (1 << 15)

                    if seg.size(1) < int(0.025 * sr):
                        q.popleft()
                        continue

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

                    truncated_context_size = chunk_size
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        enc_out, enc_len, _, att_cache, cnn_cache, offset = self.model.encoder.forward_parallel_chunk(
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

                        hyp_step = self.model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                    seg_start_ms = int(produced_steps * subsampling * 10)
                    seg_end_ms = int((produced_steps + hyp_step.numel()) * subsampling * 10)
                    produced_steps += hyp_step.numel()

                    chunk_text = get_output([hyp_step], self.char_dict)[0]

                    # Ghép chồng lấn ổn định
                    # ov = _longest_suffix_prefix_overlap(carry_text, chunk_text, max_k=32)
                    # merged = carry_text + chunk_text[ov:]
                    merged = _smart_merge(carry_text, chunk_text, lcs_window=64, lcs_min=16)
                    commit, new_tail = _split_commit_tail(merged, reserve_last_k_words=max(1, stable_reserve_words))

                    if commit:
                        cleaned = commit.strip()
                        if cleaned and cleaned != " ":
                            transcript_parts.append(cleaned)
                            payload = {"start": seg_start_ms, "end": seg_end_ms, "text": cleaned}
                            _emit("commit", payload)
                    # Nếu im lặng nhiều block thì flush tail
                    if silence_run >= SIL_RUN_TO_FLUSH and new_tail.strip():
                        cleaned_tail = new_tail.strip()
                        transcript_parts.append(cleaned_tail)
                        payload = {"start": seg_start_ms, "end": seg_end_ms, "text": cleaned_tail}
                        _emit("flush", payload)
                        new_tail = ""

                    carry_text = new_tail
                    # Trượt cửa sổ: bỏ block đã xử lý
                    q.popleft()
                    torch.cuda.empty_cache()
        except KeyboardInterrupt:
            if carry_text.strip():
                transcript_parts.append(carry_text.strip())
                _emit("final_flush", carry_text.strip())

        return " ".join(transcript_parts).strip()

    @torch.no_grad()
    def chunkformer_asr_realtime(self, mic_sr: int = 16000, stream_chunk_sec: float = 0.5
                                , lookahead_sec: float = 0.5, left_context_size: int = 128, right_context_size: int = 32
                                , max_overlap_match: int = 32, on_update=None, return_final: bool = True, stop_event=None):
        """
        Realtime ASR từ microphone cho đến khi Ctrl+C.
        Giữ logic: fbank -> forward_parallel_chunk -> CTC -> ghép overlap (suffix-prefix).
        Output: emit theo chuẩn stream_mic qua on_update(event, payload, full).

        Event:
          - "commit": có đoạn text mới ổn định
          - "final_flush": flush phần còn lại khi kết thúc
        Payload: {"start": <ms>, "end": <ms>, "text": <str>}
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _emit(event_type: str, payload, full_text: str):
            if on_update is not None:
                on_update(event_type, payload, full_text)

        # --- encoder params ---
        subsampling = self.model.encoder.embed.subsampling_factor
        num_layers = self.model.encoder.num_blocks
        conv_lorder = self.model.encoder.cnn_module_kernel // 2

        # chunk_size theo encoder steps (sau subsampling)
        enc_steps = max(1, int(round((stream_chunk_sec / 0.01) / subsampling)))
        chunk_size = enc_steps

        # caches cho streaming
        att_cache = torch.zeros(
            (num_layers, left_context_size, self.model.encoder.attention_heads,
             self.model.encoder._output_size * 2 // self.model.encoder.attention_heads),
            device=device
        )
        cnn_cache = torch.zeros(
            (num_layers, self.model.encoder._output_size, conv_lorder),
            device=device
        )
        offset = torch.zeros(1, dtype=torch.int, device=device)

        # audio buffer/lookahead
        block_samples = int(stream_chunk_sec * mic_sr)
        lookahead_blk = max(0, int(math.ceil(lookahead_sec / stream_chunk_sec)))
        q_audio = deque(maxlen=1 + lookahead_blk)

        # văn bản tích lũy
        full_hyp = ""  # toàn bộ giả thuyết sau khi ghép
        transcript_parts = []  # để build full khi emit
        produced_steps = 0  # đếm số bước encoder đã sinh (ước lượng thời gian)

        print("ASR (Chunkformer) is listening... Press Ctrl+C to stop.")
        try:
            with sd.InputStream(samplerate=mic_sr, channels=1, dtype="float32",
                                blocksize=block_samples) as stream:
                while True:

                    if stop_event is not None and stop_event.is_set():
                        break

                    audio_block, _ = stream.read(block_samples)
                    # mono float32
                    q_audio.append(np.squeeze(audio_block, axis=1).astype(np.float32, copy=True))

                    # cần đủ lookahead
                    if len(q_audio) < 1 + lookahead_blk:
                        continue

                    # ghép khối hiện tại + lookahead cố định
                    seg_np = np.concatenate(q_audio, dtype=np.float32)
                    if seg_np.size < int(0.025 * mic_sr):
                        continue

                    # fbank
                    seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * 32768.0
                    x = kaldi.fbank(
                        seg, num_mel_bins=80, frame_length=25, frame_shift=10,
                        dither=0.0, energy_floor=0.0, sample_frequency=mic_sr
                    ).unsqueeze(0).to(device)
                    x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                    # forward streaming
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        enc_out, enc_len, _, att_cache, cnn_cache, offset = self.model.encoder.forward_parallel_chunk(
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

                        T = int(enc_len.item())
                        enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :T]
                        if enc_out.size(1) > chunk_size:
                            enc_out = enc_out[:, :chunk_size]
                        offset = offset - T + enc_out.size(1)

                        # logits -> greedy CTC
                        hyp_step = self.model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                    # ước lượng mốc thời gian cho đoạn step này (ms)
                    seg_start_ms = int(produced_steps * subsampling * 10)
                    seg_end_ms = int((produced_steps + hyp_step.numel()) * subsampling * 10)
                    produced_steps += hyp_step.numel()

                    # văn bản chunk hiện tại
                    chunk_text = get_output([hyp_step], self.char_dict)[0]

                    # ghép an toàn bằng suffix-prefix overlap
                    ovl = longest_suffix_prefix_overlap(full_hyp, chunk_text, max_k=max_overlap_match)
                    new_piece = chunk_text[ovl:] if ovl < len(chunk_text) else ""

                    if new_piece.strip():
                        # cập nhật full và emit "commit" giống stream_mic
                        full_hyp += new_piece
                        cleaned = new_piece.strip()
                        transcript_parts.append(cleaned)
                        payload = {"start": seg_start_ms, "end": seg_end_ms, "text": cleaned}
                        _emit("commit", payload, " ".join(transcript_parts).strip())

                    # trượt cửa sổ: bỏ block cũ nhất
                    q_audio.popleft()
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            pass
        finally:
            # flush phần còn lại như stream_mic: "final_flush"
            remaining = ""
            if full_hyp and (not transcript_parts or " ".join(transcript_parts).strip() != full_hyp.strip()):
                # nếu còn phần chưa đóng gói (hiếm khi xảy ra với chiến lược commit theo new_piece),
                # ta coi phần sai khác là remaining để emit nốt
                already = " ".join(transcript_parts).strip()
                if len(full_hyp) > len(already):
                    remaining = full_hyp[len(already):].strip()

            if remaining:
                transcript_parts.append(remaining)
                payload = {"start": 0, "end": 0, "text": remaining}  # không có mốc chính xác → 0
                _emit("final_flush", payload, " ".join(transcript_parts).strip())

            if return_final:
                return " ".join(transcript_parts).strip()
            return ""

    @torch.no_grad()
    def chunkformer_asr_realtime_punc_norm(self, mic_sr: int = 16000, stream_chunk_sec: float = 0.5, lookahead_sec: float = 0.5,
                                left_context_size: int = 128, right_context_size: int = 32, max_overlap_match: int = 32,
                                # VAD
                                vad_threshold: float = 0.01, vad_min_silence_blocks: int = 2,

                                # Punctuation
                                punc_model=None, use_sbd: bool = False, punc_window_words: int = 240,
                                punc_commit_margin_words: int = 120,

                                # ITN
                                itn_classifier=None, itn_verbalizer=None,

                                # Logic
                                rate_limit_words: int = 4, context_buffer_size: int = 300,

                                # Control
                                on_update=None, stop_event=None, return_final: bool = True,
    ):
        """
        Realtime ASR + VAD + Punctuation + ITN (tích hợp chiến lược mới).

        - Đọc từ microphone trên server (sounddevice).
        - ASR streaming với ChunkFormer (forward_parallel_chunk + CTC).
        - VAD: phát hiện khoảng im lặng để ép commit.
        - Punctuation + truecase + ITN: chạy theo cửa sổ ngữ cảnh, commit thông minh.
        - on_update(event, payload) để UI/backend subscribe:

            event="partial":
                payload = {
                    "display": <chuỗi đã format: committed + active>,
                    "committed": <phần đã commit (ITN)>,
                    "active": <phần active (ITN)>,
                }

            event="commit":
                payload = {
                    "new_commit": <đoạn ITN mới commit>,
                    "committed": <toàn bộ committed (ITN)>,
                }

            event="final_flush":
                payload = {
                    "text": <toàn bộ ITN final>,
                }

        Lưu ý:
        - punc_model, itn_classifier, itn_verbalizer phải được truyền từ ngoài (init 1 lần).
        - Nếu không truyền punc/itn, hàm sẽ trả raw text như cũ (chỉ dùng ASR + VAD).
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _emit(event: str, payload: dict):
            if on_update is not None:
                try:
                    on_update(event, payload)
                except Exception as e:
                    print(f"[chunkformer_asr_realtime_2][on_update error] {e}", flush=True)

        # ====== ASR encoder setup ======
        subsampling = self.model.encoder.embed.subsampling_factor
        num_layers = self.model.encoder.num_blocks
        conv_lorder = self.model.encoder.cnn_module_kernel // 2

        enc_steps = max(1, int(round((stream_chunk_sec / 0.01) / subsampling)))
        chunk_size = enc_steps

        att_cache = torch.zeros((num_layers, left_context_size, self.model.encoder.attention_heads,
                                 self.model.encoder._output_size * 2 // self.model.encoder.attention_heads), device=device)
        cnn_cache = torch.zeros((num_layers, self.model.encoder._output_size, conv_lorder), device=device)
        offset = torch.zeros(1, dtype=torch.int, device=device)

        # ====== streaming audio params ======
        sr = mic_sr
        block_samples = int(stream_chunk_sec * sr)
        lookahead_blk = max(0, int(math.ceil(lookahead_sec / stream_chunk_sec)))
        q_audio = deque(maxlen=1 + lookahead_blk)

        # ====== VAD state ======
        silence_blocks = 0
        was_speaking = False

        # ====== text / formatting state ======
        # raw_text: toàn bộ giả thuyết CTC (lexical, no punc)
        raw_text = ""
        last_raw_sent = ""

        # pointer trên raw_text đánh dấu phần đã commit (tính theo ký tự, chống drift)
        committed_ptr = 0

        # context (đã punctuated) dùng làm tiền tố cho lần punc tiếp theo
        committed_text_punctuated_context = ""

        # phần đã commit (đã ITN, để display)
        committed_text_normalized_display = ""

        # cache ITN cho phần "head" active
        cached_itn_head = ("", "")

        # rate limiting cho Punc/ITN
        last_punc_call_word_count = 0

        # helper regex
        WORD_RE = re.compile(r"[0-9A-Za-zÀ-Ỵà-ỵ]+")
        TOKEN_RE = re.compile(r"\S+")
        SENT_END_RE = re.compile(r"[\.!\?…]$")

        def advance_pointer_by_words(full_text: str, start_idx: int, n_words: int) -> int:
            cnt = 0
            for m in WORD_RE.finditer(full_text, start_idx):
                cnt += 1
                if cnt == n_words:
                    return m.end()
            return len(full_text)

        def inverse_normalize_local(s: str) -> str:
            if itn_classifier is None or itn_verbalizer is None:
                return s
            if not s.strip():
                return s
            try:
                token = top_rewrite(s, itn_classifier)
                return top_rewrite(token, itn_verbalizer)
            except rewrite.Error:
                return s
            except Exception as e:
                print(f"[ITN] error: {e}", flush=True)
                return s

        def run_punc_itn_and_emit(force_commit: bool = False):
            """
            Chạy punctuation + ITN + commit logic trên raw_text hiện tại,
            update committed_text_* và emit partial/commit.
            """
            nonlocal raw_text, committed_ptr
            nonlocal committed_text_punctuated_context, committed_text_normalized_display
            nonlocal cached_itn_head, last_punc_call_word_count

            if punc_model is None:
                # Không có punc/itn: emit raw luôn
                _emit("partial", {
                    "display": raw_text,
                    "committed": raw_text,
                    "active": "",
                })
                return

            # 0. Rate limiting
            current_raw_word_count = len(WORD_RE.findall(raw_text))
            if not force_commit:
                if current_raw_word_count - last_punc_call_word_count < rate_limit_words:
                    return
            last_punc_call_word_count = current_raw_word_count

            # 1. Lấy phần tail chưa commit
            tail_raw = raw_text[committed_ptr:]
            if not tail_raw.strip():
                return

            tail_raw_words = WORD_RE.findall(tail_raw)
            if not tail_raw_words:
                return

            # 2. Xây context cho punc
            # Giới hạn độ dài context để tránh quá dài
            context_text = committed_text_punctuated_context[-context_buffer_size:]
            processing_window_raw = (context_text + " " + tail_raw).strip()

            # 3. Gọi punctuation model
            punct_window = punc_model.infer([processing_window_raw], apply_sbd=use_sbd)[0]

            # 4. Tách phần active (loại context)
            if context_text and punct_window.startswith(context_text):
                active_punc_text = punct_window[len(context_text):].strip()
            else:
                # fallback: cắt dựa trên số từ context
                punc_tokens_full = TOKEN_RE.findall(punct_window)
                raw_context_word_count = len(WORD_RE.findall(context_text))
                active_punc_text = " ".join(punc_tokens_full[raw_context_word_count:])

            if not active_punc_text.strip():
                return

            punct_tokens_active = TOKEN_RE.findall(active_punc_text)

            # 5. Logic chọn lượng commit
            commit_k_punc_tokens = 0
            found_sentence_end = False

            margin_check_word_idx = len(tail_raw_words) - punc_commit_margin_words
            temp_word_count = 0
            for i, tok in enumerate(punct_tokens_active):
                if WORD_RE.fullmatch(tok):
                    temp_word_count += 1
                if temp_word_count < margin_check_word_idx and SENT_END_RE.search(tok):
                    commit_k_punc_tokens = i + 1
                    found_sentence_end = True

            if force_commit:
                commit_k_punc_tokens = len(punct_tokens_active)
            elif not found_sentence_end and len(tail_raw_words) > punc_window_words:
                # fallback: commit đến trước margin
                commit_k_raw_words_fallback = len(tail_raw_words) - punc_commit_margin_words
                temp_word_count = 0
                for i, tok in enumerate(punct_tokens_active):
                    if WORD_RE.fullmatch(tok):
                        temp_word_count += 1
                    if temp_word_count >= commit_k_raw_words_fallback:
                        commit_k_punc_tokens = i + 1
                        break

            # 6. Xử lý commit (nếu có)
            new_commit_text_itn = ""
            if commit_k_punc_tokens > 0:
                commit_tokens_punc = punct_tokens_active[:commit_k_punc_tokens]
                commit_text_punc = " ".join(commit_tokens_punc) + " "

                # đếm số từ thô tương ứng
                commit_k_raw_words = len(WORD_RE.findall(commit_text_punc))

                # ITN phần commit
                commit_text_itn = inverse_normalize_local(commit_text_punc).strip()
                if commit_text_itn:
                    committed_text_normalized_display = (
                            committed_text_normalized_display + " " + commit_text_itn
                    ).strip()
                    new_commit_text_itn = commit_text_itn

                # cập nhật context punc
                committed_text_punctuated_context = (
                        committed_text_punctuated_context + commit_text_punc
                )[-context_buffer_size:]

                # dịch committed_ptr theo số từ (chống drift)
                committed_ptr = advance_pointer_by_words(
                    raw_text, committed_ptr, commit_k_raw_words
                )

                # phần active còn lại
                active_tokens_punc = punct_tokens_active[commit_k_punc_tokens:]
                active_text_punc = " ".join(active_tokens_punc)
            else:
                active_text_punc = " ".join(punct_tokens_active)

            # 7. Handle active part (ITN với cache)
            prefix_display = committed_text_normalized_display

            active_tokens = TOKEN_RE.findall(active_text_punc)
            margin_tok_count = punc_commit_margin_words

            if active_tokens:
                if len(active_tokens) > margin_tok_count:
                    head_punc_toks = active_tokens[:-margin_tok_count]
                    tail_punc_toks = active_tokens[-margin_tok_count:]

                    head_punc = " ".join(head_punc_toks)
                    tail_punc = " ".join(tail_punc_toks)

                    if cached_itn_head[0] == head_punc:
                        head_itn = cached_itn_head[1]
                    else:
                        head_itn = inverse_normalize_local(head_punc)
                        cached_itn_head = (head_punc, head_itn)

                    tail_itn = inverse_normalize_local(tail_punc)
                    active_itn = (head_itn + " " + tail_itn).strip()
                else:
                    active_itn = inverse_normalize_local(active_text_punc)
            else:
                active_itn = ""

            display_full = (prefix_display + " " + active_itn).strip()

            # 8. Emit sự kiện
            if new_commit_text_itn:
                _emit("commit", {
                    "new_commit": new_commit_text_itn,
                    "committed": committed_text_normalized_display,
                    "display": display_full,
                })

            _emit("partial", {
                "display": display_full,
                "committed": committed_text_normalized_display,
                "active": active_itn,
            })

        # ====== MAIN LOOP: đọc mic + ASR + VAD ======
        final_text = ""

        print("ASR + Punc + ITN realtime. Listening on server mic...")

        try:
            with sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    blocksize=block_samples,
            ) as stream:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        break

                    audio_block, _ = stream.read(block_samples)

                    # VAD RMS
                    rms = float(np.sqrt(np.mean(np.square(audio_block)))) if audio_block.size else 0.0

                    force_commit = False
                    if rms < vad_threshold:
                        silence_blocks += 1
                        if silence_blocks > vad_min_silence_blocks:
                            if was_speaking:
                                force_commit = True  # chuyển từ nói -> im lặng
                                was_speaking = False
                        # nếu im lặng dài, không cần chạy ASR cho block này
                    else:
                        silence_blocks = 0
                        was_speaking = True

                    # nếu block toàn im lặng sau commit, bỏ qua luôn
                    if rms < vad_threshold and not was_speaking and not force_commit:
                        continue

                    # ===== ASR chunkformer giống bản cũ =====
                    q_audio.append(
                        np.squeeze(audio_block, axis=1).astype(np.float32, copy=True)
                    )
                    if len(q_audio) < 1 + lookahead_blk:
                        # chưa đủ lookahead
                        if force_commit:
                            run_punc_itn_and_emit(force_commit=True)
                        continue

                    seg_np = np.concatenate(list(q_audio), dtype=np.float32)
                    if seg_np.size < int(0.025 * sr):
                        if force_commit:
                            run_punc_itn_and_emit(force_commit=True)
                        continue

                    seg = torch.from_numpy(seg_np).unsqueeze(0).to(device) * 32768.0
                    x = kaldi.fbank(
                        seg, num_mel_bins=80, frame_length=25, frame_shift=10,
                        dither=0.0, energy_floor=0.0, sample_frequency=sr
                    ).unsqueeze(0).to(device)
                    x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                    use_cuda = (device.type == "cuda")
                    ctx = torch.amp.autocast(device_type='cuda',
                                             dtype=torch.float16) if use_cuda else contextlib.nullcontext()
                    with ctx:
                        enc_out, enc_len, _, att_cache, cnn_cache, offset = \
                            self.model.encoder.forward_parallel_chunk(
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

                        T = int(enc_len.item())
                        enc_out = enc_out.reshape(1, -1, enc_out.size(-1))[:, :T]
                        if enc_out.size(1) > chunk_size:
                            enc_out = enc_out[:, :chunk_size]
                        offset = offset - T + enc_out.size(1)

                        hyp_step = self.model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                    chunk_text = get_output([hyp_step], self.char_dict)[0]
                    ovl = longest_suffix_prefix_overlap(raw_text, chunk_text, max_k=max_overlap_match)
                    if ovl < len(chunk_text):
                        raw_text = raw_text + chunk_text[ovl:]

                    # chỉ gọi pipeline khi raw thay đổi hoặc force_commit
                    if raw_text != last_raw_sent or force_commit:
                        last_raw_sent = raw_text
                        run_punc_itn_and_emit(force_commit=force_commit)

                    # trượt cửa sổ
                    if q_audio:
                        q_audio.popleft()

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[ASR] error: {e}", flush=True)
        finally:
            # Flush tail giống logic main mới
            if punc_model is not None:
                tail_raw = raw_text[committed_ptr:]
                if tail_raw.strip():
                    try:
                        context_text = committed_text_punctuated_context[-context_buffer_size:]
                        final_raw_with_context = (context_text + " " + tail_raw).strip()
                        final_punc_full = punc_model.infer([final_raw_with_context])[0]

                        if context_text and final_punc_full.startswith(context_text):
                            final_punc_text = final_punc_full[len(context_text):].strip()
                        else:
                            final_punc_text = tail_raw

                        final_itn_text = inverse_normalize_local(final_punc_text).strip()

                        if committed_text_normalized_display:
                            final_text = (committed_text_normalized_display + " " + final_itn_text).strip()
                        else:
                            final_text = final_itn_text
                    except Exception as e:
                        print(f"[FINAL PUNC/ITN] error: {e}", flush=True)
                        final_text = (committed_text_normalized_display + " " + tail_raw).strip()
                else:
                    final_text = committed_text_normalized_display.strip()
            else:
                # không punc/itn
                final_text = raw_text.strip()

            _emit("final_flush", {"text": final_text})

            if return_final:
                return final_text
            return ""

    def run_chunkformer_stt(self,audio_path):
        cmd = [
            sys.executable, "api/services/chunkformer/decode.py" ,
            "--model_checkpoint", CHUNKFORMER_CHECKPOINT,
            "--long_form_audio", audio_path,
            "--total_batch_duration", "18000",
            "--chunk_size", "64",
            "--left_context_size", "128",
            "--right_context_size", "128",
            "--device", "cuda" ,
            "--autocast_dtype", "fp16",
        ]
        # capture_output=True để lấy transcript về
        result = subprocess.run(cmd, text=True, capture_output=True)
        # print(result.stdout)
        stederr = None
        if result.stderr:
            stederr = result.stderr
        return {
            "transcribe_by_sentence": result.stdout,
            "stederr": stederr,
        }