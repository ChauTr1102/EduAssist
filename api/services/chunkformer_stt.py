from api.services import *
import subprocess, sys
import time

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

# ==================== Utils for stable streaming without CIF ====================

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

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

def _compute_rel_right_context_frames(chunk_size_enc, right_context_size, conv_lorder, num_layers, subsampling):
    r_enc = max(right_context_size, conv_lorder)
    rrel_enc = r_enc + max(chunk_size_enc, r_enc) * (num_layers - 1)
    return rrel_enc * subsampling  # đổi sang số frame 10ms trước subsampling


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

def _kaldi_fbank_cpu(wav_1xT: torch.Tensor) -> torch.Tensor:
    return kaldi.fbank(
        wav_1xT.cpu(),
        num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000
    ).unsqueeze(0)  # (1, T, 80)
# ==================== Single-line writer ====================

class _LineWriter:
    def __init__(self):
        self.prev_len = 0
    def write(self, s: str):
        s = s.replace("\n", " ")
        sys.stdout.write("\r" + s)
        # xóa phần thừa nếu ngắn hơn lần trước
        extra = self.prev_len - len(s)
        if extra > 0:
            sys.stdout.write(" " * extra)
            sys.stdout.write("\r" + s)
        sys.stdout.flush()
        self.prev_len = len(s)

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
    def stream_mic(self, stream_chunk_sec: float,
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
                    ov = _longest_suffix_prefix_overlap(carry_text, chunk_text, max_k=32)
                    merged = carry_text + chunk_text[ov:]
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

    def stream_mic_final(self, stream_chunk_sec: float, left_context_size: int, right_context_size: int,
                         mic_sr, lookahead_sec: float, stable_reserve_words: int,
                         use_gpu_mel, lcs_window: int, lcs_min, idle_flush_chunks, max_tail_chars,
                         punct_flush=True, show_tail=False, adaptive_overlap_thresh: float=0.35,
                         on_update=None):
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


        def _split_commit_tail_final(text: str, reserve_last_k_words: int = 1):
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
            L, _ea, eb = _longest_common_substring(prev_tail[-lcs_window:], cur_text[:lcs_window])
            if L >= lcs_min:
                return prev_tail + cur_text[eb:]  # bỏ trùng
            return prev_tail + cur_text

        def _adaptive_reserve_words(overlap_ratio: float, base=1, hard=2, thresh=0.35):
            return hard if overlap_ratio < thresh else base

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subsampling = self.model.encoder.embed.subsampling_factor
        num_layers = self.model.encoder.num_blocks
        conv_lorder = self.model.encoder.cnn_module_kernel // 2

        enc_steps = max(1, int(round((stream_chunk_sec / 0.01) / subsampling)))
        chunk_size = enc_steps

        att_cache = torch.zeros(
            (num_layers, left_context_size, self.model.encoder.attention_heads,
             self.model.encoder._output_size * 2 // self.model.encoder.attention_heads),
            device=device
        )
        cnn_cache = torch.zeros((num_layers, self.model.encoder._output_size, conv_lorder), device=device)
        offset = torch.zeros(1, dtype=torch.int, device=device)

        sr = mic_sr
        assert sr == 16000, "Mic nên 16 kHz"
        block = int(stream_chunk_sec * sr)
        look_blocks = max(0, int(math.ceil(lookahead_sec / stream_chunk_sec)))

        q = deque()
        full_text = ""  # đã commit
        carry_text = ""  # đuôi chưa commit
        last_snapshot = ""
        idle_counter = 0

        fe = GPUMel80(device, sr) if use_gpu_mel else None
        line = _LineWriter()

        # không in dòng hướng dẫn để giữ “một hàng”
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block) as stream:
            while True:
                audio_block, _ = stream.read(block)
                a = np.squeeze(audio_block, axis=1).astype(np.float32)
                q.append(a)

                if len(q) < 1 + look_blocks:
                    # cập nhật hiển thị ngay cả khi chưa đủ lookahead
                    if show_tail and (full_text or carry_text):
                        line.write(full_text + carry_text)
                    continue

                seg_np = np.concatenate([q[0]] + list(list(q)[1:1 + look_blocks]))
                seg = torch.from_numpy(seg_np).unsqueeze(0).to(device)
                seg = seg * (1 << 15)
                if seg.size(1) < int(0.025 * sr):
                    q.popleft()
                    continue

                x = _kaldi_fbank_cpu(seg).to(device) if fe is None else fe(seg)
                x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    enc_out, enc_len, _, att_cache, cnn_cache, offset = self.model.encoder.forward_parallel_chunk(
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
                    hyp_step = self.model.encoder.ctc_forward(enc_out).squeeze(0).cpu()

                chunk_text = get_output([hyp_step], self.char_dict)[0]

                # smart merge
                merged = _smart_merge(carry_text, chunk_text, lcs_window=lcs_window, lcs_min=lcs_min)
                overlap_used = len(carry_text) + len(chunk_text) - len(merged)
                overlap_ratio = 0.0 if len(chunk_text) == 0 else max(0.0,
                                                                     min(1.0, overlap_used / max(1, len(chunk_text))))
                reserve_words = _adaptive_reserve_words(overlap_ratio,
                                                        base=stable_reserve_words,
                                                        hard=stable_reserve_words + 1,
                                                        thresh=adaptive_overlap_thresh)
                commit, new_tail = _split_commit_tail_final(merged, reserve_last_k_words=reserve_words)

                progressed = bool(commit) or (new_tail != last_snapshot)
                idle_counter = 0 if progressed else (idle_counter + 1)
                last_snapshot = new_tail

                # cập nhật buffer
                if commit:
                    full_text += commit

                # ép commit đuôi theo các tiêu chí nhanh
                do_force = False
                if punct_flush and re.search(r"[.!?…。，、！？；：]\s*$", new_tail):
                    do_force = True
                if len(new_tail) >= max_tail_chars:
                    do_force = True
                if idle_counter >= idle_flush_chunks and new_tail.strip():
                    do_force = True

                if do_force and new_tail.strip():
                    # giữ khoảng trắng nếu có
                    full_text += new_tail
                    new_tail = ""
                    idle_counter = 0

                carry_text = new_tail

                # one-line update
                if show_tail:
                    display = (full_text + carry_text).strip()
                else:
                    display = full_text.strip()
                line.write(display)

                q.popleft()

