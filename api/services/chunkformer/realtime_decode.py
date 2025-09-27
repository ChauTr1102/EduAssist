import os
import math
import argparse
import yaml
import torch
import torchaudio
import pandas as pd
import jiwer

from tqdm import tqdm
from colorama import Fore, Style
import torchaudio.compliance.kaldi as kaldi
import sounddevice as sd
import numpy as np
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output_with_timestamps, get_output, milliseconds_to_hhmmssms


@torch.no_grad()
def init(model_checkpoint):
    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(config)
    model.eval()
    load_checkpoint(model, checkpoint_path)

    model.encoder = model.encoder.cuda()
    model.ctc = model.ctc.cuda()

    symbol_table = read_symbol_table(symbol_table_path)
    char_dict = {v: k for k, v in symbol_table.items()}

    return model, char_dict


def _compute_rel_right_context_frames(chunk_size_enc, right_context_size, conv_lorder, num_layers, subsampling):
    r_enc = max(right_context_size, conv_lorder)
    rrel_enc = r_enc + max(chunk_size_enc, r_enc) * (num_layers - 1)
    return rrel_enc * subsampling  # đổi sang số frame 10ms trước subsampling


@torch.no_grad()
def stream_audio(args, model, char_dict):
    """
    Giả lập streaming với audio: cắt 0.5s, thêm lookahead theo relative right context, đẩy qua encoder với cache.
    """
    subsampling = model.encoder.embed.subsampling_factor  # thường = 8
    num_layers = model.encoder.num_blocks
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # chunk_size theo không gian encoder-steps (sau subsampling)
    if args.stream_chunk_sec is not None:
        enc_steps = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))  # 0.5s ~ 50 frames -> ~6 steps
        chunk_size = enc_steps
    else:
        chunk_size = args.chunk_size

    left_context_size = args.left_context_size
    right_context_size = args.right_context_size

    rel_right_frames = _compute_rel_right_context_frames(
        chunk_size_enc=chunk_size,
        right_context_size=right_context_size,
        conv_lorder=conv_lorder,
        num_layers=num_layers,
        subsampling=subsampling,
    )

    # cache
    att_cache = torch.zeros(
        (num_layers, left_context_size, model.encoder.attention_heads,
         model.encoder._output_size * 2 // model.encoder.attention_heads)
    ).cuda()
    cnn_cache = torch.zeros((num_layers, model.encoder._output_size, conv_lorder)).cuda()
    offset = torch.zeros(1, dtype=torch.int, device="cuda")

    # audio
    wav_path = args.long_form_audio
    waveform, sample_rate = torchaudio.load(wav_path)
    # mono + 16kHz
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    assert sample_rate == 16000, "Yêu cầu audio 16 kHz"
    waveform = waveform * (1 << 15)

    hop_samples = int(args.stream_chunk_sec * sample_rate)  # ví dụ 0.5s
    # lookahead_samples = int((rel_right_frames * 0.01) * sample_rate)

    produced_steps = 0
    all_tokens = []
    cur = 0
    while cur < waveform.size(1):
        # comment cái này khi cần lookahead

        seg_end = min(cur + hop_samples, waveform.size(1))
        # seg_end_with_look = min(seg_end + lookahead_samples, waveform.size(1))

        # seg = waveform[:, cur:seg_end_with_look]
        seg = waveform[:, cur:seg_end]
        # đảm bảo đủ tối thiểu 25ms cho fbank
        min_samp = int(0.025 * sample_rate)  # 400 tại 16kHz
        if seg.size(1) < min_samp:
            # chọn 1: bỏ mẩu quá ngắn
            break
            # hoặc chọn 2: pad 0 để đủ 25ms
            # seg = torch.nn.functional.pad(seg, (0, min_samp - seg.size(1)))

        x = kaldi.fbank(
            seg,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000
        ).unsqueeze(0)  # (1, T_fbank, 80)

        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).cuda()

        truncated_context_size = chunk_size  # chỉ xuất đúng số bước hữu ích
        encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
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

        hyp_step = model.encoder.ctc_forward(encoder_outs).squeeze(0)
        all_tokens.append(hyp_step.cpu())

        # thời gian hiển thị theo số bước encoder đã tạo
        seg_start_ms = int(produced_steps * subsampling * 10)
        seg_end_ms = int((produced_steps + hyp_step.numel()) * subsampling * 10)
        produced_steps += hyp_step.numel()

        text_step = get_output([hyp_step], char_dict)[0]
        print(
            f"{Fore.CYAN}{milliseconds_to_hhmmssms(seg_start_ms)}{Style.RESET_ALL}"
            f" - "
            f"{Fore.CYAN}{milliseconds_to_hhmmssms(seg_end_ms)}{Style.RESET_ALL}"
            f": {text_step}"
        )

        cur = seg_end
        torch.cuda.empty_cache()

    if args.print_final:
        hyps = torch.cat(all_tokens) if all_tokens else torch.tensor([], dtype=torch.long)
        final_decode = get_output_with_timestamps([hyps], char_dict)[0]
        print("\n=== Final (merged) ===")
        for item in final_decode:
            start = f"{Fore.RED}{item['start']}{Style.RESET_ALL}"
            end = f"{Fore.RED}{item['end']}{Style.RESET_ALL}"
            print(f"{start} - {end}: {item['decode']}")


@torch.no_grad()
def endless_decode(args, model, char_dict):
    # Giữ mode cũ cho long audio theo lô lớn
    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n - 1)

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

        encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
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
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
            encoder_outs = encoder_outs[:, :truncated_context_size]
        offset = offset - encoder_lens + encoder_outs.shape[1]

        hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
        hyps.append(hyp)
        torch.cuda.empty_cache()
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
            break
    hyps = torch.cat(hyps)
    decode = get_output_with_timestamps([hyps], char_dict)[0]

    for item in decode:
        start = f"{Fore.RED}{item['start']}{Style.RESET_ALL}"
        end = f"{Fore.RED}{item['end']}{Style.RESET_ALL}"
        print(f"{start} - {end}: {item['decode']}")


@torch.no_grad()
def batch_decode(args, model, char_dict):
    df = pd.read_csv(args.audio_list, sep="\t")

    max_length_limited_context = args.max_duration
    max_length_limited_context = int((max_length_limited_context // 0.01)) // 2
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size

    decodes = []
    xs = []
    xs_origin_lens = []
    max_frames = max_length_limited_context

    for idx, wav_path in tqdm(enumerate(df['wav'].to_list())):
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform * (1 << 15)
        x = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000
        )

        xs.append(x)
        xs_origin_lens.append(x.shape[0])
        max_frames -= xs_origin_lens[-1]

        if (max_frames <= 0) or (idx == len(df) - 1):
            xs_origin_lens = torch.tensor(xs_origin_lens, dtype=torch.int, device="cuda")
            offset = torch.zeros(len(xs), dtype=torch.int, device="cuda")
            encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(
                xs=xs,
                xs_origin_lens=xs_origin_lens,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
                offset=offset
            )

            hyps = model.encoder.ctc_forward(encoder_outs, encoder_lens, n_chunks)
            decodes += get_output(hyps, char_dict)

            xs = []
            xs_origin_lens = []
            max_frames = max_length_limited_context

    df['decode'] = decodes
    if "txt" in df:
        wer = jiwer.wer(df["txt"].to_list(), decodes)
        print("WER: ", wer)
    df.to_csv(args.audio_list, sep="\t", index=False)

@torch.no_grad()
@torch.no_grad()
def stream_mic(args, model, char_dict):
    import sounddevice as sd
    import numpy as np
    from model.utils.ctc_utils import get_output, milliseconds_to_hhmmssms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subsampling = model.encoder.embed.subsampling_factor
    num_layers = model.encoder.num_blocks
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # chunk_size theo encoder-steps từ stream_chunk_sec
    enc_steps = max(1, int(round((args.stream_chunk_sec / 0.01) / subsampling)))
    chunk_size = enc_steps
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size  # nên đặt 0 cho low-latency

    # cache để giữ LEFT CONTEXT giữa các bước
    att_cache = torch.zeros(
        (num_layers, left_context_size, model.encoder.attention_heads,
         model.encoder._output_size * 2 // model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros((num_layers, model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    # cấu hình thu âm
    sr = args.mic_sr
    assert sr == 16000, "Mic nên 16 kHz để khớp feature"
    block_samples = int(args.stream_chunk_sec * sr)

    produced_steps = 0
    print("Mic streaming... Ctrl+C để dừng.")
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_samples) as stream:
        while True:
            audio_block, _ = stream.read(block_samples)        # (N, 1)
            a = np.squeeze(audio_block, axis=1).astype(np.float32)  # (N,)

            # to torch mono 16k, scale như pipeline gốc
            seg = torch.from_numpy(a).unsqueeze(0).to(device)  # (1, T)
            seg = seg * (1 << 15)

            # đảm bảo đủ 25ms cho fbank (phòng block đầu quá ngắn nếu driver không chuẩn)
            min_samp = int(0.025 * sr)
            if seg.size(1) < min_samp:
                continue

            x = kaldi.fbank(
                seg,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=16000
            ).unsqueeze(0)  # (1, T_fbank, 80)
            x = x.to(device)
            x_len = torch.tensor([x.size(1)], dtype=torch.int, device=device)

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

            text_step = get_output([hyp_step], char_dict)[0]
            print(f"{milliseconds_to_hhmmssms(seg_start_ms)} - {milliseconds_to_hhmmssms(seg_end_ms)}: {text_step}")



def main():
    parser = argparse.ArgumentParser(description="ChunkFormer decoding")

    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--max_duration", type=int, default=1800, help="Max audio seconds GPU can handle at once")
    parser.add_argument("--chunk_size", type=int, default=64, help="Encoder chunk size (steps after subsampling)")
    parser.add_argument("--left_context_size", type=int, default=128, help="Left context size (encoder steps)")
    parser.add_argument("--right_context_size", type=int, default=128, help="Right context size (encoder steps)")

    # audio input
    parser.add_argument("--long_form_audio", type=str, help="Path to WAV/FLAC audio")

    # batch tsv
    parser.add_argument("--audio_list", type=str, default=None,
                        help="TSV path with 'wav' column. If 'txt' provided, compute WER")

    # streaming mode
    # parser.add_argument("--stream", action="store_true", help="Enable streaming simulation on --long_form_audio")
    # parser.add_argument("--stream_chunk_sec", type=float, default=0.5, help="Slice length in seconds")
    # parser.add_argument("--print_final", action="store_true", help="Print merged final decode with timestamps")

    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--autocast_dtype", type=str, default="fp32")
    parser.add_argument("--mic", action="store_true", help="Stream from microphone")
    parser.add_argument("--mic_sr", type=int, default=16000, help="Mic sample rate")
    parser.add_argument("--stream_chunk_sec", type=float, default=0.5, help="Slice length sec")
    parser.add_argument("--lookahead_sec", type=float, default=0.2, help="Extra audio for right-context")
    parser.add_argument("--print_final", action="store_true")
    # ... ngay sau các add_argument khác
    parser.add_argument("--stream", action="store_true", help="Enable streaming on --long_form_audio")

    args = parser.parse_args()

    print(f"Model Checkpoint: {args.model_checkpoint}")
    print(f"Chunk Size (encoder steps): {args.chunk_size}")
    print(f"Left Context Size: {args.left_context_size}")
    print(f"Right Context Size: {args.right_context_size}")
    print(f"Long Form Audio Path: {args.long_form_audio}")
    print(f"Audio List Path: {args.audio_list}")
    print(f"Streaming: {args.stream} | stream_chunk_sec: {args.stream_chunk_sec}")

    assert any([getattr(args, "mic", False), args.long_form_audio,
                args.audio_list]), "Cần --mic hoặc --long_form_audio hoặc --audio_list"

    model, char_dict = init(args.model_checkpoint)

    # thêm nhánh mic đứng trước
    if getattr(args, "mic", False):
        stream_mic(args, model, char_dict)
        return

    if args.stream and args.long_form_audio:
        stream_audio(args, model, char_dict)
    elif args.long_form_audio:
        endless_decode(args, model, char_dict)
    else:
        batch_decode(args, model, char_dict)


# if __name__ == "__main__":
#
#     import sys
#
#     # ghi cứng tham số để chạy trực tiếp
#     sys.argv = [
#         "decode.py",
#         "--model_checkpoint", "/home/trinhchau/code/chunkformer/chunkformer-large-vie",
#         "--long_form_audio", "/home/trinhchau/code/EduAssist/data/kinh_te_chinh_tri_2m_47s.MP3",
#         "--left_context_size", "128",
#         "--right_context_size", "0",
#         "--stream",
#         "--mic",
#         "--stream_chunk_sec", "0.5",
#         "--print_final"
#     ]
#
#     main()
if __name__ == "__main__":

    import sys

    # ghi cứng tham số để chạy trực tiếp (streaming từ mic)
    sys.argv = [
        "decode.py",
        "--model_checkpoint", "/home/trinhchau/code/chunkformer/chunkformer-large-vie",
        "--mic",
        "--mic_sr", "16000",
        "--left_context_size", "128",
        "--right_context_size", "8",
        "--stream_chunk_sec", "1",
        "--print_final"
    ]

    main()
