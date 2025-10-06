import os
import re
import time
import argparse

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
import yaml
from pydub import AudioSegment

# ====== repo imports ======
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.cif import CifMiddleware
# ==========================


# ---------------- CIF Config ----------------
class CIFCfg:
    cif_threshold = 1.0
    cif_embedding_dim = 512
    encoder_embed_dim = 512
    produce_weight_type = "conv"    # "conv" | "dense" | "linear"
    conv_cif_width = 5
    conv_cif_dropout = 0.1
    apply_scaling = True
    apply_tail_handling = True
    tail_handling_firing_threshold = 0.4


# ---------------- Wrapper chỉ dùng CIF ----------------
class CIFOnly(torch.nn.Module):
    def __init__(self, cif_cfg: CIFCfg):
        super().__init__()
        self.cif = CifMiddleware(cif_cfg)


def maybe_resume_cif(cif_module: CIFOnly, resume_path: str):
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        if "cif" in ckpt:
            cif_module.cif.load_state_dict(ckpt["cif"])
        else:
            # fallback: cho phép load trực tiếp state_dict
            cif_module.cif.load_state_dict(ckpt)
        print(f"[resume] loaded CIF from {resume_path}")


def build_char_maps(char_dict):
    # char_dict: id->char
    id2ch = char_dict
    ch2id = {ch: i for i, ch in id2ch.items()}

    # tìm blank
    blank_id = 0
    for k, v in id2ch.items():
        if v in ("<blk>", "<blank>", "<ctc_blank>"):
            blank_id = k
            break

    # tìm unk
    unk_id = None
    for k, v in id2ch.items():
        if v in ("<unk>", "<UNK>"):
            unk_id = k
            break

    return id2ch, ch2id, blank_id, unk_id


def ids_to_text(ids, id2ch):
    return "".join(id2ch[i] for i in ids if i in id2ch)


def load_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
    wav = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
    return wav  # (1, T)


# ---- Greedy collapse CTC với nhớ token cuối xuyên span/chunk ----
def ctc_greedy_collapse_with_memory(frame_ids, blank_id, last_nonblank_id):
    out = []
    prev = None
    for t in frame_ids.tolist():
        if t == blank_id:
            prev = None
            continue
        if t != prev:
            out.append(t)
            prev = t
    if out and last_nonblank_id is not None and out[0] == last_nonblank_id:
        out = out[1:]
    new_last = out[-1] if out else last_nonblank_id
    return out, new_last


# ---------------- Model init ----------------
@torch.no_grad()
def init_checkpoint(model_checkpoint, device):
    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

    with open(config_path, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(config, config_path)
    model.eval()
    load_checkpoint(model, checkpoint_path)
    model.encoder = model.encoder.to(device)
    if hasattr(model, "ctc"):
        model.ctc = model.ctc.to(device)

    symbol_table = read_symbol_table(symbol_table_path)
    char_dict = {v: k for k, v in symbol_table.items()}  # id->char
    return model, char_dict


# ---------------- Inference (streaming) ----------------
@torch.no_grad()
def stream_infer_with_cif(args):
    assert os.path.isfile(args.long_form_audio), "Missing --long_form_audio"
    assert os.path.isfile(args.cif_ckpt), "Missing --cif_ckpt"

    device = torch.device(args.device)
    model, char_dict = init_checkpoint(args.model_checkpoint, device)
    id2ch, ch2id, blank_id, _ = build_char_maps(char_dict)

    # CIF config khớp encoder dim
    cif_cfg = CIFCfg()
    cif_cfg.encoder_embed_dim = model.encoder._output_size
    cif_cfg.cif_embedding_dim = model.encoder._output_size

    cif_mod = CIFOnly(cif_cfg).to(device)
    maybe_resume_cif(cif_mod, args.cif_ckpt)
    cif_mod.eval()

    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n - 1)

    subsampling = model.encoder.embed.subsampling_factor
    chunk_size = args.chunk_size
    left_ctx = args.left_context_size
    right_ctx = args.right_context_size
    conv_lorder = model.encoder.cnn_module_kernel // 2

    max_len_limited_ctx = int((args.total_batch_duration // 0.01)) // 2  # frame 10ms
    multiply_n = max(1, max_len_limited_ctx // chunk_size // subsampling)
    truncated_ctx = chunk_size * multiply_n

    rel_right_ctx = get_max_input_context(
        chunk_size, max(right_ctx, conv_lorder), model.encoder.num_blocks
    ) * subsampling

    # audio -> fbank
    waveform = load_audio(args.long_form_audio).to(device)
    xs = kaldi.fbank(
        waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000
    ).unsqueeze(0)  # (1, T_frames, 80)

    # caches
    att_cache = torch.zeros(
        (model.encoder.num_blocks, left_ctx, model.encoder.attention_heads,
         model.encoder._output_size * 2 // model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros((model.encoder.num_blocks, model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    results = []
    last_nonblank_id = None
    cif_carry = None

    total_frames = xs.shape[1]
    step = truncated_ctx * subsampling
    idx = 0
    while True:
        start = step * idx
        if start >= total_frames:
            break
        end = min(step * (idx + 1) + 7, total_frames)

        x = xs[:, start:end + rel_right_ctx]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int, device=device)

        # Encoder streaming
        enc_out, enc_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
            xs=x,
            xs_origin_lens=x_len,
            chunk_size=chunk_size,
            left_context_size=left_ctx,
            right_context_size=right_ctx,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            truncated_context_size=truncated_ctx,
            offset=offset
        )
        enc_out = enc_out.reshape(1, -1, enc_out.shape[-1])[:, :enc_len]
        if start + rel_right_ctx < total_frames:
            enc_out = enc_out[:, :truncated_ctx]
        offset = offset - enc_len + enc_out.shape[1]

        # CIF
        Te = enc_out.shape[1]
        enc_pad = torch.zeros(1, Te, dtype=torch.bool, device=enc_out.device)
        cif_in = {"encoder_raw_out": enc_out, "encoder_padding_mask": enc_pad}

        is_last = (start + rel_right_ctx) >= total_frames
        cif_pack = cif_mod.cif(
            cif_in,
            target_lengths=None,
            carry=cif_carry,
            flush_tail=is_last
        )
        cif_carry = cif_pack["carry"]

        fires = cif_pack["fire_mask_per_frame"][0].bool()  # (Te,)
        spans, s0 = [], 0
        for i, f in enumerate(fires.tolist()):
            if f:
                spans.append((s0, i))
                s0 = i + 1

        # Decode từng span với CTC frame-level của encoder
        for (s, e) in spans:
            if e < s:
                continue
            seg_feats = enc_out[:, s:e + 1, :]  # (1, L, C)
            frame_ids = model.encoder.ctc_forward(seg_feats).squeeze(0)  # (L,)
            out_ids, last_nonblank_id = ctc_greedy_collapse_with_memory(
                frame_ids, blank_id, last_nonblank_id
            )
            if not out_ids:
                continue
            text = ids_to_text(out_ids, id2ch)
            text = re.sub(r"\s+", " ", text.replace("▁", " ")).strip()
            if text:
                print(text)
                results.append(text)

        if is_last:
            break
        idx += 1

    return results


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Streaming inference with ChunkFormer + CIF")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total_batch_duration", type=int, default=1800)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--left_context_size", type=int, default=128)
    parser.add_argument("--right_context_size", type=int, default=128)
    parser.add_argument("--cif_ckpt", type=str, required=True)
    parser.add_argument("--long_form_audio", type=str, required=True)

    args = parser.parse_args()

    t0 = time.time()
    _ = stream_infer_with_cif(args)
    print(f"[done] elapsed: {round(time.time() - t0, 3)}s")


if __name__ == "__main__":
    import sys, glob
    print("exe:", sys.executable)
    print("CUDA avail:", torch.cuda.is_available(), "CUDA ver:", torch.version.cuda)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("nvidia devs:", glob.glob("/dev/nvidia*"))

    # Ví dụ:
    sys.argv = [
        "infer_cif.py",
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--device", "cuda",
        "--cif_ckpt", "/home/trinhchau/code/EduAssist/api/services/chunkformer/model/cif_best.pt",
        "--long_form_audio", "/home/trinhchau/code/EduAssist/data/AI_voice_2p.wav",
        "--total_batch_duration", "2",
        "--chunk_size", "2",
        "--left_context_size", "2",
        "--right_context_size", "2",
    ]
    main()
