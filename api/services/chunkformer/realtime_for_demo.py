# realtime_cif_mic.py
import os
import re
import math
import argparse
import yaml
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np

# mic
import sounddevice as sd

# ====== repo imports (giữ nguyên theo repo của bạn) ======
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.cif import CifMiddleware
from model.cif_head import CifCtcHead
# ========================================================


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
    tail_handling_firing_threshold = 0.5


# ---------------- Utilities ----------------
def build_char_maps(char_dict):
    id2ch = char_dict
    ch2id = {ch: i for i, ch in id2ch.items()}
    blank_id = None
    for k, v in id2ch.items():
        if v in ("<blk>", "<blank>", "<ctc_blank>"):
            blank_id = k
            break
    if blank_id is None:
        blank_id = 0
    unk_id = None
    for k, v in id2ch.items():
        if v in ("<unk>", "<UNK>"):
            unk_id = k
            break
    return id2ch, ch2id, blank_id, unk_id


def maybe_resume_cif(cif: CifMiddleware, cif_ctc: CifCtcHead, resume_path: str):
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        if "cif" in ckpt:
            cif.load_state_dict(ckpt["cif"])
        if "cif_ctc" in ckpt:
            cif_ctc.load_state_dict(ckpt["cif_ctc"])
        print(f"[resume] loaded CIF from {resume_path}")


def ids_to_text(ids, id2ch):
    return "".join(id2ch[i] for i in ids if i in id2ch)


def ctc_greedy_collapse_with_memory(frame_ids: torch.Tensor, blank_id: int, last_nonblank_id: int | None):
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


def fmt_ts_from_frames(fr_10ms: int):
    ms = fr_10ms * 10
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d}:{ms:03d}"


@torch.no_grad()
def init_model_and_vocab(model_checkpoint: str, device: torch.device):
    cfg_path = os.path.join(model_checkpoint, "config.yaml")
    ckpt_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    vocab_path = os.path.join(model_checkpoint, "vocab.txt")

    with open(cfg_path, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(cfg, cfg_path)
    model.eval()
    load_checkpoint(model, ckpt_path)

    model.encoder = model.encoder.to(device)
    if hasattr(model, "ctc"):
        model.ctc = model.ctc.to(device)

    symbol_table = read_symbol_table(vocab_path)
    char_dict = {v: k for k, v in symbol_table.items()}  # id->char
    return model, char_dict


# ---------------- Mic + CIF streaming ----------------
@torch.no_grad()
def stream_mic_with_cif(args):
    device = torch.device(args.device)
    assert args.mic_sr == 16000, "Mic phải 16 kHz."

    # model + vocab
    model, char_dict = init_model_and_vocab(args.model_checkpoint, device)
    id2ch, ch2id, blank_id, _ = build_char_maps(char_dict)

    # CIF modules
    cif_cfg = CIFCfg()
    cif_cfg.encoder_embed_dim = model.encoder._output_size
    cif_cfg.cif_embedding_dim = model.encoder._output_size
    cif = CifMiddleware(cif_cfg).to(device)
    cif_ctc = CifCtcHead(cif_cfg.cif_embedding_dim, vocab_size=len(id2ch), blank_id=blank_id).to(device)
    if args.cif_ckpt:
        maybe_resume_cif(cif, cif_ctc, args.cif_ckpt)

    # encoder params
    subsampling = getattr(model.encoder.embed, "subsampling_factor", 4)
    num_layers = model.encoder.num_blocks
    conv_lorder = model.encoder.cnn_module_kernel // 2

    hop_sec = float(args.stream_hop_sec)
    frames_per_hop = max(1, int(round(hop_sec / 0.01)))          # 10ms/frame
    chunk_size = max(1, int(math.ceil(frames_per_hop / subsampling)))  # encoder steps/hop

    left_context_size = int(args.left_context_size)
    right_context_size = int(args.right_context_size)

    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n - 1)

    rel_right_context_frames = get_max_input_context(
        chunk_size, max(right_context_size, conv_lorder), num_layers
    ) * subsampling  # trên miền fbank-frames

    # encoder caches
    att_cache = torch.zeros(
        (num_layers, left_context_size, model.encoder.attention_heads,
         model.encoder._output_size * 2 // model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros((num_layers, model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    # CIF carry + CTC boundary memory
    cif_carry = None
    last_nonblank_id = None

    # hop timing
    produced_hop_frames = 0

    # mic stream
    block_samples = int(hop_sec * args.mic_sr)
    lookahead_samples = int(max(0.0, args.lookahead_sec) * args.mic_sr)
    GROUP_K = int(args.group_k)

    # small ring buffer for lookahead
    la_buf = np.zeros((0,), dtype=np.float32)

    print(f"[mic] hop={hop_sec:.3f}s | enc_step/hop={chunk_size} | subsampling={subsampling}")
    print("[mic] bắt đầu. Ctrl+C để dừng.")
    with sd.InputStream(samplerate=args.mic_sr, channels=1, dtype="float32", blocksize=block_samples) as stream:
        while True:
            # read core hop
            block_core, _ = stream.read(block_samples)  # (N,1)
            core = np.squeeze(block_core, axis=1).astype(np.float32)

            # optional lookahead: accumulate until đủ lookahead_samples
            if lookahead_samples > 0:
                need = max(0, lookahead_samples - la_buf.shape[0])
                if need > 0:
                    add, _ = stream.read(need)
                    add = np.squeeze(add, axis=1).astype(np.float32)
                    la_buf = np.concatenate([la_buf, add], axis=0)
                use_la = la_buf[:lookahead_samples]
                la_buf = la_buf[lookahead_samples:]
                mono = np.concatenate([core, use_la], axis=0)
            else:
                mono = core

            # to torch mono 16k, scale như pipeline
            seg = torch.from_numpy(mono).unsqueeze(0).to(device)  # (1,T)
            seg = seg * (1 << 15)

            # đảm bảo tối thiểu 25ms
            if seg.size(1) < int(0.025 * args.mic_sr):
                continue

            # fbank
            x = kaldi.fbank(
                seg,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=16000
            ).unsqueeze(0)  # (1, T_fbank, 80)
            x_len = torch.tensor([x.shape[1]], dtype=torch.int, device=device)

            # một hop encoder đúng chunk_size; append lookahead theo rel_right_context_frames
            # x đã bao gồm lookahead trên waveform; không cần cộng thêm ở đây
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
            B, _, C = enc_out.shape
            Tenc = int(enc_len if isinstance(enc_len, int) else int(enc_len.item()))
            enc_out = enc_out.reshape(1, -1, C)[:, :Tenc]
            if enc_out.shape[1] > truncated_context_size:
                enc_out = enc_out[:, :truncated_context_size]
            offset = offset - Tenc + enc_out.shape[1]

            # ===== CIF integrate-and-fire trên output của hop này =====
            Te = enc_out.shape[1]
            enc_pad = torch.zeros(1, Te, dtype=torch.bool, device=device)
            cif_in = {"encoder_raw_out": enc_out, "encoder_padding_mask": enc_pad}
            cif_pack = cif(
                cif_in,
                target_lengths=None,
                carry=cif_carry,
                flush_tail=False,   # mic: không flush tail trừ khi bạn tự quyết định
            )
            cif_carry = cif_pack["carry"]
            fires = cif_pack["fire_mask_per_frame"][0].bool().tolist()

            # spans từ fire mask
            spans = []
            s0 = 0
            for i, f in enumerate(fires):
                if f:
                    spans.append((s0, i))
                    s0 = i + 1
            # không flush tail ở mic; phần dư để lại cho lần sau

            # gộp K spans thành super-spans
            super_spans = []
            if spans:
                for i in range(0, len(spans), GROUP_K):
                    s = spans[i][0]
                    e = spans[min(i + GROUP_K - 1, len(spans) - 1)][1]
                    if e >= s:
                        super_spans.append((s, e))

            # decode từng super-span bằng CTC head encoder
            hop_text_parts = []
            for (s, e) in super_spans:
                seg_feats = enc_out[:, s:e + 1, :]
                hyp_ids_frame = model.encoder.ctc_forward(seg_feats).squeeze(0)   # (L,)
                out_ids, last_nonblank_id = ctc_greedy_collapse_with_memory(
                    frame_ids=hyp_ids_frame,
                    blank_id=blank_id,
                    last_nonblank_id=last_nonblank_id,
                )
                if not out_ids:
                    continue
                text = ids_to_text(out_ids, id2ch)
                text = re.sub(r"\s+", " ", text.replace("▁", " ")).strip()
                if text:
                    hop_text_parts.append(text)

            # in theo hop cố định (dù rỗng)
            start_ms_str = fmt_ts_from_frames(produced_hop_frames)
            produced_hop_frames += frames_per_hop
            end_ms_str = fmt_ts_from_frames(produced_hop_frames)
            hop_text = re.sub(r"\s+", " ", " ".join(hop_text_parts)).strip()
            print(f"{start_ms_str} - {end_ms_str}: {hop_text}")

            torch.cuda.empty_cache() if device.type == "cuda" else None


def main():
    p = argparse.ArgumentParser("Realtime mic streaming with CIF segmentation over ChunkFormer")
    p.add_argument("--model_checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--cif_ckpt", type=str, default=None, help="CIF checkpoint (.pt) đã train")
    # streaming
    p.add_argument("--stream_hop_sec", type=float, default=0.5)
    p.add_argument("--left_context_size", type=int, default=64)
    p.add_argument("--right_context_size", type=int, default=8)
    p.add_argument("--group_k", type=int, default=2, help="Gộp K spans trước khi decode")
    # mic
    p.add_argument("--mic_sr", type=int, default=16000)
    p.add_argument("--lookahead_sec", type=float, default=0.0, help="Nhìn trước trên waveform")
    args = p.parse_args()

    # log ngắn
    print(f"ckpt={args.model_checkpoint} | device={args.device} | hop={args.stream_hop_sec}s "
          f"| L={args.left_context_size} R={args.right_context_size} | K={args.group_k} | la={args.lookahead_sec}s")

    stream_mic_with_cif(args)


# run_realtime_cif.py
from types import SimpleNamespace


args = SimpleNamespace(
    model_checkpoint="/home/trinhchau/code/chunkformer/chunkformer-large-vie",
    cif_ckpt="/home/trinhchau/code/EduAssist/api/services/chunkformer/model/cif_best.pt",
    device="cuda",
    stream_hop_sec=1,
    left_context_size=32,
    right_context_size=2,
    group_k=2,
    mic_sr=16000,
    lookahead_sec=0.3,
)
stream_mic_with_cif(args)
