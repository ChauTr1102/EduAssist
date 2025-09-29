import os
import re
import time
import argparse
from contextlib import nullcontext
import io

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from datasets import load_dataset, Audio

from pydub import AudioSegment
from tqdm import tqdm

# ====== repo imports (giữ nguyên theo repo của bạn) ======
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output_with_timestamps_1
from model.utils.ctc_utils import get_output_with_timestamps_1
from model.cif import CifMiddleware
from model.cif_head import CifCtcHead
# ========================================================


# ---------------- CIF Config ----------------
class CIFCfg:
    # bạn có thể đưa các giá trị này từ config.yaml
    cif_threshold = 1.0
    cif_embedding_dim = 512
    encoder_embed_dim = 512
    produce_weight_type = "conv"    # "conv" | "dense" | "linear"
    conv_cif_width = 5
    conv_cif_dropout = 0.1
    apply_scaling = True
    apply_tail_handling = True
    tail_handling_firing_threshold = 0.4


# ---------------- Runner: encoder -> CIF -> CTC ----------------
class EndlessRunner(nn.Module):
    def __init__(self, model, vocab_size, blank_id, cif_cfg: CIFCfg):
        super().__init__()
        self.model = model                       # có encoder (+ có thể có CTC_enc)
        self.cif = CifMiddleware(cif_cfg)
        self.cif_ctc = CifCtcHead(cif_cfg.cif_embedding_dim, vocab_size, blank_id)

        # loss weights
        self.lam_ctc_enc = 0.25     # phụ trên encoder (nếu có)
        self.lam_qua = 1.0          # quantity loss
        self.lam_ctc_cif = 1.0      # chính trên chuỗi CIF

    def forward_stream_step(self, x, x_len, caches, stream_args,
                            targets=None, target_lens=None, train_mode=True):
        """
        x: (B=1, T_in, feat)
        x_len: (B=1,)
        caches: (att_cache, cnn_cache, offset)
        stream_args: {chunk_size, left_context_size, right_context_size,
                      truncated_context_size, subsampling_factor, idx, rel_right_context_size}
        """
        self.train(train_mode)

        att_cache, cnn_cache, offset = caches

        encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = self.model.encoder.forward_parallel_chunk(
            xs=x,
            xs_origin_lens=x_len,
            chunk_size=stream_args["chunk_size"],
            left_context_size=stream_args["left_context_size"],
            right_context_size=stream_args["right_context_size"],
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            truncated_context_size=stream_args["truncated_context_size"],
            offset=offset
        )
        # reshape + cắt right-context out
        encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
        if stream_args["truncated_context_size"] * stream_args["subsampling_factor"] * stream_args["idx"] + stream_args["rel_right_context_size"] < x.shape[1]:
            encoder_outs = encoder_outs[:, :stream_args["truncated_context_size"]]
        offset = offset - encoder_lens + encoder_outs.shape[1]

        enc_lens = torch.tensor([encoder_outs.shape[1]], device=encoder_outs.device, dtype=torch.long)  # (1,)

        # ---- CIF ----
        if train_mode:
            assert targets is not None and target_lens is not None, "Need token targets for scaling & losses"

        # Chuẩn bị inputs cho CIF: encoder_raw_out + encoder_padding_mask (padded=1, valid=0)
        B, Te, D = encoder_outs.shape
        encoder_padding_mask = torch.zeros(B, Te, dtype=torch.bool, device=encoder_outs.device)

        encoder_inputs_for_cif = {
            "encoder_raw_out": encoder_outs,  # (B, T, C)
            "encoder_padding_mask": encoder_padding_mask  # (B, T), padded=1
        }

        cif_pack = self.cif(
            encoder_inputs_for_cif,
            target_lengths=(target_lens if train_mode else None),
        )
        cif_out = cif_pack["cif_out"]  # (B, Tc, Dc)
        cif_mask = cif_pack["cif_out_padding_mask"].to(torch.bool)  # (B, Tc) True=valid
        quantity_out = cif_pack["quantity_out"]  # (B,)

        # ---- Losses ----
        losses = {}
        total_loss = None

        if train_mode:
            # 1) CTC over CIF (chính)
            Tc = cif_mask.sum(-1).to(torch.int64)   # (B,)
            logp_TNC = self.cif_ctc.log_probs_TNC(cif_out)  # (Tc, B, V)
            # lengths bắt buộc là CPU int64
            input_lengths = Tc.cpu()
            target_lengths = target_lens.to(torch.int64).cpu()
            ctc_loss_cif = F.ctc_loss(
                logp_TNC, targets, input_lengths=input_lengths, target_lengths=target_lengths,
                blank=self.cif_ctc.blank_id, reduction='mean', zero_infinity=True
            )
            losses["ctc_cif"] = ctc_loss_cif

            # 2) Quantity loss: |sum(α) - S~|
            qua_loss = (quantity_out - target_lens.float()).abs().mean()
            losses["quantity"] = qua_loss

            # 3) CTC encoder (phụ) – nếu encoder có API cung cấp log_probs
            if hasattr(self.model.encoder, "ctc_log_probs"):
                logp_enc_TNC, enc_T = self.model.encoder.ctc_log_probs(encoder_outs)  # (Te,B,V), (B,)
                ctc_loss_enc = F.ctc_loss(
                    logp_enc_TNC, targets,
                    input_lengths=enc_T.to(torch.int64).cpu(),
                    target_lengths=target_lengths,
                    blank=self.cif_ctc.blank_id, reduction='mean', zero_infinity=True
                )
                losses["ctc_enc"] = ctc_loss_enc
            else:
                ctc_loss_enc = None

            total_loss = self.lam_ctc_cif * ctc_loss_cif + self.lam_qua * qua_loss
            if ctc_loss_enc is not None:
                total_loss = total_loss + self.lam_ctc_enc * ctc_loss_enc

        # ---- Decode (greedy) ----
        if not train_mode:
            hyps = self.cif_ctc.greedy_ctc(cif_out)  # List[List[int]]
        else:
            hyps = None

        new_caches = (att_cache, cnn_cache, offset)
        return {
            "encoder_outs": encoder_outs,
            "cif_out": cif_out,
            "cif_mask": cif_mask,
            "loss": total_loss,
            "loss_items": losses,
            "hyps": hyps,
            "caches": new_caches,
        }


# ---------------- Utilities ----------------
def save_cif_checkpoint(save_dir, step_or_epoch, runner, cif_cfg):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"cif_step_{step_or_epoch}.pt")
    payload = {
        "cif": runner.cif.state_dict(),
        "cif_ctc": runner.cif_ctc.state_dict(),
        "cif_cfg": {
            "cif_threshold": cif_cfg.cif_threshold,
            "cif_embedding_dim": cif_cfg.cif_embedding_dim,
            "encoder_embed_dim": cif_cfg.encoder_embed_dim,
            "produce_weight_type": cif_cfg.produce_weight_type,
            "conv_cif_width": cif_cfg.conv_cif_width,
            "conv_cif_dropout": cif_cfg.conv_cif_dropout,
            "apply_scaling": cif_cfg.apply_scaling,
            "apply_tail_handling": cif_cfg.apply_tail_handling,
            "tail_handling_firing_threshold": cif_cfg.tail_handling_firing_threshold,
        },
    }
    torch.save(payload, path)
    print(f"[save] CIF checkpoint -> {path}")
    return path


def maybe_resume_cif(runner, resume_path: str):
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=next(runner.parameters()).device)
        runner.cif.load_state_dict(ckpt["cif"])
        runner.cif_ctc.load_state_dict(ckpt["cif_ctc"])
        print(f"[resume] loaded CIF from {resume_path}")


def build_char_maps(char_dict):
    # char_dict: id->char (đọc từ vocab.txt theo init() của bạn)
    id2ch = char_dict
    ch2id = {ch: i for i, ch in id2ch.items()}

    # cố gắng bắt các token đặc biệt
    blank_id = None
    for k, v in id2ch.items():
        if v in ("<blk>", "<blank>", "<ctc_blank>"):
            blank_id = k
            break
    if blank_id is None:
        blank_id = 0  # fallback
    unk_id = None
    for k, v in id2ch.items():
        if v in ("<unk>", "<UNK>"):
            unk_id = k
            break

    return id2ch, ch2id, blank_id, unk_id


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s.lower())
    return s


def text_to_ids(s: str, ch2id: dict, unk_id: int | None):
    ids = []
    for ch in s:
        if ch in ch2id:
            ids.append(ch2id[ch])
        elif unk_id is not None:
            ids.append(unk_id)
        else:
            continue
    if len(ids) == 0:
        # tránh target rỗng khiến CTC lỗi
        ids = [list(ch2id.values())[0]]
    return torch.tensor(ids, dtype=torch.long)


def audio_to_fbank(audio_array: torch.Tensor, sr: int, target_sr: int = 16000):
    """
    audio_array: (T,) hoặc (1,T)
    return: (1, T_frames, 80)
    """
    wav = audio_array if audio_array.dim() == 1 else audio_array.squeeze(0)
    wav = wav.to(torch.float32)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.unsqueeze(0)  # (1, T)
    feats = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=target_sr
    ).unsqueeze(0)  # (1, T_frames, 80)
    return feats


def load_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)  # 16bit
    audio = audio.set_channels(1)      # mono
    audio = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
    return audio


@torch.no_grad()
def init(model_checkpoint, device):
    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

    with open(config_path, 'r') as fin:
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


# ---------------- Training (streaming) ----------------
def train_cif_streaming(args):
    device = torch.device(args.device)
    model, char_dict = init(args.model_checkpoint, device)
    id2ch, ch2id, blank_id, unk_id = build_char_maps(char_dict)

    cif_cfg = CIFCfg()
    # đảm bảo dim CIF khớp encoder_out
    cif_cfg.encoder_embed_dim = model.encoder._output_size
    cif_cfg.cif_embedding_dim = model.encoder._output_size

    runner = EndlessRunner(model, vocab_size=len(id2ch), blank_id=blank_id, cif_cfg=cif_cfg).to(device)
    maybe_resume_cif(runner, args.resume_cif)

    # freeze encoder để ổn định
    for p in runner.model.encoder.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(
        list(runner.cif.parameters()) + list(runner.cif_ctc.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    subsampling_factor = model.encoder.embed.subsampling_factor
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # dataset
    ds = load_dataset("doof-ferb/Speech-MASSIVE_vie", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    def pick_text(ex):
        for key in ["text", "transcription", "utt", "sentence", "label_text"]:
            if key in ex and isinstance(ex[key], str) and len(ex[key].strip()) > 0:
                return ex[key]
        return None

    runner.train(True)
    try:
        # PyTorch >= 2.x (nhưng không truyền device_type)
        scaler = torch.amp.GradScaler(
            enabled=(device.type == "cuda" and args.autocast_dtype in ("fp16", "bf16"))
        )
    except Exception:
        # Fallback cho bản cũ: dùng API torch.cuda.amp
        scaler = torch.cuda.amp.GradScaler(
            enabled=(device.type == "cuda" and args.autocast_dtype in ("fp16", "bf16"))
        )
    autocast_dtype = {
        "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, None: None
    }[args.autocast_dtype]

    epochs = args.epochs
    log_interval = 20
    global_utt = 0
    best_avg = float("inf")

    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n - 1)

    for epoch in range(1, epochs + 1):
        total = 0.0
        n_loss = 0
        for i, ex in enumerate(tqdm(ds, desc=f"epoch {epoch}")):
            # --- audio ---
            arr = None
            sr = 16000

            if "audio" in ex and isinstance(ex["audio"], dict):
                aud = ex["audio"]
                # 1) path cục bộ
                wav_path = aud.get("path", None)
                if isinstance(wav_path, str) and len(wav_path) > 0:
                    wav, sr = torchaudio.load(wav_path)
                    arr = wav.squeeze(0)
                # 2) bytes (khi không có path)
                elif aud.get("bytes", None) is not None:
                    fileobj = io.BytesIO(aud["bytes"])
                    wav, sr = torchaudio.load(fileobj)
                    arr = wav.squeeze(0)

            # 3) fallback nếu dataset có trường 'path' riêng
            if arr is None and "path" in ex and isinstance(ex["path"], str):
                wav, sr = torchaudio.load(ex["path"])
                arr = wav.squeeze(0)

            # Nếu vẫn không có audio thì bỏ mẫu
            if arr is None:
                # Optional: debug ngắn gọn để biết cấu trúc audio
                # print("skip sample; audio keys:", list(ex.get("audio", {}).keys()))
                continue

            x = audio_to_fbank(arr, sr=sr, target_sr=16000).to(device)  # (1,Tf,80)

            # --- text ---
            txt = pick_text(ex)
            if txt is None:
                continue
            y = text_to_ids(normalize_text(txt), ch2id, unk_id)  # (U,)
            y_len = torch.tensor([y.numel()], dtype=torch.long, device=device)
            y = y.unsqueeze(0).to(device)  # (1,U)

            # --- streaming params ---
            chunk_size = args.chunk_size
            left_context_size = args.left_context_size
            right_context_size = args.right_context_size

            max_length_limited_context = int((args.total_batch_duration // 0.01)) // 2
            multiply_n = max(1, max_length_limited_context // chunk_size // subsampling_factor)
            truncated_context_size = chunk_size * multiply_n

            rel_right_context_size = get_max_input_context(
                chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks
            ) * subsampling_factor

            att_cache = torch.zeros(
                (model.encoder.num_blocks, left_context_size, model.encoder.attention_heads,
                 model.encoder._output_size * 2 // model.encoder.attention_heads), device=device
            )
            cnn_cache = torch.zeros(
                (model.encoder.num_blocks, model.encoder._output_size, conv_lorder), device=device
            )
            offset = torch.zeros(1, dtype=torch.int, device=device)

            total_loss_one_utt = 0.0

            with torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype is not None else torch.autocast(device_type=device.type, enabled=False):
                for idx, _ in enumerate(range(0, x.shape[1], truncated_context_size * subsampling_factor)):
                    start = max(truncated_context_size * subsampling_factor * idx, 0)
                    end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, x.shape[1])
                    x_seg = x[:, start:end + rel_right_context_size]
                    x_len_seg = torch.tensor([x_seg[0].shape[0]], dtype=torch.int, device=device)

                    stream_args = dict(
                        chunk_size=chunk_size,
                        left_context_size=left_context_size,
                        right_context_size=right_context_size,
                        truncated_context_size=truncated_context_size,
                        subsampling_factor=subsampling_factor,
                        idx=idx,
                        rel_right_context_size=rel_right_context_size,
                    )

                    out = runner.forward_stream_step(
                        x=x_seg, x_len=x_len_seg,
                        caches=(att_cache, cnn_cache, offset),
                        stream_args=stream_args,
                        targets=y, target_lens=y_len,
                        train_mode=True
                    )
                    att_cache, cnn_cache, offset = out["caches"]

                    loss = out["loss"]
                    if loss is None:
                        continue

                    scaler.scale(loss).backward()
                    total_loss_one_utt += float(loss.detach().item())

                    if truncated_context_size * subsampling_factor * idx + rel_right_context_size >= x.shape[1]:
                        break

                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            total += total_loss_one_utt
            n_loss += 1
            global_utt += 1

            if global_utt % args.save_every == 0:
                save_cif_checkpoint(args.save_dir, f"utt{global_utt}", runner, cif_cfg)

            if i % log_interval == 0 and n_loss > 0:
                avg = total / max(1, n_loss)
                print(f"[epoch {epoch} step {i}] avg_loss={avg:.4f}")

        if n_loss > 0:
            epoch_avg = total / n_loss
            print(f"==> epoch {epoch} done. avg_loss={epoch_avg:.4f}")
            path = save_cif_checkpoint(args.save_dir, f"epoch{epoch}", runner, cif_cfg)
            if epoch_avg < best_avg:
                best_avg = epoch_avg
                best_path = os.path.join(args.save_dir, "cif_best.pt")
                torch.save(torch.load(path), best_path)
                print(f"[save] new best -> {best_path}")


# ---------------- Inference with CIF (A: Runner) ----------------
@torch.no_grad()
def stream_infer_with_runner(args):
    """
    Streaming: encoder -> CIF (tạo chuỗi h1..hU) -> CTC head của encoder -> decode + timestamps (giống legacy).
    """
    assert args.long_form_audio is not None, "Please provide --long_form_audio for inference."
    assert args.cif_ckpt is not None and os.path.isfile(args.cif_ckpt), "Please provide a valid --cif_ckpt."

    device = torch.device(args.device)
    model, char_dict = init(args.model_checkpoint, device)

    # CIF config khớp encoder dim
    cif_cfg = CIFCfg()
    cif_cfg.encoder_embed_dim = model.encoder._output_size
    cif_cfg.cif_embedding_dim = model.encoder._output_size

    # Runner chỉ để lấy self.cif (không dùng cif_ctc)
    runner = EndlessRunner(model, vocab_size=len(char_dict), blank_id=0, cif_cfg=cif_cfg).to(device)
    maybe_resume_cif(runner, args.cif_ckpt)
    runner.eval()

    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n - 1)

    subsampling_factor = model.encoder.embed.subsampling_factor
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # max length
    max_length_limited_context = int((args.total_batch_duration // 0.01)) // 2  # in 10ms
    multiply_n = max(1, max_length_limited_context // chunk_size // subsampling_factor)
    truncated_context_size = chunk_size * multiply_n

    # relative right context (frames)
    rel_right_context_size = get_max_input_context(
        chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks
    ) * subsampling_factor

    # audio -> fbank
    waveform = load_audio(args.long_form_audio).to(device)
    xs = kaldi.fbank(
        waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000
    ).unsqueeze(0)  # (1, T_frames, 80)

    # caches
    att_cache = torch.zeros(
        (model.encoder.num_blocks, left_context_size, model.encoder.attention_heads,
         model.encoder._output_size * 2 // model.encoder.attention_heads),
        device=device
    )
    cnn_cache = torch.zeros((model.encoder.num_blocks, model.encoder._output_size, conv_lorder), device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)
    cif_carry = None
    # loop giống legacy
    for idx, _ in tqdm(list(enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)))):
        start = max(truncated_context_size * subsampling_factor * idx, 0)
        end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs.shape[1])

        x = xs[:, start:end + rel_right_context_size]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int, device=device)

        # Encoder streaming
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
        # (1, T_enc, C)
        encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
        if truncated_context_size * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
            encoder_outs = encoder_outs[:, :truncated_context_size]
        offset = offset - encoder_lens + encoder_outs.shape[1]

        # === CIF từ encoder_outs ===
        B, Te, _ = encoder_outs.shape
        encoder_padding_mask = torch.zeros(1, Te, dtype=torch.bool, device=encoder_outs.device)
        cif_in = {"encoder_raw_out": encoder_outs, "encoder_padding_mask": encoder_padding_mask}

        is_last = (truncated_context_size * subsampling_factor * idx + rel_right_context_size) >= xs.shape[1]
        # if cif_carry is not None:
        #     print(f"[chunk {idx}] residual_w={float(cif_carry['res_w'].view(-1)[0]):.3f}")
        cif_pack = runner.cif(
            cif_in,
            target_lengths=None,
            carry=cif_carry,  # <-- truyền carry vào
            flush_tail=is_last  # <-- chỉ flush ở cuối thật
        )
        cif_carry = cif_pack["carry"]  # <-- lưu carry mới

        cif_out = cif_pack["cif_out"]  # (1, Tc, C) cho chunk này (đã bao gồm “nối” nếu fire)

        fires = cif_pack["fire_mask_per_frame"][0].bool()  # (Te,)
        # Chuyển fire positions -> segment spans
        # Ví dụ: mỗi fire đánh dấu KẾT THÚC 1 đơn vị. Ta gom frame từ start..i
        spans = []
        start = 0
        for i, f in enumerate(fires.tolist()):
            if f:
                spans.append((start, i))  # inclusive
                start = i + 1


        # === Dùng CTC head của encoder cho CIF output (để hợp với get_output_with_timestamps_1) ===
        # hyp = model.encoder.ctc_forward(cif_out).squeeze(0)  # (Tc,)
        # decode = get_output_with_timestamps_1(hyp, char_dict)  # giữ nguyên cách tách câu + timestamp
        #
        # for item in decode:
        #     print(f"{item['decode']}")
        emp = []
        for (s, e) in spans:

            if e < s:
                continue
            # LẤY encoder_outs frame-level
            seg_feats = encoder_outs[:, s:e + 1, :]  # (1, L, C)
            hyp = model.encoder.ctc_forward(seg_feats).squeeze(0)  # (L,)
            items = get_output_with_timestamps_1(hyp, char_dict)

            for it in items:
                # print(it["decode"])
                emp.append(it["decode"])
        print(' '.join(emp))

        if truncated_context_size * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
            break



# ---------------- Legacy decode (frame-CTC) ----------------
@torch.no_grad()
def endless_decode_legacy(args):
    """
    Decode theo pipeline cũ: encoder -> CTC head frame-level của encoder
    (để so sánh với CIF-CTC).
    """
    assert args.long_form_audio is not None, "Please provide --long_form_audio for legacy decode."

    device = torch.device(args.device)
    model, char_dict = init(args.model_checkpoint, device)

    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n - 1)

    subsampling_factor = model.encoder.embed.subsampling_factor
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # max length
    max_length_limited_context = int((args.total_batch_duration // 0.01)) // 2  # in 10ms
    multiply_n = max_length_limited_context // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n

    # relative right context
    rel_right_context_size = get_max_input_context(
        chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks
    )
    rel_right_context_size = rel_right_context_size * subsampling_factor

    waveform = load_audio(args.long_form_audio).to(device)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    xs = kaldi.fbank(
        waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000
    ).unsqueeze(0)

    att_cache = torch.zeros((model.encoder.num_blocks, left_context_size, model.encoder.attention_heads,
                             model.encoder._output_size * 2 // model.encoder.attention_heads)).to(device)
    cnn_cache = torch.zeros((model.encoder.num_blocks, model.encoder._output_size, conv_lorder)).to(device)

    for idx, _ in tqdm(list(enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)))):
        start = max(truncated_context_size * subsampling_factor * idx, 0)
        end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs.shape[1])

        x = xs[:, start:end + rel_right_context_size]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(device)

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
        if truncated_context_size * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
            encoder_outs = encoder_outs[:, :truncated_context_size]
        offset = offset - encoder_lens + encoder_outs.shape[1]

        hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
        decode = get_output_with_timestamps_1(hyp, char_dict)
        for item in decode:
            print(f"{item['decode']}")

        if truncated_context_size * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
            break


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Train/Infer CIF with streaming chunkformer.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--autocast_dtype", type=str, choices=["fp32", "bf16", "fp16"], default=None)

    # streaming params
    parser.add_argument("--total_batch_duration", type=int, default=1800)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--left_context_size", type=int, default=128)
    parser.add_argument("--right_context_size", type=int, default=128)

    # train
    parser.add_argument("--train_cif", action="store_true", help="Train CIF over Speech-MASSIVE_vie")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./cif_ckpts")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume_cif", type=str, default="")

    # infer (CIF)
    parser.add_argument("--infer_cif", action="store_true", help="Run streaming inference with CIF checkpoint")
    parser.add_argument("--cif_ckpt", type=str, default=None)
    parser.add_argument("--long_form_audio", type=str, default=None)

    # legacy decode (frame-CTC)
    parser.add_argument("--legacy_decode", action="store_true", help="Run legacy streaming decode (frame CTC)")

    args = parser.parse_args()

    # dtype context
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, None: None}[args.autocast_dtype]

    start_time = time.time()

    if args.train_cif:
        train_cif_streaming(args)

    elif args.infer_cif:
        stream_infer_with_runner(args)

    elif args.legacy_decode:
        endless_decode_legacy(args)

    else:
        raise SystemExit("Please choose one mode: --train_cif | --infer_cif | --legacy_decode")

    end_time = time.time()
    print(f"[done] elapsed: {round(end_time - start_time, 3)}s")


if __name__ == "__main__":
    import sys
    import torch, sys, os, glob

    print("exe:", sys.executable)
    print("CUDA avail:", torch.cuda.is_available(), "CUDA ver:", torch.version.cuda)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("nvidia devs:", glob.glob("/dev/nvidia*"))


    # sys.argv = [
    #     "decoder_train.py",
    #     "--model_checkpoint", "/home/manh/chunkformer/chunkformer/chunkformer-large-vie",
    #     "--device", "cuda",
    #     "--autocast_dtype", "fp32",
    #     "--train_cif",
    #     "--epochs", "5",
    #     "--save_dir", "/home/manh/chunkformer/cif_ckpts",
    #     "--save_every", "500",
    #     "--total_batch_duration", "2",
    #     "--chunk_size", "2",
    #     "--left_context_size", "128",
    #     "--right_context_size", "2",
    # ]

    sys.argv = [
        "decoder_train.py",
        "--model_checkpoint", "/home/trinhchau/code/EduAssist/api/services/chunkformer-large-vie",
        "--device", "cuda",
        "--autocast_dtype", "fp16",
        "--infer_cif",
        "--cif_ckpt", "/home/trinhchau/code/EduAssist/api/services/chunkformer/model/cif_best.pt",
        "--long_form_audio", "/home/trinhchau/code/EduAssist/data/AI_voice_2p.wav",
        "--total_batch_duration", "2",
        "--chunk_size", "2",
        "--left_context_size", "2",
        "--right_context_size", "2",
    ]

    main()
