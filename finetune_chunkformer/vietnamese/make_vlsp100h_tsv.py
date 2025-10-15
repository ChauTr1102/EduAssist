import os, glob, random, pandas as pd
from pathlib import Path
random.seed(42)

FRACTION = 0.1  # dùng 10% dữ liệu

BASE = "/home/bojjoo/data_audio/vlsp2020_100h/"
WAV_DIR = os.path.join(BASE, "wavs")
TXT_DIR = os.path.join(BASE, "texts")

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR / "ctc" / "data"

OUTS = {
    "train": ROOT / "vlsp100h_train" / "data.tsv",
    "dev":   ROOT / "vlsp100h_dev"   / "data.tsv",
    "test":  ROOT / "vlsp100h_test"  / "data.tsv",
}

def stem(p): return os.path.splitext(os.path.basename(p))[0]

wav_paths = glob.glob(os.path.join(WAV_DIR, "**", "*.wav"), recursive=True)
wav_map = {stem(p): os.path.abspath(p) for p in wav_paths}

txt_paths = glob.glob(os.path.join(TXT_DIR, "**", "*.txt"), recursive=True)
txt_paths += glob.glob(os.path.join(TXT_DIR, "**", "*.lab"), recursive=True)
txt_map = {stem(p): os.path.abspath(p) for p in txt_paths}

pairs = []
dups = set()
for k, wavp in wav_map.items():
    tp = txt_map.get(k)
    if not tp:
        continue
    try:
        txt = open(tp, "r", encoding="utf-8").read()
    except UnicodeDecodeError:
        txt = open(tp, "r", encoding="latin-1").read()
    txt = txt.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

    key = k
    if key in dups:
        i = 1
        while f"{key}_{i}" in dups:
            i += 1
        key = f"{key}_{i}"
    dups.add(key)

    pairs.append((key, wavp, txt))

print(f"Total WAV: {len(wav_map)} | Paired: {len(pairs)}")

# >>> LẤY MẪU 10% TOÀN TẬP <<<
random.shuffle(pairs)
m = max(1, int(len(pairs) * FRACTION))
pairs = pairs[:m]
print(f"Using subset: {len(pairs)} (~{int(FRACTION*100)}%)")

# Chia 80/10/10 TRÊN SUBSET
n = len(pairs)
n_dev  = max(1, int(0.1*n))
n_test = max(1, int(0.1*n))
dev  = pairs[:n_dev]
test = pairs[n_dev:n_dev+n_test]
train = pairs[n_dev+n_test:]

splits = {"train": train, "dev": dev, "test": test}

for name, rows in splits.items():
    out = str(OUTS[name])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.DataFrame(rows, columns=["key", "wav", "txt"])
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {name}: {out} ({len(rows)} rows)")
