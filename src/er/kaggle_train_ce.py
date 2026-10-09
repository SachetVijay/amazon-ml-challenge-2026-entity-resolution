"""Cross-encoder training that runs on Kaggle (T4, fp16) or any CUDA box. Self-contained: reads a pairs parquet
(text_a, text_b, y) and writes a HF model dir. Same recipe as er.cross_encoder.train.

Kaggle:  data at /kaggle/input/<dataset>/pairs.parquet, output in /kaggle/working/model
Local :  python train_ce.py --pairs kaggle_ce/pairs.parquet --out models/ce_mdeberta
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class DS(Dataset):
    def __init__(self, df):
        self.a, self.b = df.text_a.tolist(), df.text_b.tolist()
        self.y = df.y.to_numpy().astype(np.float32)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--base", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--out", default="/kaggle/working/model")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-hours", type=float, default=10.5, help="stop + save before Kaggle's 12 h limit")
    a = ap.parse_args()
    path = a.pairs or glob.glob("/kaggle/input/**/pairs.parquet", recursive=True)[0]
    df = pd.read_parquet(path).sample(frac=1, random_state=0)
    print(f"pairs {len(df)} pos {df.y.mean():.3f} from {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1, torch_dtype=torch.float32).float().cuda()
    core = model
    if torch.cuda.device_count() > 1:          # Kaggle T4 x2: split each batch across both GPUs
        model = torch.nn.DataParallel(model)
        print(f"DataParallel over {torch.cuda.device_count()} GPUs", flush=True)

    def collate(batch):
        x, y2, ys = zip(*batch)
        enc = tok(list(x), list(y2), truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        return enc, torch.tensor(ys)

    dl = DataLoader(DS(df), batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(dl) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    model.train()
    t, bad = time.time(), 0
    for ep in range(a.epochs):
        for it, (enc, y) in enumerate(dl):
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y.cuda())
            if not torch.isfinite(loss):
                bad += 1
                opt.zero_grad()
                if bad > 50:
                    raise RuntimeError("too many non-finite losses (fp16 instability)")
                continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            if it % 500 == 0:
                print(f"ep {ep} it {it}/{len(dl)} loss {loss.item():.4f} {time.time() - t:.0f}s", flush=True)
            if time.time() - t > a.max_hours * 3600:
                print("time budget reached, saving early", flush=True)
                break
    os.makedirs(a.out, exist_ok=True)
    core.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
