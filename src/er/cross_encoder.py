"""Cross-encoder re-ranker for uncertain pairs (reads both records jointly, token by token).

Base: intfloat/multilingual-e5-small (MIT, 118M) + 1-logit head. Trained on S1 hash-bucket 2 — disjoint from
the classifier (bucket 0) and the bi-encoder (bucket 1) — on bi-encoder top-3 candidates (true matches +
the hardest look-alikes: planted distractors, adjacent house numbers, chain namesakes).

  python -m er.cross_encoder train --art artifacts/train --gt .../train_ground_truth.tsv --out models/ce
  python -m er.cross_encoder score --art artifacts/train --pairs artifacts/train/oof_hyb5.parquet \
      --model models/ce --out artifacts/train/ce_oof.parquet      # pairs: a, b columns (global rows)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE = "intfloat/multilingual-e5-small"


def rec_text(df: pd.DataFrame) -> list[str]:
    return (df.business_name.astype(str).str.slice(0, 80) + " | " + df.business_address.astype(str).str.slice(0, 140)).tolist()


def norm_text(df: pd.DataFrame) -> list[str]:
    """Normalised view: core name (legal + EN/FR filler words removed) | canonical address (regions, abbrevs)."""
    return (df.name_core.astype(str) + " | " + df.addr_clean.astype(str).str.slice(0, 160)).tolist()


def load_texts(art: str, rows: np.ndarray, view: str = "raw") -> dict:
    """Global row -> text, reading only the needed rows (row order of records.parquet)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(Path(art) / "records.parquet")
    need = np.unique(rows)
    out, off = {}, 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        lo, hi = np.searchsorted(need, [off, off + n])
        if hi > lo:
            cols = ["business_name", "business_address"] if view == "raw" else ["name_core", "addr_clean"]
            t = pf.read_row_group(rg, columns=cols).to_pandas()
            loc = need[lo:hi] - off
            fn = rec_text if view == "raw" else norm_text
            for r, s in zip(need[lo:hi], fn(t.iloc[loc])):
                out[int(r)] = s
        off += n
    return out


def build_train_pairs(a) -> pd.DataFrame:
    from .predict import country_rows
    from .prep import load_gt
    path = Path(a.art) / "records.parquet"
    ids = pd.read_parquet(path, columns=["entity_id"]).entity_id.to_numpy(dtype=object)
    gt = load_gt(a.gt)
    pos = set(zip(gt.s1, gt.other))
    parts = []
    for country, rows in country_rows(path).items():
        dp = Path(a.art) / f"dense_{country}_k10_10.parquet"
        if not dp.exists():
            continue
        d = pd.read_parquet(dp, columns=["a", "b", "dense_rank_b2a", "dense_rank_a2b"],
                            filters=[[("dense_rank_b2a", "<=", 3)], [("dense_rank_a2b", "<=", 3)]])
        s1 = ids[d.a.to_numpy()]
        d = d[np.isin(pd.util.hash_array(s1) % 8, a.buckets)]
        parts.append(d[["a", "b"]])
    P = pd.concat(parts, ignore_index=True).drop_duplicates()
    P["y"] = np.fromiter(((ids[x], ids[y]) in pos for x, y in zip(P.a.to_numpy(), P.b.to_numpy())), bool, len(P))
    # keep all positives' share natural but cap total size
    P = P.sample(min(a.max_pairs, len(P)), random_state=0)
    print(f"train pairs {len(P)} pos rate {P.y.mean():.3f}", flush=True)
    return P


class PairDS(torch.utils.data.Dataset):
    def __init__(self, P, txt):
        self.a, self.b = P.a.to_numpy(), P.b.to_numpy()
        self.y = P.y.to_numpy().astype(np.float32) if "y" in P else np.zeros(len(P), np.float32)
        self.txt = txt

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.txt[int(self.a[i])], self.txt[int(self.b[i])], self.y[i]


def collate(tok, max_len):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return enc, torch.tensor(y)
    return f


def train(a):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    P = build_train_pairs(a)
    txt = load_texts(a.art, np.concatenate([P.a.to_numpy(), P.b.to_numpy()]))
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1).cuda()
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    dl = DataLoader(PairDS(P, txt), batch_size=a.bs, shuffle=True, collate_fn=collate(tok, a.max_len),
                    num_workers=4, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(dl) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    model.train()
    t = time.time()
    for ep in range(a.epochs):
        for it, (enc, y) in enumerate(dl):
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y.cuda())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            if it % 500 == 0:
                print(f"ep {ep} it {it}/{len(dl)} loss {loss.item():.4f} {time.time() - t:.0f}s", flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out, flush=True)


@torch.no_grad()
def score(a):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    P = pd.read_parquet(a.pairs, columns=["a", "b"] + (["p"] if a.band else []))
    if a.band:
        P = P[(P.p > a.band[0]) & (P.p < a.band[1])]
    P = P[["a", "b"]].drop_duplicates().reset_index(drop=True)
    print(f"scoring {len(P)} pairs", flush=True)
    txt = load_texts(a.art, np.concatenate([P.a.to_numpy(), P.b.to_numpy()]), getattr(a, "view", "raw"))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).cuda().half().eval()
    order = np.argsort([len(txt[int(x)]) + len(txt[int(y)]) for x, y in zip(P.a, P.b)])
    Ps = P.iloc[order]
    dl = DataLoader(PairDS(Ps, txt), batch_size=512, shuffle=False, collate_fn=collate(tok, a.max_len), num_workers=4)
    out, t = [], time.time()
    for enc, _ in dl:
        out.append(model(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy())
    s = np.empty(len(P), np.float32)
    s[order] = np.concatenate(out)
    P["ce"] = s
    P.to_parquet(a.out)
    print(f"scored {len(P)} in {time.time() - t:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--pairs")
    ap.add_argument("--model", default="models/ce")
    ap.add_argument("--out", default="models/ce")
    ap.add_argument("--buckets", type=int, nargs="+", default=[2])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=800_000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--band", type=float, nargs=2, default=None, help="only score pairs with lo < p < hi")
    ap.add_argument("--view", default="raw", choices=["raw", "norm"], help="text view fed to the model")
    a = ap.parse_args()
    {"train": train, "score": score}[a.cmd](a)
