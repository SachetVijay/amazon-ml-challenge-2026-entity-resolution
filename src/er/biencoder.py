"""Fine-tuned multilingual bi-encoder: dense blocking (exact GPU top-k) + a cosine feature.

Base: intfloat/multilingual-e5-small (MIT, 118M params). It reads Devanagari/Tamil/Kannada natively,
so transliterated names meet their Latin counterparts in one space.
Train: MultipleNegativesRanking loss on (record text, matching S1 text, hard negative S1) triplets. Hard
negatives = top TF-IDF candidates that are NOT true matches (same-name chains, neighbours).
Entities used here are disjoint from the classifier's sample (hash % 8 == 1 vs == 0): no label leak.

  python -m er.biencoder train  --art artifacts/train --gt .../train_ground_truth.tsv --out models/e5er
  python -m er.biencoder encode --art artifacts/test --model models/e5er        # -> emb_<country>.npy memmaps
  python -m er.biencoder search --art artifacts/test --k-other 10 --k-s1 10     # -> dense_<country>.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = "intfloat/multilingual-e5-small"
DIM = 384


def record_text(name: pd.Series, addr: pd.Series) -> list[str]:
    # raw fields: the model sees original scripts; e5 expects a "query: " prefix
    return ("query: " + name.astype(str).str.slice(0, 80) + " | " + addr.astype(str).str.slice(0, 160)).tolist()


def train(a):
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    from .predict import country_rows
    from .prep import load_gt
    path = Path(a.art) / "records.parquet"
    gt = load_gt(a.gt)
    s1u = gt.s1.unique()
    keep = set(s1u[(pd.util.hash_array(s1u.astype(object)) % a.sample_mod) == a.sample_rem])
    gt = gt[gt.s1.isin(keep)].sample(frac=1, random_state=0).head(a.max_pairs)
    # hard negatives: S1s the sparse blocker ranked highly for a record, but which are not its match
    ids = pd.read_parquet(path, columns=["entity_id"]).entity_id.to_numpy(dtype=object)
    rows_needed = np.flatnonzero(pd.Series(ids).isin(set(gt.other)).to_numpy())  # int rows, cheap to match
    hard = {}
    for country in country_rows(path):
        cp = Path(a.art) / f"cand_{country}_k5_5.parquet"
        if cp.exists():
            c = pd.read_parquet(cp, columns=["a", "b", "score_all_b2a"])
            c = c[np.isin(c.b.to_numpy(), rows_needed)]
            c = c.assign(s1=ids[c.a.to_numpy()], o=ids[c.b.to_numpy()]).sort_values("score_all_b2a", ascending=False)
            for o, g in c.groupby("o"):
                hard[o] = g.s1.tolist()
    truth = dict(zip(gt.other, gt.s1))
    need = set(gt.s1) | set(gt.other) | {s for v in hard.values() for s in v[:3]}
    rec = pd.read_parquet(path, columns=["entity_id", "business_name", "business_address"],
                          filters=[("entity_id", "in", list(need))])
    txt = dict(zip(rec.entity_id, record_text(rec.business_name, rec.business_address)))
    ex, n_hard = [], 0
    for s, o in zip(gt.s1, gt.other):
        if s not in txt or o not in txt:
            continue
        neg = next((h for h in hard.get(o, []) if h != truth[o] and h in txt), None)
        if neg is not None:
            ex.append(InputExample(texts=[txt[o], txt[s], txt[neg]]))  # query=record, pos=S1, hard neg S1
            n_hard += 1
        else:
            ex.append(InputExample(texts=[txt[o], txt[s]]))
    # keep one shape per batch: triplets first, then pairs are dropped if too few
    ex = [e for e in ex if len(e.texts) == 3] if n_hard > 0.5 * len(ex) else [InputExample(texts=e.texts[:2]) for e in ex]
    print(f"train examples {len(ex)} ({n_hard} had hard negatives)", flush=True)
    model = SentenceTransformer(BASE, device="cuda")
    model.max_seq_length = 96
    dl = DataLoader(ex, shuffle=True, batch_size=a.bs, drop_last=True)
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)
    model.fit(train_objectives=[(dl, loss)], epochs=a.epochs, warmup_steps=200, optimizer_params={"lr": a.lr},
              use_amp=True, show_progress_bar=True)
    model.save(a.out)
    print("saved", a.out)


def encode(a):
    """Per country, streamed in chunks into a disk memmap aligned with country_rows order."""
    from sentence_transformers import SentenceTransformer

    from .predict import country_rows
    path = Path(a.art) / "records.parquet"
    model = SentenceTransformer(a.model, device="cuda")
    model.max_seq_length = 96
    model.half()
    for country, rows in country_rows(path).items():
        out = Path(a.art) / f"emb_{country}.npy"
        if out.exists():
            continue
        sub = pd.read_parquet(path, columns=["business_name", "business_address"],
                              filters=[("country", "==", country)])
        assert len(sub) == len(rows)
        tmp = Path(a.art) / f"emb_{country}.tmp.npy"
        E = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=np.float16, shape=(len(sub), DIM))
        t = time.time()
        for i in range(0, len(sub), 500_000):
            txt = record_text(sub.business_name.iloc[i:i + 500_000], sub.business_address.iloc[i:i + 500_000])
            order = np.argsort([len(x) for x in txt])
            e = model.encode([txt[j] for j in order], batch_size=512, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
            E[i + order] = e
            print(f"  {country} {i + len(txt)}/{len(sub)} {time.time() - t:.0f}s", flush=True)
        E.flush()
        del E
        tmp.rename(out)
        print(f"encoded {country} {len(sub)} in {time.time() - t:.0f}s", flush=True)


def load_emb(art, country) -> np.ndarray:
    return np.load(Path(art) / f"emb_{country}.npy", mmap_mode="r")


@torch.no_grad()
def gpu_topk(E: np.ndarray, qidx: np.ndarray, didx: np.ndarray, k: int, qbs: int = 8192,
             dchunk: int = 100_000):
    """Exact inner-product top-k of rows E[qidx] over rows E[didx]. E may be a disk memmap: rows are
    gathered per batch, so RAM stays small. Score tiles are qbs x dchunk fp16 (8192 x 100k = 1.6 GB),
    corpus chunks are resident on the GPU, and a running top-k is merged on the CPU.
    Returned indices are positions into didx."""
    best_s = np.full((len(qidx), k), -np.inf, np.float32)
    best_i = np.zeros((len(qidx), k), np.int64)
    for d0 in range(0, len(didx), dchunk):
        Dt = torch.from_numpy(np.ascontiguousarray(E[didx[d0:d0 + dchunk]])).cuda()
        kk = min(k, len(Dt))
        for q0 in range(0, len(qidx), qbs):
            q = torch.from_numpy(np.ascontiguousarray(E[qidx[q0:q0 + qbs]])).cuda()
            s, i = (q @ Dt.T).topk(kk, dim=1)
            s = s.float()
            s, i = s.cpu().numpy(), i.cpu().numpy() + d0
            cs = np.concatenate([best_s[q0:q0 + qbs], s], 1)
            ci = np.concatenate([best_i[q0:q0 + qbs], i], 1)
            top = np.argsort(-cs, 1)[:, :k]
            best_s[q0:q0 + qbs] = np.take_along_axis(cs, top, 1)
            best_i[q0:q0 + qbs] = np.take_along_axis(ci, top, 1)
        del Dt
        torch.cuda.empty_cache()
    return best_s, best_i


def search(a):
    from .predict import country_rows
    path = Path(a.art) / "records.parquet"
    src_all = pd.read_parquet(path, columns=["src"]).src.to_numpy()
    for country, rows in country_rows(path).items():
        out = Path(a.art) / f"dense_{country}_k{a.k_other}_{a.k_s1}.parquet"
        if out.exists():
            continue
        t = time.time()
        E = load_emb(a.art, country)
        src = src_all[rows]
        ai, bi = np.flatnonzero(src == 1), np.flatnonzero(src != 1)
        s, i = gpu_topk(E, bi, ai, a.k_other)
        b2a = pd.DataFrame({"a": rows[ai[i.ravel()]], "b": np.repeat(rows[bi], a.k_other),
                            "dense_b2a": s.ravel(), "dense_rank_b2a": np.tile(np.arange(1, a.k_other + 1), len(bi))})
        del s, i
        s, i = gpu_topk(E, ai, bi, a.k_s1)
        a2b = pd.DataFrame({"a": np.repeat(rows[ai], a.k_s1), "b": rows[bi[i.ravel()]],
                            "dense_a2b": s.ravel(), "dense_rank_a2b": np.tile(np.arange(1, a.k_s1 + 1), len(ai))})
        c = b2a.merge(a2b, on=["a", "b"], how="outer")
        c = c.astype({col: np.float32 for col in c.columns if col.startswith("dense")})
        c.to_parquet(out)
        print(f"dense {country}: {len(c)} pairs ({len(c) / len(ai):.1f}/S1) {time.time() - t:.0f}s", flush=True)
        del b2a, a2b, c


def recall(a):
    """Recall of dense (and dense ∪ sparse) candidates on the classifier's sampled S1 (train only)."""
    from .predict import country_rows
    from .prep import load_gt
    path = Path(a.art) / "records.parquet"
    ids = pd.read_parquet(path, columns=["entity_id"]).entity_id.to_numpy(dtype=object)
    gt = load_gt(a.gt)
    gt = gt[(pd.util.hash_array(gt.s1.to_numpy(dtype=object)) % 8) == 0]
    g = set(zip(gt.s1, gt.other))
    for country in country_rows(path):
        d = pd.read_parquet(Path(a.art) / f"dense_{country}_k{a.k_other}_{a.k_s1}.parquet")
        sp = pd.read_parquet(Path(a.art) / f"cand_{country}_k5_5.parquet", columns=["a", "b"])
        s1c = set(ids[d.a.to_numpy()])
        gc_ = {p for p in g if p[0] in s1c}
        hit = lambda df: {p for p in zip(ids[df.a.to_numpy()], ids[df.b.to_numpy()]) if p in gc_}  # noqa: E731
        hs = hit(sp)
        for k in (1, 3, 5, 10):
            dk = d[(d.dense_rank_b2a <= k) | (d.dense_rank_a2b <= k)]
            hd = hit(dk)
            print(f"{country} dense k={k}: {len(hd) / len(gc_):.4f} | dense ∪ sparse(k5): "
                  f"{len(hd | hs) / len(gc_):.4f}  ({len(dk) / len(set(d.a)):.1f} dense pairs/S1)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "encode", "search", "recall"])
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--model", default="models/e5er")
    ap.add_argument("--out", default="models/e5er")
    ap.add_argument("--max-pairs", type=int, default=400_000)
    ap.add_argument("--sample-mod", type=int, default=8)
    ap.add_argument("--sample-rem", type=int, default=1)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--k-other", type=int, default=10)
    ap.add_argument("--k-s1", type=int, default=10)
    a = ap.parse_args()
    {"train": train, "encode": encode, "search": search, "recall": recall}[a.cmd](a)
