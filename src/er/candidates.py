"""Hybrid candidate set: sparse TF-IDF top-k  ∪  fine-tuned bi-encoder top-k, with context features.

Dense-side context (a record's cosine gap to its best S1, an S1's gap to its best record) is computed on
the FULL dense table before any sampling, so it reflects all competitors, exactly as at test time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cosine(E: np.ndarray, rows: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Cosine of bi-encoder embeddings for global row pairs (E is the country memmap, aligned with rows)."""
    pa, pb = np.searchsorted(rows, a), np.searchsorted(rows, b)
    out = np.empty(len(a), np.float32)
    for i in range(0, len(a), chunk):
        ea = np.asarray(E[pa[i:i + chunk]], np.float32)
        eb = np.asarray(E[pb[i:i + chunk]], np.float32)
        out[i:i + chunk] = (ea * eb).sum(1)
    return out


def record_best(art: str, country: str) -> pd.Series:
    """Each record's best cosine over all S1s (its dense rank-1 row). Small; cache per country."""
    dp = Path(art) / f"dense_{country}_k10_10.parquet"
    return pd.read_parquet(dp, columns=["b", "dense_b2a"], filters=[("dense_rank_b2a", "==", 1)]) \
        .set_index("b")["dense_b2a"]


def hybrid(sparse: pd.DataFrame, art: str, country: str, rows: np.ndarray, dense_k: int = 5,
           keep_a=None, b_best: pd.Series | None = None) -> pd.DataFrame:
    """sparse: sparse candidates (+ sparse context), already restricted to keep_a if given.
    rows: global rows of this country (sorted). keep_a: S1 rows to keep (training sample) or None."""
    from .biencoder import load_emb
    dp = Path(art) / f"dense_{country}_k10_10.parquet"
    rank_or = [[("dense_rank_b2a", "<=", dense_k)], [("dense_rank_a2b", "<=", dense_k)]]
    if keep_a is not None:
        ka = list(map(int, keep_a))
        d = pd.read_parquet(dp, filters=[[("a", "in", ka)] + r for r in rank_or])
    else:
        d = pd.read_parquet(dp, filters=rank_or)
    # per-side best cosine over ALL S1s / records: rank-1 rows (record side read separately so S1
    # sampling cannot bias it); S1 side is complete because a sampled S1 keeps all its rows
    if b_best is None:
        b_best = record_best(art, country)
    a_best = d.groupby("a")["dense_a2b"].max()
    d["dense_b_gap"] = (d["dense_b2a"] - d.b.map(b_best)).astype(np.float32)
    d["dense_a_gap"] = (d["dense_a2b"] - d.a.map(a_best)).astype(np.float32)
    c = sparse.merge(d, on=["a", "b"], how="outer")
    del d
    c["in_sparse"] = c["score_all_b2a"].notna() | c["score_all_a2b"].notna()
    c["in_dense"] = c["dense_b2a"].notna() | c["dense_a2b"].notna()
    c[["in_sparse", "in_dense"]] = c[["in_sparse", "in_dense"]].astype(np.float32)
    c["dense_cos"] = cosine(load_emb(art, country), rows, c.a.to_numpy(), c.b.to_numpy())
    c["cos_b_gap"] = (c["dense_cos"] - c.b.map(b_best).fillna(c["dense_cos"])).astype(np.float32)
    c["cos_a_gap"] = (c["dense_cos"] - c.a.map(a_best).fillna(c["dense_cos"])).astype(np.float32)
    c["u_a_ncand"] = c.groupby("a")["b"].transform("size").astype(np.float32)
    c["cos_a_rank"] = c.groupby("a")["dense_cos"].rank(ascending=False, method="first").astype(np.float32)
    return c


HYBRID_CTX = ["dense_b2a", "dense_a2b", "dense_rank_b2a", "dense_rank_a2b", "dense_b_gap", "dense_a_gap",
              "in_sparse", "in_dense", "dense_cos", "u_a_ncand", "cos_b_gap", "cos_a_gap", "cos_a_rank"]
