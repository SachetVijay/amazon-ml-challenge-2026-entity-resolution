"""Candidate generation. Output: one row per (s1 index, other index) with retrieval scores/ranks.

Retrievers, run per country (country is an open label set -> France just works):
  name_char : TF-IDF char 2-4-grams of the space-free core name (typos, domain names, spacing)
  combo_tok : TF-IDF over name-skeleton tokens + address words + numbers (transliteration, reordering)
  addr_tok  : TF-IDF over address tokens + numbers only (catches renamed businesses at the same address)
Both directions: every S2/S3 record retrieves its top-k S1s, and every S1 retrieves its top-k S2/S3s.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

RETRIEVERS = {
    # max_df prunes very frequent features: they carry little identity signal but make the sparse
    # product quadratic (posting lists of 10^5+). Cost ~ sum_q sum_{f in q} df(f).
    "name_char": dict(analyzer="char", ngram_range=(3, 4), min_df=2, max_df=0.0005, sublinear_tf=True, dtype=np.float32),
    "combo_tok": dict(analyzer=str.split, min_df=1, max_df=0.001, sublinear_tf=True, dtype=np.float32),
    "addr_tok": dict(analyzer=str.split, min_df=1, max_df=0.001, sublinear_tf=True, dtype=np.float32),
    # absolute max_df: posting-list cap independent of dataset size (keeps cost ~linear in N)
    # single retriever mixing all evidence: name char-3grams, name skeleton tokens, address words, numbers
    "all": dict(analyzer=str.split, min_df=2, max_df=12_000, sublinear_tf=True, dtype=np.float32),
}


def _char3(s: str) -> str:
    s = f"_{s.replace(' ', '')}_"
    return " ".join("c" + s[i:i + 3] for i in range(len(s) - 2))


def texts_for(rec: pd.DataFrame, kind: str) -> pd.Series:
    rec = rec[["name_core", "name_skel", "addr_words", "addr_skel", "addr_nums"]].astype(object)  # arrow-safe
    if kind == "name_char":
        return rec.name_core.str.replace(" ", "", regex=False).radd(" ").add(" ")
    if kind == "combo_tok":
        return ("n_" + rec.name_skel.str.replace(" ", " n_", regex=False)) + " " + rec.addr_words + " " + rec.addr_nums
    if kind == "all":
        return (rec.name_core.map(_char3) + " " + ("n_" + rec.name_skel.str.replace(" ", " n_", regex=False))
                + " " + ("w_" + rec.addr_skel.str.replace(" ", " w_", regex=False))
                + " " + ("d_" + rec.addr_nums.str.replace(" ", " d_", regex=False)))
    if kind == "addr_tok":
        return rec.addr_skel + " " + rec.addr_nums
    raise KeyError(kind)


def _topk(Q, D, k, thr, threads):
    """For each row of Q, top-k rows of D by cosine. Returns (q_idx, d_idx, score)."""
    M = sp_matmul_topn(Q, D.T.tocsr(), top_n=k, threshold=thr, sort=True, n_threads=threads).tocoo()
    return M.row.astype(np.int64), M.col.astype(np.int64), M.data.astype(np.float32)


def generate(rec: pd.DataFrame, k_other: int = 10, k_s1: int = 20, threads: int = 16,
             retrievers=("name_char", "combo_tok", "addr_tok")) -> pd.DataFrame:
    """rec: normalised records of one split (src 1/2/3). Returns candidate pairs of global row indices."""
    out = []
    for country, g in rec.groupby("country", sort=False):
        src = np.asarray(g.src.to_numpy(), dtype=np.int64)
        a_idx = g.index[src == 1].to_numpy()
        b_idx = g.index[src != 1].to_numpy()
        if len(a_idx) == 0 or len(b_idx) == 0:
            continue
        for kind in retrievers:
            vec = TfidfVectorizer(**RETRIEVERS[kind])
            t0 = time.time()
            X = vec.fit_transform(texts_for(g, kind))  # fit on this country's records only (unsupervised)
            pos = {i: j for j, i in enumerate(g.index)}
            XA = X[[pos[i] for i in a_idx]]
            XB = X[[pos[i] for i in b_idx]]
            # other -> s1
            q, d, s = _topk(XB, XA, k_other, 0.05, threads)
            out.append(pd.DataFrame({"a": a_idx[d], "b": b_idx[q], "kind": kind, "dir": "b2a", "score": s}))
            # s1 -> other
            q, d, s = _topk(XA, XB, k_s1, 0.05, threads)
            out.append(pd.DataFrame({"a": a_idx[q], "b": b_idx[d], "kind": kind, "dir": "a2b", "score": s}))
            print(f"    {country} {kind}: vocab {X.shape[1]} nnz/row {X.nnz / X.shape[0]:.1f} {time.time() - t0:.0f}s", flush=True)
        print(f"  blocked {country}: {len(a_idx)} s1 x {len(b_idx)} other", flush=True)
    cand = pd.concat(out, ignore_index=True)
    return cand


def pivot_scores(cand: pd.DataFrame) -> pd.DataFrame:
    """One row per unique (a, b) with per-retriever max score and within-query ranks (features)."""
    cand = cand.copy()
    cand["rank"] = cand.groupby(["kind", "dir", np.where(cand["dir"] == "b2a", cand.b, cand.a)])["score"] \
        .rank(ascending=False, method="first")
    wide = cand.pivot_table(index=["a", "b"], columns=["kind", "dir"], values=["score", "rank"], aggfunc="max")
    wide.columns = [f"{v}_{k}_{d}" for v, k, d in wide.columns]
    return wide.reset_index()
