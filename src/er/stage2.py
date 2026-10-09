"""Stage 2: cluster-aware re-scoring on top of stage-1 pair probabilities (stacking).

Stage 1 judges each (S1, record) pair in isolation. Stage 2 adds *collective* evidence:
  - competition:   rank of p within the S1 and within the record, margin over the record's runner-up S1
  - entity state:  how many confident matches the S1 already has, their total mass
  - sibling sim:   similarity of this record to the S1's most confident *other* record. A renamed record
                   ("Brixarc", same address) or a transliterated one can be vouched for by its siblings.
Train on out-of-fold stage-1 p (no leakage); at test time use the full stage-1 model's p.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist


def consensus_features(d: pd.DataFrame, E: np.ndarray, rows: np.ndarray, col: str = "p",
                       conf: float = 0.9, chunk: int = 500_000) -> pd.DataFrame:
    """Cosine of each candidate record to the centroid of the S1's *other* confident records.

    An entity's ~3.5 noisy records average into a denoised view of the business; a genuine member (even
    renamed / partially corrupted) sits close to it, a planted look-alike of the S1 record does not.
    E: bi-encoder embeddings of this country (memmap, aligned with `rows`, L2-normalised).
    """
    pos_b = np.searchsorted(rows, d.b.to_numpy())
    mem = (d[col].to_numpy() >= conf)
    a_codes, a_uni = pd.factorize(d.a.to_numpy())
    dim = E.shape[1]
    S = np.zeros((len(a_uni), dim), np.float32)
    n = np.bincount(a_codes[mem], minlength=len(a_uni)).astype(np.float32)
    mi = np.flatnonzero(mem)
    for i in range(0, len(mi), chunk):
        idx = mi[i:i + chunk]
        np.add.at(S, a_codes[idx], np.asarray(E[pos_b[idx]], np.float32))
    cos = np.full(len(d), np.nan, np.float32)
    for i in range(0, len(d), chunk):
        sl = slice(i, i + chunk)
        eb = np.asarray(E[pos_b[sl]], np.float32)
        ac = a_codes[sl]
        c = S[ac] - eb * mem[sl, None]                  # leave this record out of its own centroid
        k = n[ac] - mem[sl]
        norm = np.linalg.norm(c, axis=1)
        ok = (k > 0) & (norm > 0)
        cos[sl] = np.where(ok, (c * eb).sum(1) / np.maximum(norm, 1e-6), np.nan)
    d["cons_cos"] = cos
    d["cons_n"] = (n[a_codes] - mem).astype(np.float32)
    return d


def context_features(P: pd.DataFrame, rec: pd.DataFrame, col: str = "p") -> pd.DataFrame:
    d = P[["a", "b", col]].copy()
    ga, gb = d.groupby("a")[col], d.groupby("b")[col]
    d["rank_in_a"] = ga.rank(ascending=False, method="first").astype(np.float32)
    d["rank_in_b"] = gb.rank(ascending=False, method="first").astype(np.float32)
    d["a_max"] = ga.transform("max").astype(np.float32)
    d["a_sum"] = ga.transform("sum").astype(np.float32)
    d["a_n50"] = d.assign(_c=d[col] > 0.5).groupby("a")["_c"].transform("sum").astype(np.float32)
    d["b_max"] = gb.transform("max").astype(np.float32)
    # margin against the best *other* S1 for this record
    second = d.sort_values(col, ascending=False).groupby("b")[col].nth(1)
    d["b_second"] = d.b.map(pd.Series(second.values, index=d.loc[second.index, "b"].values)).fillna(0).astype(np.float32)
    d["b_margin"] = np.where(d["rank_in_b"] == 1, d[col] - d["b_second"], d[col] - d["b_max"]).astype(np.float32)

    # sibling = the S1's most confident record other than this one
    top = d.sort_values(col, ascending=False).groupby("a").head(2)
    top["r"] = top.groupby("a").cumcount()
    t1 = top[top.r == 0].set_index("a")["b"]
    t2 = top[top.r == 1].set_index("a")["b"]
    sib = np.where(d.b.to_numpy() == d.a.map(t1).to_numpy(), d.a.map(t2).to_numpy(), d.a.map(t1).to_numpy())
    has = ~pd.isna(sib)
    sib_idx = np.where(has, sib, d.b.to_numpy()).astype(np.int64)  # self as placeholder; masked below
    B = rec.loc[d.b.to_numpy(), ["name_core", "addr_clean"]].astype(object)
    S = rec.loc[sib_idx, ["name_core", "addr_clean"]].astype(object)
    for c, nm in (("name_core", "sib_name"), ("addr_clean", "sib_addr")):
        v = cpdist(B[c].tolist(), S[c].tolist(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        d[nm] = np.where(has, v, -1).astype(np.float32)
    # probability of the sibling itself (how trustworthy the voucher is)
    pmap = d.set_index(["a", "b"])[col]
    d["sib_p"] = np.where(has, pmap.reindex(pd.MultiIndex.from_arrays([d.a.to_numpy(), sib_idx])).to_numpy(), -1)
    return d.rename(columns={col: "p1"})
