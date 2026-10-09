"""Turn pair probabilities into per-S1 match lists, and score them with the official macro F0.5.

Decision rules (composable):
  exclusive   each S2/S3 record belongs to at most one S1 (true in train GT): keep only its best S1
  threshold   keep pairs with p >= t
  expected_f  per S1, choose the top-m candidates (by p) maximising *expected* F0.5 under independent
              Bernoulli(p) labels, with m = 0 allowed (singleton). This is the Bayes-optimal plug-in
              rule for a per-entity F-measure (cf. Jansche 2007; Nan Ye et al. 2012).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25


def f05(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    return float(np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in s1_ids]))


def exclusive(df: pd.DataFrame, col: str = "p") -> pd.DataFrame:
    """Keep, for every b, only its highest-probability a."""
    i = df.groupby("b")[col].idxmax()
    return df.loc[i]


def _expected_f_topm(p: np.ndarray, n_mc: int = 256, rng=None) -> int:
    """Best m (0..len(p)) for top-m selection maximising E[F0.5]; p sorted descending.
    Monte-Carlo over the label vector (exact enough, vectorised)."""
    if len(p) == 0:
        return 0
    rng = rng or np.random.default_rng(0)
    Y = rng.random((n_mc, len(p))) < p                           # sampled truth among candidates
    n_true = Y.sum(1)                                            # (ignores true matches outside candidates)
    tp_cum = np.cumsum(Y, 1)                                     # tp if we predict top-m
    best_m, best = 0, np.mean(n_true == 0)                       # m = 0: F = 1 iff no true matches
    for m in range(1, len(p) + 1):
        tp = tp_cum[:, m - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            prec, rec = tp / m, np.where(n_true > 0, tp / n_true, 0)
            f = np.where(tp > 0, (1 + BETA2) * prec * rec / (BETA2 * prec + rec), 0.0)
        ef = f.mean()
        if ef > best:
            best, best_m = ef, m
    return best_m


def expected_f_select(df: pd.DataFrame, col: str = "p", max_cand: int = 12) -> pd.DataFrame:
    df = df.sort_values(["a", col], ascending=[True, False])
    keep = []
    for a, g in df.groupby("a", sort=False):
        p = g[col].to_numpy()[:max_cand]
        m = _expected_f_topm(p)
        keep.append(g.index[:m])
    return df.loc[np.concatenate(keep) if keep else []]


def expected_f_fast(df: pd.DataFrame, col: str = "p", empty_weight: float = 1.0) -> pd.DataFrame:
    """Vectorised plug-in version of expected_f_select.

    F_beta = (1+b2)*TP / (b2*|true| + |pred|). For top-m selection plug in E[TP_m] = sum_{i<=m} p_i and
    E[|true|] = sum_i p_i. The empty prediction scores 1 iff the entity has no match: E = prod(1 - p_i).
    empty_weight > 1 biases toward predicting singletons (tunable on validation).
    """
    d = df[["a", "b", col]].sort_values(["a", col], ascending=[True, False]).reset_index(drop=True)
    g = d.groupby("a", sort=False)
    d["m"] = g.cumcount() + 1
    cum = g[col].cumsum()
    tot = g[col].transform("sum")
    d["ef"] = (1 + BETA2) * cum / (BETA2 * tot + d["m"])
    d["logq"] = np.log1p(-d[col].clip(0, 1 - 1e-9))
    per = d.groupby("a", sort=False).agg(best_ef=("ef", "max"), logq=("logq", "sum"))
    # smallest m achieving the max expected F (ef is evaluated for every prefix length)
    best_m = d[d["ef"] == d["a"].map(per["best_ef"])].groupby("a", sort=False)["m"].min()
    per["m"] = np.where(per["best_ef"] > empty_weight * np.exp(per["logq"]), best_m.reindex(per.index), 0)
    return d.loc[d["m"] <= d["a"].map(per["m"]), ["a", "b", col]]


def to_lists(sel: pd.DataFrame, ids: np.ndarray) -> dict:
    out = {}
    for a, b in zip(sel.a.to_numpy(), sel.b.to_numpy()):
        out.setdefault(ids[a], set()).add(ids[b])
    return out
