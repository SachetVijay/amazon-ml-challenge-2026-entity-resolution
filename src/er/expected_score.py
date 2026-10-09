"""Label-free estimate of macro F0.5 from calibrated pair probabilities.

For each S1: decision = exclusive winners + expected-F0.5 selection. Its expected F0.5 under independent
Bernoulli(p) labels is estimated by Monte Carlo over the S1's candidate labels. Validated on OOF (where
the true F0.5 is known), then applied per test country — the only way to measure France without labels.

  python -m er.expected_score --oof artifacts/train/oof_stage2_hyb5_ce_v4.parquet --col p2 --gt ...
  python -m er.expected_score --test artifacts/test/v4_backup --col p
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .decide import BETA2, exclusive, expected_f_fast


def expected_f(P: pd.DataFrame, col: str, n_s1: int, n_mc: int = 64, seed: int = 0) -> float:
    """Mean expected F0.5 over n_s1 entities (entities without candidates count as correct empties)."""
    rng = np.random.default_rng(seed)
    ex = exclusive(P[["a", "b", col]].rename(columns={col: "p"}))
    sel = expected_f_fast(ex)
    ex = ex.assign(chosen=ex.set_index(["a", "b"]).index.isin(sel.set_index(["a", "b"]).index))
    ex = ex.sort_values("a")
    a = ex.a.to_numpy()
    starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    p, ch = ex.p.to_numpy(), ex.chosen.to_numpy()
    tot = 0.0
    for s, e in zip(starts, np.r_[starts[1:], len(a)]):
        Y = rng.random((n_mc, e - s)) < p[s:e]              # sampled truth among this S1's winners
        n_true, m = Y.sum(1), ch[s:e].sum()
        tp = Y[:, ch[s:e]].sum(1)
        if m == 0:
            f = (n_true == 0).astype(float)
        else:
            prec, rec = tp / m, np.divide(tp, n_true, out=np.zeros(n_mc), where=n_true > 0)
            f = np.where(tp > 0, (1 + BETA2) * prec * rec / (BETA2 * prec + rec + 1e-12), 0.0)
        tot += f.mean()
    n_with = len(starts)
    return (tot + (n_s1 - n_with)) / n_s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof")
    ap.add_argument("--gt")
    ap.add_argument("--test", help="dir with pred_s2_<country>.parquet")
    ap.add_argument("--art", default="artifacts/test")
    ap.add_argument("--col", default="p")
    ap.add_argument("--sample", type=int, default=60000, help="S1 entities to sample per group (speed)")
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    if a.oof:
        from .decide import f05
        from .prep import load_gt
        D = pd.read_parquet(a.oof, columns=["a", "b", "s1_id", "o_id", "country", a.col])
        gt = load_gt(a.gt)
        for c in D.country.unique():
            Dc = D[D.country == c]
            s1s = Dc.s1_id.unique()
            keep = set(rng.choice(s1s, min(a.sample, len(s1s)), replace=False))
            Dc = Dc[Dc.s1_id.isin(keep)]
            truth = gt[gt.s1.isin(keep)].groupby("s1").other.agg(set).to_dict()
            ex = exclusive(Dc.rename(columns={a.col: "p"}))
            sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
            pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
            actual = np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in keep])
            est = expected_f(Dc, a.col, len(keep))
            print(f"OOF {c:6s}: actual F0.5 {actual:.4f} | label-free estimate {est:.4f}", flush=True)
    if a.test:
        from .predict import country_rows
        src = pd.read_parquet(Path(a.art) / "records.parquet", columns=["src"]).src.to_numpy()
        for c, rows in country_rows(Path(a.art) / "records.parquet").items():
            P = pd.read_parquet(Path(a.test) / f"pred_s2_{c}.parquet")
            s1_rows = rows[src[rows] == 1]
            keep = rng.choice(s1_rows, min(a.sample, len(s1_rows)), replace=False)
            Pc = P[P.a.isin(set(keep))]
            print(f"TEST {c:6s}: label-free estimate {expected_f(Pc, a.col, len(keep)):.4f}", flush=True)


if __name__ == "__main__":
    main()
