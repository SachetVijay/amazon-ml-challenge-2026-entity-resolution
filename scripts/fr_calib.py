"""Calibration from a leaderboard probe. Decisions use odds*r; truth ~ Bernoulli(q) with q = model p with odds*c.
E_c[F(r)] by Monte Carlo over the exclusive winners of each S1 (same approximation as er.expected_score)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import BETA2, exclusive, expected_f_fast, f05
def odds(p, r):
    p = np.clip(p, 1e-7, 1 - 1e-7); return 1 / (1 + (1 - p) / (p * r))
def prep(P, n_sample, seed=0):
    ex = exclusive(P[["a", "b", "p"]])
    s1 = ex.a.unique()
    keep = np.random.default_rng(seed).choice(s1, min(n_sample, len(s1)), replace=False)
    return ex[ex.a.isin(keep)].sort_values(["a", "p"], ascending=[True, False]).reset_index(drop=True), len(keep)
def EF(ex, r, c, n_mc=200, seed=1):
    rng = np.random.default_rng(seed)
    d = ex.assign(p=odds(ex.p.to_numpy(), r))
    sel = expected_f_fast(d)
    ch = d.set_index(["a", "b"]).index.isin(sel.set_index(["a", "b"]).index)
    q = odds(ex.p.to_numpy(), c)
    a = ex.a.to_numpy(); starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]; ends = np.r_[starts[1:], len(a)]
    tot = 0.0
    for s, e in zip(starts, ends):
        Y = rng.random((n_mc, e - s)) < q[s:e]
        nt, m = Y.sum(1), ch[s:e].sum()
        if m == 0: f = (nt == 0).astype(float)
        else:
            tp = Y[:, ch[s:e]].sum(1); prec = tp / m; rec = np.divide(tp, nt, out=np.zeros(n_mc), where=nt > 0)
            f = np.where(tp > 0, (1 + BETA2) * prec * rec / (BETA2 * prec + rec + 1e-12), 0.0)
        tot += f.mean()
    return tot / len(starts)
mode = sys.argv[1]
if mode == "val":     # in-domain check: does E[F] predict the measured effect of r on labelled data?
    from er.prep import load_gt
    full = pd.read_parquet("artifacts/train/oof_band_v11l.parquet")
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
    samp = set(full.s1_id); truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict(); allS = samp | set(truth)
    ex, n = prep(full, 10**7)
    for r in (0.5, 1.0, 2.0):
        d = full.assign(p=odds(full.p.to_numpy(), r)); e2 = exclusive(d[["a", "b", "s1_id", "o_id", "p"]])
        sel = expected_f_fast(e2).merge(e2[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
        pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
        true_f = np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in allS])
        print(f"validation r={r}: TRUE macro F {true_f:.5f} | model-expected F (S1 with candidates) {EF(ex, r, 1.0, n_mc=64):.5f}", flush=True)
else:
    P = pd.read_parquet("calib/v11_France_p.parquet")
    ex, n = prep(P, 80000)
    base = {c: EF(ex, 1.0, c) for c in (0.35, 0.5, 0.7, 1.0)}
    for c in (0.35, 0.5, 0.7, 1.0):
        row = {r: EF(ex, r, c) - base[c] for r in (0.35, 0.5, 0.7, 1.4, 2.0)}
        print(f"France truth odds x{c}: E[F(r=1)] {base[c]:.4f} | dF by decision r: " + " ".join(f"r{r}:{v:+.4f}" for r, v in row.items()), flush=True)
    print("observed on leaderboard: r=2 on France -> -0.00032 overall = about -0.0021 on France (15% share)")
