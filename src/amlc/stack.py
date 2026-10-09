"""Blend OOF predictions by directly optimising the competition metric, then write the submission.

  python -m amlc.stack                       # all models in artifacts/oof
  python -m amlc.stack --models lgb_v1 mlp_v1 qwen8_lora --out submissions/sub_v3.csv

Blend = inv(b + sum_i w_i * fwd(pred_i)), weights fitted on rows where every model has
an OOF prediction. The reported score uses nested CV (weights fit on k-1 folds, scored on
the held-out fold), which is the number to trust over the public leaderboard.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import ART, ROOT, cfg, folds, fwd, inv, test_df, y_raw
from .metrics import greater_is_better, score


def fit_weights(P, y, gib):
    n = P.shape[1]
    sign = -1 if gib else 1

    def obj(w):
        return sign * score(y, inv(w[-1] + P @ w[:-1]))

    x0 = np.r_[np.full(n, 1 / n), 0.0]
    return minimize(obj, x0, method="Powell", options={"maxiter": 20000, "xtol": 1e-4, "ftol": 1e-6}).x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default="submissions/submission.csv")
    a = ap.parse_args()
    c, y, f = cfg(), y_raw(), folds()
    names = a.models or sorted(p.stem for p in (ART / "oof").glob("*.npy"))
    O = np.column_stack([np.load(ART / "oof" / f"{n}.npy") for n in names])
    T = np.column_stack([np.load(ART / "test" / f"{n}.npy") for n in names])
    if c.get("positive_only"):
        O, T = np.clip(O, 1e-6, None), np.clip(T, 1e-6, None)
    mask = np.isfinite(O).all(1)
    gib = greater_is_better()
    print(f"blend rows: {mask.sum()} / {len(y)}")
    for i, n in enumerate(names):
        print(f"  {n:30s} {score(y[mask], O[mask, i]):.4f}")

    P, TP = fwd(O[mask]), fwd(T)
    ym, fm = y[mask], f[mask]
    nested = np.zeros(len(ym))
    for k in np.unique(fm):
        w = fit_weights(P[fm != k], ym[fm != k], gib)
        nested[fm == k] = inv(w[-1] + P[fm == k] @ w[:-1])
    w = fit_weights(P, ym, gib)
    print("weights:", dict(zip(names + ["bias"], np.round(w, 4))))
    print(f"blend nested-CV: {score(ym, nested):.4f}   (in-sample {score(ym, inv(w[-1] + P @ w[:-1])):.4f})")

    pred = inv(w[-1] + TP @ w[:-1])
    if c.get("positive_only"):
        pred = np.clip(pred, 1e-3, None)
    te = test_df()
    sub = pd.DataFrame({c["id_col"]: te[c["id_col"]].values, c["pred_col"]: pred})
    sample_path = ROOT / c["sample_submission"]
    if sample_path.exists():  # match sample's id order + column order exactly
        sample = pd.read_csv(sample_path)
        assert set(sample[c["id_col"]]) == set(sub[c["id_col"]]), "id mismatch vs sample submission"
        sub = sample[[c["id_col"]]].merge(sub, on=c["id_col"], how="left")[sample.columns]
    assert sub[c["pred_col"]].notna().all() and np.isfinite(sub[c["pred_col"]]).all()
    out = ROOT / a.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
