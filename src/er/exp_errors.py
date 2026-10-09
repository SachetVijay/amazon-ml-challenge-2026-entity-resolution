"""Error analysis on out-of-fold predictions: where does macro F0.5 go?

  python -m er.exp_errors --art artifacts/train --gt .../train_ground_truth.tsv --tag _hyb5
"""
import argparse

import numpy as np
import pandas as pd

from .decide import exclusive, expected_f_fast, f05
from .prep import load_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", default="_hyb5")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    ev = pd.read_parquet(f"{a.art}/oof{a.tag}.parquet")
    gt = load_gt(a.gt)
    samp = set(ev.s1_id.unique())
    gt = gt[gt.s1.isin(samp) | (pd.util.hash_array(gt.s1.to_numpy(dtype=object)) % 8 == 0)]
    truth = gt.groupby("s1").other.agg(set).to_dict()
    ex = exclusive(ev)
    sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id", "country"]], on=["a", "b"])
    pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
    cand = {s: set(g) for s, g in ev.groupby("s1_id")["o_id"]}
    s1_all = sorted(samp | set(truth))
    rows = []
    for s in s1_all:
        P, T, C = pred.get(s, set()), truth.get(s, set()), cand.get(s, set())
        rows.append((s, f05(P, T), len(T), len(P), len(P - T), len((T - P) & C), len(T - C)))
    R = pd.DataFrame(rows, columns=["s1", "f", "n_true", "n_pred", "fp", "fn_scored", "fn_unretrieved"])
    print(f"macro F0.5 {R.f.mean():.4f} over {len(R)} S1")
    loss = 1 - R.f
    print("loss share by entity type:")
    for nm, m in [("singleton, predicted non-empty", (R.n_true == 0) & (R.n_pred > 0)),
                  ("non-singleton, predicted empty", (R.n_true > 0) & (R.n_pred == 0)),
                  ("has false positives", (R.n_true > 0) & (R.fp > 0)),
                  ("only false negatives", (R.n_true > 0) & (R.fp == 0) & (R.n_pred > 0) & (R.f < 1))]:
        print(f"  {nm:34s} n={m.sum():7d}  loss share {loss[m].sum() / loss.sum():.1%}")
    print(f"missed true links: scored-but-rejected {R.fn_scored.sum()}, never retrieved {R.fn_unretrieved.sum()}; "
          f"false positive links {R.fp.sum()}")

    # examples
    need = set()
    fp_pairs = sel[[o not in truth.get(s, ()) for s, o in zip(sel.s1_id, sel.o_id)]]
    fn_pairs = ex[[(o in truth.get(s, ())) and (o not in pred.get(s, ())) for s, o in zip(ex.s1_id, ex.o_id)]]
    fp_s = fp_pairs.sample(min(a.n, len(fp_pairs)), random_state=0)
    fn_s = fn_pairs.sample(min(a.n, len(fn_pairs)), random_state=0)
    for d in (fp_s, fn_s):
        need |= set(d.s1_id) | set(d.o_id)
    for d in (fp_s,):
        need |= {x for s in d.s1_id for x in truth.get(s, ())}
    rec = pd.read_parquet(f"{a.art}/records.parquet", columns=["entity_id", "business_name", "business_address"],
                          filters=[("entity_id", "in", list(need))]).set_index("entity_id")
    fmt = lambda i: f"{rec.at[i, 'business_name']} | {rec.at[i, 'business_address']}" if i in rec.index else i  # noqa
    print("\n=== FALSE POSITIVES (predicted, wrong) ===")
    for r in fp_s.itertuples():
        print(f"\n[p={r.p:.2f} {r.country}] S1: {fmt(r.s1_id)}\n    predicted: {fmt(r.o_id)}")
        tr = list(truth.get(r.s1_id, ()))[:2]
        print("    true matches:", "; ".join(fmt(t) for t in tr) if tr else "(singleton)")
    print("\n=== FALSE NEGATIVES (true, scored but rejected) ===")
    for r in fn_s.itertuples():
        print(f"\n[p={r.p:.2f} {r.country}] S1: {fmt(r.s1_id)}\n    missed: {fmt(r.o_id)}")
    # calibration of p on the exclusive winners
    ex["y"] = [o in truth.get(s, ()) for s, o in zip(ex.s1_id, ex.o_id)]
    ex["bin"] = pd.cut(ex.p, [0, .1, .3, .5, .7, .8, .9, .95, .99, 1.0])
    print("\ncalibration (exclusive winners):\n", ex.groupby("bin", observed=True).y.agg(["mean", "size"]).round(3))


if __name__ == "__main__":
    main()
