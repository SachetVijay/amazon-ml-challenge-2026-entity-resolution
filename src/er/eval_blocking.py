"""Measure blocking recall on a labelled split.

  python -m er.eval_blocking --art artifacts/dev10 --gt data/dev10/ground_truth.tsv --k-other 10 --k-s1 20
"""
import argparse
import time

import numpy as np
import pandas as pd

from .blocking import generate, pivot_scores
from .prep import load_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--k-other", type=int, default=10)
    ap.add_argument("--k-s1", type=int, default=20)
    ap.add_argument("--threads", type=int, default=20)
    a = ap.parse_args()
    rec = pd.read_parquet(f"{a.art}/records.parquet")
    t = time.time()
    cand = generate(rec, a.k_other, a.k_s1, a.threads)
    print(f"blocking {time.time() - t:.0f}s, raw rows {len(cand)}")
    ids = rec.entity_id.to_numpy()
    gt = load_gt(a.gt)
    key = lambda x, y: pd.Series(x).astype(str) + "|" + pd.Series(y).astype(str)  # noqa: E731
    gt_key = set(key(gt.s1.values, gt.other.values))
    print(f"GT pairs {len(gt_key)}")
    for kind in cand.kind.unique():
        for d in ["b2a", "a2b"]:
            c = cand[(cand.kind == kind) & (cand.dir == d)]
            ck = set(key(ids[c.a.values], ids[c.b.values]))
            print(f"  {kind:10s} {d}: pairs {len(ck):>9d} recall {len(ck & gt_key) / len(gt_key):.4f}")
    u = cand[["a", "b"]].drop_duplicates()
    uk = set(key(ids[u.a.values], ids[u.b.values]))
    n_s1 = (rec.src == 1).sum()
    print(f"UNION pairs {len(uk)} ({len(uk) / n_s1:.1f} per S1) recall {len(uk & gt_key) / len(gt_key):.4f}")
    w = pivot_scores(cand)
    w.to_parquet(f"{a.art}/cand.parquet")
    print("saved", w.shape)


if __name__ == "__main__":
    main()
