"""Tune the decision rule on out-of-fold pair probabilities (dev)."""
import sys, time
import numpy as np, pandas as pd
from .decide import exclusive, expected_f_fast, macro_f05, to_lists
from .prep import load_gt
art, gtp = sys.argv[1], sys.argv[2]
rec = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src"])
ids = rec.entity_id.to_numpy(); s1 = ids[rec.src.to_numpy() == 1]
P = pd.read_parquet(f"{art}/oof.parquet")
gt = load_gt(gtp); truth = {}
for s, o in zip(gt.s1, gt.other): truth.setdefault(s, set()).add(o)
ex = exclusive(P)
for t in (0.6, 0.7, 0.75, 0.8, 0.85):
    print(f"exclusive thr {t}: {macro_f05(to_lists(ex[ex.p >= t], ids), truth, s1):.4f}")
for w in (0.5, 1.0, 1.5, 2.0):
    t0 = time.time(); sel = expected_f_fast(ex, empty_weight=w)
    print(f"exclusive + expected-F fast (empty_weight {w}): {macro_f05(to_lists(sel, ids), truth, s1):.4f}  {time.time()-t0:.1f}s")
