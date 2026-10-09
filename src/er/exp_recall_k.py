"""Recall@k for one retriever, b2a direction (each S2/S3 record retrieves its top-k S1)."""
import sys, time
import numpy as np, pandas as pd
from .blocking import generate
from .prep import load_gt
art, gtp, kind = sys.argv[1], sys.argv[2], sys.argv[3]
rec = pd.read_parquet(f"{art}/records.parquet"); ids = rec.entity_id.to_numpy()
t = time.time(); c = generate(rec, k_other=20, k_s1=1, threads=22, retrievers=(kind,)); print(f"{time.time()-t:.0f}s")
c = c[c.dir == "b2a"].copy()
c["r"] = c.groupby("b").score.rank(ascending=False, method="first")
gt = load_gt(gtp); g = set(zip(gt.s1, gt.other))
c["hit"] = [(x, y) in g for x, y in zip(ids[c.a.values], ids[c.b.values])]
for k in (1, 3, 5, 10, 20):
    print(f"{kind} k={k:2d} recall {c.hit[c.r <= k].sum() / len(g):.4f} pairs/S1 {(c.r <= k).sum() / (rec.src == 1).sum():.1f}")
