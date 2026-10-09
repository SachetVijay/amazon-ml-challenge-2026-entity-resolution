"""Leakage check: is the cross-encoder better on OOF pairs whose record it saw during training?"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .cross_encoder import build_train_pairs
from .prep import load_gt

ap = argparse.ArgumentParser()
ap.add_argument("--art", default="artifacts/train")
ap.add_argument("--gt", required=True)
a = ap.parse_args()
a.buckets, a.max_pairs = [2], 800_000
T = build_train_pairs(a)
seen_b = set(T.b.to_numpy())
ce = pd.read_parquet(f"{a.art}/ce_oof.parquet")
oof = pd.read_parquet(f"{a.art}/oof_hyb5.parquet", columns=["a", "b", "s1_id", "o_id", "p"])
d = ce.merge(oof, on=["a", "b"])
ys = set(map(tuple, load_gt(a.gt)[["s1", "other"]].to_numpy()))
d["y"] = [(s, o) in ys for s, o in zip(d.s1_id, d.o_id)]
d["seen"] = d.b.isin(seen_b)
print(f"band pairs {len(d)}, record seen in CE training: {d.seen.mean():.1%}")
for nm, g in d.groupby("seen"):
    print(f"  seen={nm}: n={len(g)} pos={g.y.mean():.3f}  AUC ce={roc_auc_score(g.y, g.ce):.4f}  AUC stage1 p={roc_auc_score(g.y, g.p):.4f}")
