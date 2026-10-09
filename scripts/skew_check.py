"""Residual FALSE pairs: does the record belong to some other S1 (invisible to bucket-0 validation), or is it a
pure distractor?  If the former is common, the band model never learned real competition (test has it)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.prep import load_gt
full = pd.read_parquet("artifacts/train/oof_band_v11l.parquet")
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
owner = gt.drop_duplicates("other").set_index("other").s1
full["y"] = np.fromiter(((s, o) in ys for s, o in zip(full.s1_id, full.o_id)), bool, len(full))
bucket = lambda ids: pd.util.hash_array(np.asarray(ids, dtype=object)) % 8
for lo, hi, nm in ((0.01, 0.99, "residual"), (0.5, 1.01, "selected-ish p>0.5")):
    R = full[(full.p > lo) & (full.p < hi) & ~full.y]
    own = R.o_id.map(owner)
    has = own.notna()
    ob = bucket(own[has].to_numpy())
    print(f"{nm} FALSE pairs {len(R)}: record matched to another S1 {has.mean():.1%} "
          f"(of which owner in bucket 0 {np.mean(ob == 0):.1%}), pure distractor {(~has).mean():.1%}")
# how often is the true owner of a record a *candidate* of it at all (test would see it) -- need full-density cands:
# proxy: same record's pairs in bucket 0 only
R = full[(full.p > 0.01) & (full.p < 0.99) & ~full.y]
own = R.o_id.map(owner)
x = R[own.notna()].assign(owner=own[own.notna()])
print("examples (S1 in sample | record | true owner S1):")
rec = pd.read_parquet("artifacts/train/records.parquet", columns=["entity_id", "business_name", "business_address"])
idx = pd.Series(np.arange(len(rec)), index=rec.entity_id.to_numpy())
for _, r in x.sample(12, random_state=1).iterrows():
    A, B, O = rec.iloc[int(r.a)], rec.iloc[int(r.b)], rec.iloc[int(idx[r.owner])]
    print(f"p={r.p:.2f}\n  cand : {A.business_name} | {A.business_address}\n  rec  : {B.business_name} | {B.business_address}\n  owner: {O.business_name} | {O.business_address}")
