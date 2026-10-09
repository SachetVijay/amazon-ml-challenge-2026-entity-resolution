import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.prep import load_gt
full = pd.read_parquet("artifacts/train/oof_band_v11l.parquet")
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
R = full[(full.p > 0.25) & (full.p < 0.6)].copy()
R["y"] = np.fromiter(((s, o) in ys for s, o in zip(R.s1_id, R.o_id)), bool, len(R))
rec = pd.read_parquet("artifacts/train/records.parquet", columns=["business_name", "business_address", "country"])
S = pd.concat([R[R.y].sample(22, random_state=3), R[~R.y].sample(22, random_state=3)])
for _, r in S.iterrows():
    A, B = rec.iloc[int(r.a)], rec.iloc[int(r.b)]
    print(f"{'TRUE ' if r.y else 'FALSE'} p={r.p:.2f} {A.country[:2]} | {A.business_name} | {A.business_address}\n{'':17s}{r.o_id[:2]} | {B.business_name} | {B.business_address}")
