"""Pairs just below the band (1e-3 < p1 <= 0.005) never reach the cross-encoders. How many does a CE strongly accept?
France scored with the French-adapted ce1, US with the original; US validation truth for the same slice."""
import sys
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.cross_encoder import score
from er.prep import load_gt
rows = []
for c, model in (("US", "models/ce_base2"), ("France", "models/ce_base2_fr")):
    p1 = pd.read_parquet(f"artifacts/test/pred_hyb_{c}.parquet", columns=["a", "b", "p"])
    sb = p1[(p1.p > 1e-3) & (p1.p <= 0.005)]
    n1 = p1.a.nunique(); s = sb.sample(min(30000, len(sb)), random_state=0)
    s[["a", "b"]].to_parquet(f"/tmp/sb_{c}.parquet")
    score(SimpleNamespace(pairs=f"/tmp/sb_{c}.parquet", band=None, art="artifacts/test", model=model, out=f"/tmp/sb_{c}_ce.parquet", max_len=128, view="raw"))
    ce = pd.read_parquet(f"/tmp/sb_{c}_ce.parquet").ce
    rows.append((c, len(sb) / n1, np.mean(ce > 3), np.mean(ce > 0)))
for c, per, hi, pos in rows:
    print(f"test {c:7s}: sub-band pairs per S1 {per:.3f} | CE logit > 3: {hi:.2%} ({hi*per:.4f}/S1) | > 0: {pos:.2%}")
# validation truth of the US sub-band slice with ce1 OOF-like scores: use train oof pairs of bucket 0, score a sample with ce_base2 (bucket 0 never seen)
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
o = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "s1_id", "o_id", "country"])
o = o[(o.p > 1e-3) & (o.p <= 0.005)].sample(30000, random_state=0)
o[["a", "b"]].to_parquet("/tmp/sb_val.parquet")
score(SimpleNamespace(pairs="/tmp/sb_val.parquet", band=None, art="artifacts/train", model="models/ce_base2", out="/tmp/sb_val_ce.parquet", max_len=128, view="raw"))
o = o.merge(pd.read_parquet("/tmp/sb_val_ce.parquet"), on=["a", "b"]); o["y"] = [(s_, t) in ys for s_, t in zip(o.s1_id, o.o_id)]
print(f"validation US+India sub-band: true rate {o.y.mean():.3%} | CE>3: n {np.sum(o.ce > 3)} true rate {o.y[o.ce > 3].mean():.1%} | CE>0 true rate {o.y[o.ce > 0].mean():.1%}")
