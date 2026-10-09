import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast, f05
from er.prep import load_gt
D = pd.read_parquet("artifacts/train/oof_stage2_hyb5_v6.parquet")   # a b s1_id o_id country p1 p2 y
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
samp = set(D.s1_id)
truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
allS = samp | set(truth)
def score(col):
    ex = exclusive(D.rename(columns={col: "p"})[["a", "b", "p", "s1_id", "o_id"]])
    sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
    return np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in allS])
print("v6 stage 2 (actual):", round(score("p2"), 5))
for lo, hi in [(0.005, 0.995), (0.02, 0.98), (0.1, 0.9)]:
    band = (D.p1 > lo) & (D.p1 < hi)
    D["o"] = np.where(band, D.y.astype(float) * 0.999 + 0.0005, D.p2)
    print(f"oracle on stage-1 band {lo}-{hi} ({band.mean():.2%} of pairs): {score('o'):.5f}")
D["o"] = D.y.astype(float) * 0.999 + 0.0005
print("oracle on ALL candidate pairs (blocking-recall ceiling):", round(score("o"), 5))
