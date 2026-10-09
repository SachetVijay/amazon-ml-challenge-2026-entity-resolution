import itertools, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast, f05
from er.prep import load_gt
T = "artifacts/train"
tags = ["_v9", "_v10a", "_v11h", "_v11l"]
P = None
for t in tags:
    d = pd.read_parquet(f"{T}/oof_band{t}.parquet", columns=["a", "b", "s1_id", "o_id", "p"]).rename(columns={"p": t})
    P = d if P is None else P.merge(d[["a", "b", t]], on=["a", "b"])
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
samp = set(P.s1_id)
truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
allS = samp | set(truth)
def score(p):
    ex = exclusive(P[["a", "b", "s1_id", "o_id"]].assign(p=p))
    sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
    return np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in allS])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
for t in tags:
    print(f"{t:18s} {score(P[t].to_numpy()):.5f}")
for k in (2, 3, 4):
    for combo in itertools.combinations(tags, k):
        z = np.mean([lg(P[t].to_numpy()) for t in combo], 0)
        print(f"{'+'.join(combo):18s} {score(1 / (1 + np.exp(-z))):.5f}")
