"""Where is the remaining validation loss, and how much could a better model on the *residual* uncertain pairs
(p between lo and hi after the band specialist) recover?  Oracle = set those pairs' p to their label."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast, f05
from er.prep import load_gt
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")          # a b s1_id o_id p  (bucket-0 OOF, all pairs)
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
samp = set(full.s1_id)
truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
allS = sorted(samp | set(truth))
ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full["y"] = np.fromiter(((s, o) in ys for s, o in zip(full.s1_id, full.o_id)), bool, len(full))
def score(P, detail=False):
    ex = exclusive(P[["a", "b", "s1_id", "o_id", "p"]])
    sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
    f = np.array([f05(pred.get(s, set()), truth.get(s, set())) for s in allS])
    return (f, pred) if detail else f.mean()
f, pred = score(full, True)
print(f"baseline v11 OOF macro F0.5 {f.mean():.5f} on {len(allS)} S1 ({len(full)} pairs)")
# --- loss decomposition
cand = set(zip(full.s1_id, full.o_id))
L = {"singleton FP": 0.0, "matched: has FP": 0.0, "matched: FN only (pair in cands)": 0.0, "matched: FN only (not in cands)": 0.0}
for s, fs in zip(allS, f):
    if fs >= 1: continue
    t, p = truth.get(s, set()), pred.get(s, set())
    if not t: L["singleton FP"] += 1 - fs
    elif p - t: L["matched: has FP"] += 1 - fs
    elif all((s, o) in cand for o in t - p): L["matched: FN only (pair in cands)"] += 1 - fs
    else: L["matched: FN only (not in cands)"] += 1 - fs
for k, v in L.items(): print(f"  loss {k:36s} {v / len(allS):.5f}")
# --- residual oracle
for lo, hi in ((0.01, 0.99), (0.02, 0.98), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8)):
    m = (full.p > lo) & (full.p < hi)
    o = full.assign(p=np.where(m, full.y.astype(float), full.p))
    print(f"residual ({lo},{hi}): {m.sum():7d} pairs ({m.mean():.2%}), pos rate {full.y[m].mean():.3f} -> oracle F0.5 {score(o):.5f}")
