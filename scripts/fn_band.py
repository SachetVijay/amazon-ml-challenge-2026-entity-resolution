import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast, f05
from er.prep import load_gt
D = pd.read_parquet("artifacts/train/oof_stage2_hyb5_ce.parquet")   # v5 stage 2 OOF: a b s1_id o_id country p1 p2 y
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
samp = set(D.s1_id)
truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
ex = exclusive(D.rename(columns={"p2": "p"}))
sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
print("v5 OOF macro F0.5:", round(np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in samp | set(truth)]), 4))
chosen = set(zip(sel.s1_id, sel.o_id))
T = D[D.y].copy()
T["chosen"] = [(s, o) in chosen for s, o in zip(T.s1_id, T.o_id)]
fn = T[~T.chosen]
bins = [0, 1e-4, 1e-3, 5e-3, 0.05, 0.2, 0.5, 0.8, 0.995, 1.01]
print(f"\ntrue pairs in candidates {len(T)}, missed (FN) {len(fn)}")
print("FN count by stage-1 p1 bin:\n", pd.cut(fn.p1, bins).value_counts().sort_index().to_string())
print("\nFN count by stage-2 p2 bin:\n", pd.cut(fn.p2, bins).value_counts().sort_index().to_string())
F = D[~D.y & D.set_index(["s1_id", "o_id"]).index.isin(list(chosen))]
print(f"\nfalse positives chosen: {len(F)}; by p1 bin:\n", pd.cut(F.p1, bins).value_counts().sort_index().to_string())
# how many pairs would a wider CE band add?
for lo in (5e-3, 1e-3, 1e-4):
    print(f"pairs with {lo} < p1 < 0.995: {((D.p1 > lo) & (D.p1 < 0.995)).sum()}")
