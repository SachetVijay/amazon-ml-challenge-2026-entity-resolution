"""Why are true in-candidate pairs missed?  (a) record given to another S1 by exclusive assignment,
(b) survived exclusivity but cut by expected-F, (c) p tiny.  Plus: for (a), how close was the rival?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full["y"] = np.fromiter(((s, o) in ys for s, o in zip(full.s1_id, full.o_id)), bool, len(full))
ex = exclusive(full[["a", "b", "s1_id", "o_id", "p"]])
sel = expected_f_fast(ex)[["a", "b"]].assign(sel=True)
F = full.merge(ex[["a", "b"]].assign(kept=True), on=["a", "b"], how="left").merge(sel, on=["a", "b"], how="left")
F["kept"] = F.kept.fillna(False).astype(bool); F["sel"] = F.sel.fillna(False).astype(bool)
pos = F[F.y]
fn = pos[~pos.sel]
best_other = F[~F.y].groupby("b").p.max()
fn = fn.assign(rival=fn.b.map(best_other).fillna(0))
print(f"true in-cand pairs {len(pos)}, missed {len(fn)} ({len(fn)/len(pos):.2%})")
print(f"  (a) lost exclusivity to another S1 : {(~fn.kept).sum()}  | rival p median {fn[~fn.kept].rival.median():.3f}, own p median {fn[~fn.kept].p.median():.3f}")
print(f"  (b) kept but cut by expected-F      : {(fn.kept & (fn.p > 0.01)).sum()} | own p median {fn[fn.kept & (fn.p > .01)].p.median():.3f}")
print(f"  (c) kept, p <= 0.01                 : {(fn.kept & (fn.p <= 0.01)).sum()}")
fp = F[F.sel & ~F.y]
own_true = F[F.y].groupby("b").p.max()
fp = fp.assign(true_elsewhere=fp.b.isin(set(F[F.y].b)))
print(f"false selected pairs {len(fp)}: record truly belongs to another candidate S1: {fp.true_elsewhere.mean():.1%} (else a distractor record)")
