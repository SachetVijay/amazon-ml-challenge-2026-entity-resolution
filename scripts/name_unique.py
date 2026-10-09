"""Full-density S1 name multiplicity for missed true pairs vs residual false pairs."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
full = pd.read_parquet("artifacts/train/oof_band_v11l.parquet")
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full["y"] = np.fromiter(((s, o) in ys for s, o in zip(full.s1_id, full.o_id)), bool, len(full))
rec = pd.read_parquet("artifacts/train/records.parquet", columns=["src", "country", "name_core", "addr_clean"])
s1 = rec[rec.src == 1]
cnt = s1.groupby(["country", "name_core"]).size()
key = lambda idx: pd.MultiIndex.from_arrays([rec.country.to_numpy()[idx], rec.name_core.to_numpy()[idx]])
full["n_same_a"] = cnt.reindex(key(full.a.to_numpy())).fillna(0).to_numpy()
full["n_same_b"] = cnt.reindex(key(full.b.to_numpy())).fillna(0).to_numpy()
full["b_noaddr"] = rec.addr_clean.to_numpy()[full.b.to_numpy()] == ""
ex = exclusive(full[["a", "b", "s1_id", "o_id", "p"]])
sel = set(map(tuple, expected_f_fast(ex)[["a", "b"]].to_numpy()))
full["sel"] = [(a, b) in sel for a, b in zip(full.a, full.b)]
FN = full[full.y & ~full.sel]
FPr = full[~full.y & (full.p > 0.01) & (full.p < 0.99)]
TPr = full[full.y & (full.p > 0.01) & (full.p < 0.99)]
for nm, g in (("missed TRUE pairs", FN), ("residual TRUE pairs", TPr), ("residual FALSE pairs", FPr)):
    u = (g.n_same_b <= 1)
    print(f"{nm:22s} n={len(g):6d} | record has no address {g.b_noaddr.mean():.1%} | record's exact name held by "
          f"<=1 S1: {u.mean():.1%}, >=2 S1: {(g.n_same_b >= 2).mean():.1%} | p median {g.p.median():.3f}")
print("missed TRUE with unique record name (<=1 S1):  p median", round(FN[FN.n_same_b <= 1].p.median(), 3))
