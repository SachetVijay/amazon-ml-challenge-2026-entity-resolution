"""Validation true F0.5 by namesake PROXIMITY: ks = # S1 with the same core name on the same street (full density);
reweight to each test country's ks distribution."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from street_mix import street
from er.decide import exclusive, expected_f_fast, f05
from er.prep import load_gt
def ks(split):
    r = pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["entity_id", "src", "country", "name_core", "business_address"]); r = r[r.src == 1].copy()
    r["st"] = [street(x)[1] for x in r.business_address.astype(object)]
    r["ks"] = r.groupby(["country", "name_core", "st"]).entity_id.transform("size").clip(upper=4)
    r["kn"] = r.groupby(["country", "name_core"]).entity_id.transform("size").clip(upper=5)
    return r
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
samp = set(full.s1_id); truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
ex = exclusive(full[["a", "b", "s1_id", "o_id", "p"]]); sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
pred = sel.groupby("s1_id").o_id.agg(set).to_dict()
T = pd.DataFrame({"s1": sorted(samp | set(truth))}); T["F"] = [f05(pred.get(s, set()), truth.get(s, set())) for s in T.s1]
tr = ks("train"); T = T.merge(tr[["entity_id", "ks", "kn"]], left_on="s1", right_on="entity_id")
g = T.groupby("ks").agg(n=("F", "size"), trueF=("F", "mean")); print(g.round(4).to_string())
g2 = T.groupby(["kn", "ks"]).F.mean()
te = ks("test")
for c, h in te.groupby("country"):
    w = h.groupby(["kn", "ks"]).size() / len(h); m = g2.reindex(w.index)
    print(f"{c:7s}: same-street namesake share {h.ks.value_counts(normalize=True).sort_index().round(3).to_dict()} -> predicted F {float((w * m).sum() / w[m.notna()].sum()):.4f}")
