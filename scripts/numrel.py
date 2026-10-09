"""Relation between house numbers for 'clean' number-noise pairs (record adds no name tokens, same street, different number):
digit-drop/insert (one is a prefix or suffix of the other), transposition, one digit changed, +-1..2 adjacent, other.
US/India validation: truth rate + acceptance per relation.  Test: counts per S1 + acceptance per relation."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from clean_numnoise import pick, toks
from street_mix import street
from er.prep import load_gt
def rel(x, y):
    if x.startswith(y) or x.endswith(y) or y.startswith(x) or y.endswith(x): return "drop/insert digit"
    if len(x) == len(y):
        d = [i for i in range(len(x)) if x[i] != y[i]]
        if len(d) == 2 and x[d[0]] == y[d[1]] and x[d[1]] == y[d[0]]: return "transposition"
        if len(d) == 1: return "one digit changed"
    try:
        if abs(int(x) - int(y)) <= 2: return "adjacent +-2"
    except ValueError: pass
    return "other"
split = sys.argv[1]
if split == "train":
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
    oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "country"]).rename(columns={"p": "p1"}); full = full.merge(oof, on=["a", "b"])
out = {}
for c in (("US", "India") if split == "train" else ("US", "India", "France")):
    art = f"artifacts/{split}"
    rows = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["business_address", "name_clean", "src"], filters=[("country", "==", c)]); r.index = rows
    if split == "train":
        P = full[(full.country == c) & (full.p1 > 1e-3)].reset_index(drop=True); n1 = P.s1_id.nunique()
    else:
        P = pd.read_parquet(f"{art}/pred_s2_{c}.parquet"); n1 = (r.src == 1).sum()
    Q = pick(r, P)
    na = [street(x)[0] for x in r.business_address.astype(object).loc[Q.a]]; nb = [street(x)[0] for x in r.business_address.astype(object).loc[Q.b]]
    Q["rel"] = [rel(x, y) for x, y in zip(na, nb)]
    agg = {"pairs/S1": Q.groupby("rel").size() / n1, "accepted/S1": Q[Q.p > .5].groupby("rel").size() / n1}
    if split == "train":
        Q["y"] = [(a, b) in ys for a, b in zip(Q.s1_id, Q.o_id)]
        agg["TRUE/S1"] = Q[Q.y].groupby("rel").size() / n1; agg["true rate"] = Q.groupby("rel").y.mean()
    print(f"== {split} {c}\n" + pd.DataFrame(agg).fillna(0).round(4).to_string(), flush=True)
