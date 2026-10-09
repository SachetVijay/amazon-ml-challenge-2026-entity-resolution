"""'Clean' number noise: record name adds NO tokens to the S1 name (legal forms ignored), same street, different number.
US/India validation: truth rate + acceptance by number length. France/US/India test: count + acceptance."""
import re, sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from street_mix import street
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def pick(r, P):
    A, B = P.a.to_numpy(), P.b.to_numpy()
    na = r.name_clean.astype(object).loc[A].tolist(); nb = r.name_clean.astype(object).loc[B].tolist()
    clean = np.array([bool(toks(y)) and toks(y) <= toks(x) for x, y in zip(na, nb)])
    P = P[clean]; A, B = P.a.to_numpy(), P.b.to_numpy()
    sa = [street(x) for x in r.business_address.astype(object).loc[A]]; sb = [street(x) for x in r.business_address.astype(object).loc[B]]
    st = cpdist([x[1] for x in sa], [x[1] for x in sb], scorer=fuzz.ratio, workers=-1)
    ok = (st >= 85) & np.array([x[0] != y[0] and x[0] != "" and y[0] != "" for x, y in zip(sa, sb)])
    P = P[ok].copy(); P["nlen"] = [("1-2" if len(x[0]) <= 2 else "3+") for x, o in zip(sa, ok) if o]
    return P
split = sys.argv[1]
if split == "train":
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
    oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "country"]).rename(columns={"p": "p1"}); full = full.merge(oof, on=["a", "b"])
for c in (("US", "India") if split == "train" else ("US", "India", "France")):
    art = f"artifacts/{split}"
    rows = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["business_address", "name_clean", "src"], filters=[("country", "==", c)]); r.index = rows
    n1 = (r.src == 1).sum()
    if split == "train":
        P = full[(full.country == c) & (full.p1 > 1e-3)].reset_index(drop=True); n1 = P.s1_id.nunique()
    else:
        P = pd.read_parquet(f"{art}/pred_s2_{c}.parquet")
    Q = pick(r, P)
    for k, g in Q.groupby("nlen"):
        s = f"{split} {c:7s} num {k}: pairs/S1 {len(g)/n1:.4f} | accepted(p>0.5)/S1 {np.sum(g.p > .5)/n1:.4f} ({np.mean(g.p > .5):.1%})"
        if split == "train": s += f" | TRUE/S1 {np.sum([(a, b) in ys for a, b in zip(g.s1_id, g.o_id)])/n1:.4f} (rate {np.mean([(a, b) in ys for a, b in zip(g.s1_id, g.o_id)]):.1%})"
        print(s, flush=True)
