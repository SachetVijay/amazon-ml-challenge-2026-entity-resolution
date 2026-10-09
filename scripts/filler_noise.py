"""Pattern F: record = S1 name tokens minus <=1 dropped word plus >=1 FILLER tokens (per-country filler score
df_records/df_S1 >= 2, label-free), same address. Validation: truth rate + model p. Test: counts + p per country,
split by whether the pair was in the band (CE-decided) or confident in stage 1."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src")
from er.normalize import LEGAL
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def filler_set(r):
    s = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
    o = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
    ratio = (o + 1e-4) / (s.reindex(o.index).fillna(0) + 1e-4)
    return set(ratio[(ratio >= 2) & (o >= 2e-4)].index)
def mark(r, P, fill):
    A = r.name_clean.astype(object).loc[P.a].tolist(); B = r.name_clean.astype(object).loc[P.b].tolist()
    ok = np.zeros(len(P), bool); ndrop = np.zeros(len(P), int)
    for i, (x, y) in enumerate(zip(A, B)):
        ta, tb = toks(x), toks(y); add, drop = tb - ta, ta - tb
        if add and add <= fill and len(drop) <= 1 and (ta & tb): ok[i] = True; ndrop[i] = len(drop)
    ad = np.zeros(len(P)); idx = np.flatnonzero(ok)
    ad[idx] = cpdist(r.addr_clean.astype(object).loc[P.a.to_numpy()[idx]].tolist(), r.addr_clean.astype(object).loc[P.b.to_numpy()[idx]].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    return ok & (ad >= 90), ndrop
split = sys.argv[1]; art = f"artifacts/{split}"
rows_all = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
if split == "train":
    from er.prep import load_gt
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    full = pd.read_parquet(f"{art}/oof_band_v12.parquet")
    oof = pd.read_parquet(f"{art}/oof_hyb5.parquet", columns=["a", "b", "p", "country"]).rename(columns={"p": "p1"})
    full = full.merge(oof, on=["a", "b"])
for c in (("US", "India") if split == "train" else ("US", "India", "France")):
    r = pd.read_parquet(f"{art}/records.parquet", columns=["src", "name_clean", "addr_clean"], filters=[("country", "==", c)]); r.index = np.flatnonzero(rows_all == c)
    fill = filler_set(r)
    if split == "train":
        P = full[(full.country == c) & (full.p1 > 1e-3)].reset_index(drop=True)
    else:
        P = pd.read_parquet(f"{art}/pred_hyb_{c}.parquet", columns=["a", "b", "p"]).rename(columns={"p": "p1"})
        P = P[P.p1 > 1e-3].merge(pd.read_parquet(f"{art}/pred_s2_{c}.parquet"), on=["a", "b"]).reset_index(drop=True)
    m, nd = mark(r, P, fill)
    Q = P[m].assign(ndrop=nd[m]); Q["zone"] = np.where(Q.p1 >= 0.995, "confident", "band")
    msg = f"{split} {c:7s}: fillers={len(fill)} (e.g. {sorted(fill)[:8]}) | pattern pairs {len(Q)}"
    print(msg, flush=True)
    for (z, k), g in Q.groupby(["zone", "ndrop"]):
        s = f"    {z:9s} drop={k}: n={len(g):6d} final p>0.5 {np.mean(g.p > .5):.1%}  median p {g.p.median():.3f}"
        if split == "train": s += f" | TRUE rate {np.mean([(a, b) in ys for a, b in zip(g.s1_id, g.o_id)]):.3f}"
        print(s, flush=True)
