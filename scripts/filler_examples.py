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
sys.path.insert(0, "src")
from er.prep import load_gt
def show(split, c, P, n, label=None):
    art = f"artifacts/{split}"
    rows_all = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
    r = pd.read_parquet(f"{art}/records.parquet", columns=["src", "business_name", "business_address", "name_clean", "addr_clean"], filters=[("country", "==", c)]); r.index = np.flatnonzero(rows_all == c)
    fill = filler_set(r); m, nd = mark(r, P.reset_index(drop=True), fill)
    Q = P.reset_index(drop=True)[m & (nd == 0)]
    Q = Q[(Q.p1 > 0.005) & (Q.p1 < 0.995)]
    if label is not None:
        Q = Q.assign(y=[(a, b) in label for a, b in zip(Q.s1_id, Q.o_id)])
        for yv in (True, False):
            print(f"--- {split} {c} band, name+filler, TRUE={yv}")
            for _, x in Q[Q.y == yv].sample(5, random_state=1).iterrows():
                a, b = r.loc[int(x.a)], r.loc[int(x.b)]; print(f"p1={x.p1:.3f} p={x.p:.3f}\n   S1 {a.business_name} | {a.business_address}\n   S{b.src} {b.business_name} | {b.business_address}")
    else:
        print(f"--- {split} {c} band, name+filler: p1 quantiles {Q.p1.quantile([.25,.5,.75]).round(3).tolist()}")
        for _, x in Q.sample(8, random_state=1).iterrows():
            a, b = r.loc[int(x.a)], r.loc[int(x.b)]; print(f"p1={x.p1:.3f} p={x.p:.3f}\n   S1 {a.business_name} | {a.business_address}\n   S{b.src} {b.business_name} | {b.business_address}")
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "country"]).rename(columns={"p": "p1"})
full = full.merge(oof, on=["a", "b"])
show("train", "US", full[(full.country == "US") & (full.p1 > 0.005) & (full.p1 < 0.995)], 5, ys)
P = pd.read_parquet("artifacts/test/pred_hyb_France.parquet", columns=["a", "b", "p"]).rename(columns={"p": "p1"})
P = P[(P.p1 > 0.005) & (P.p1 < 0.995)].merge(pd.read_parquet("artifacts/test/pred_s2_France.parquet"), on=["a", "b"])
show("test", "France", P, 8)
