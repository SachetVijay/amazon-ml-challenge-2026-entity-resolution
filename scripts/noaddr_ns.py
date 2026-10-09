import sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
def key(n): return " ".join(sorted(t for t in n.split() if t not in LEGAL and len(t) > 1))
# train truth by namesake count k (OOF p>0.5), address-less records, exact core name
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "s1_id", "o_id", "country"])
r = pd.read_parquet("artifacts/train/records.parquet", columns=["name_clean", "addr_clean", "src", "country"])
K = [key(n) for n in r.name_clean.astype(object).fillna("")]; r["k"] = K
ns = r[r.src == 1].groupby(["country", "k"]).size()
ad = r.addr_clean.astype(object).fillna("").to_numpy(); kk = np.array(K, dtype=object); cc = r.country.to_numpy()
ys = set(map(tuple, load_gt("student_resource/dataset/train/train_ground_truth.tsv")[["s1", "other"]].to_numpy()))
o = oof[oof.p > 0.5]
sel = [(a, b, s, x) for a, b, s, x in zip(o.a, o.b, o.s1_id, o.o_id) if ad[b] == "" and kk[a] == kk[b] and kk[a]]
n1 = oof.s1_id.nunique()
D = pd.DataFrame([(min(ns.get((cc[a], kk[a]), 1), 4), (s, x) in ys) for a, b, s, x in sel], columns=["k", "y"])
print("TRAIN (p>0.5 OOF) address-less exact-name pairs by #S1 sharing name: per1k S1 / true rate")
print(D.groupby("k").y.agg(n="size", true="mean").assign(per1k=lambda x: x.n / n1 * 1000).round(3).to_string())
# test v22
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "addr_clean", "src", "country"])
Kt = [key(n) for n in rt.name_clean.astype(object).fillna("")]; rt["k"] = Kt
nst = rt[rt.src == 1].groupby(["country", "k"]).size()
ad = rt.addr_clean.astype(object).fillna("").to_numpy(); kk = np.array(Kt, dtype=object); cc = rt.country.to_numpy()
idx = pd.Series(rt.index, index=rt.entity_id.astype(object))
m = pd.read_csv("output_v22.tsv", sep="\t", dtype=str, keep_default_na=False)
C = Counter(); NS = Counter()
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    a = idx[s]; NS[cc[a]] += 1
    if not l: continue
    for x in l.split(","):
        b = idx[x]
        if ad[b] == "" and kk[a] == kk[b] and kk[a]: C[(cc[a], min(nst.get((cc[a], kk[a]), 1), 4))] += 1
print("TEST v22 address-less exact-name links per 1k S1 by #S1 sharing name:")
for c in ("US", "India", "France"):
    print(" ", c, {k: round(C[(c, k)] / NS[c] * 1000, 2) for k in (1, 2, 3, 4)}, "count k>=2:", sum(C[(c, k)] for k in (2, 3, 4)))
