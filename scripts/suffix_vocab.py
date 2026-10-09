"""Suffix-noise vocabulary: last token of the other record when it is ADDED (not in S1) and comes after the legal form
or at the end. Train: split by label. Test France: v16 links."""
import sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def last_added(na, nb):
    ta = set(na.split()); w = nb.split()
    return w[-1] if w and w[-1] not in ta and w[-1] not in LEGAL and len(w[-1]) > 1 else None
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "s1_id", "o_id", "country"]); oof = oof[oof.p > 0.05]
r = pd.read_parquet("artifacts/train/records.parquet", columns=["name_clean", "src"])
ys = set(map(tuple, load_gt("student_resource/dataset/train/train_ground_truth.tsv")[["s1", "other"]].to_numpy()))
nm = r.name_clean.astype(object).to_numpy()
s1n = r.name_clean[r.src == 1].astype(object)
sdf = Counter(t for n in s1n for t in set(n.split())); N1 = len(s1n)
for c in ("US", "India"):
    P = oof[oof.country == c]; T, Fz = Counter(), Counter()
    for a, b, s, o in zip(P.a, P.b, P.s1_id, P.o_id):
        w = last_added(nm[a], nm[b])
        if w: (T if (s, o) in ys else Fz)[w] += 1
    tot = sum(T.values()) + sum(Fz.values())
    print(f"TRAIN {c}: appended-last-token pairs {tot}")
    print("  top:", [(w, T[w] + Fz[w], f"{T[w]/(T[w]+Fz[w]):.2f}", f"S1df {sdf[w]/N1:.4f}") for w, _ in (T + Fz).most_common(12)])
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "country", "src"])
F = rt[rt.country == "France"]; idx = pd.Series(F.index, index=F.entity_id.astype(object)); nm = rt.name_clean.astype(object).to_numpy()
s1f = F.name_clean[F.src == 1].astype(object); fdf = Counter(t for n in s1f for t in set(n.split()))
m = pd.read_csv("output_v16/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
s1 = set(F.entity_id[F.src == 1].astype(object)); C = Counter()
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    if s not in s1 or not l: continue
    for o in l.split(","):
        w = last_added(nm[idx[s]], nm[idx[o]])
        if w: C[w] += 1
print(f"TEST France v16 links: appended-last-token links {sum(C.values())}")
print("  top:", [(w, n, f"S1df {fdf[w]/len(s1f):.4f}") for w, n in C.most_common(15)])
