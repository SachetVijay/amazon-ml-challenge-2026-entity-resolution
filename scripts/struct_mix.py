"""Link structure mix per 1k S1: v22 test links per country vs train OOF (p>0.5) with true rate per cell."""
import sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
NOISE = {"center", "services", "service", "partners", "fils", "associes", "cie", "developpement", "groupe", "france"}
def core(n): return {t for t in n.split() if t not in LEGAL and len(t) > 1}
def ncat(na, nb):
    ta, tb = core(na), core(nb); D, A = ta - tb, tb - ta
    if not ta & tb: return "disjoint"
    if not D and not A: return "exact"
    if not D and len(A) == 1: return "app1-noise" if A <= NOISE else "app1-other"
    if len(D) == 1 and not A: return "drop1"
    if len(D) == 1 and len(A) == 1: return "swap1-noise" if A <= NOISE else "swap1-other"
    return "multi"
def acat(x, y):
    if not x or not y: return "noaddr"
    if x == y: return "same"
    sx, sy = x.split(), y.split()
    nx, ny = [t for t in sx if any(c.isdigit() for c in t)], [t for t in sy if any(c.isdigit() for c in t)]
    tx, ty = set(sx) - set(nx), set(sy) - set(ny)
    if tx == ty: return "samestreet-num" if nx != ny else "same~"
    if len(tx & ty) >= max(1, min(len(tx), len(ty)) - 1): return "near"
    return "diff"
# train
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "s1_id", "o_id"])
n1 = oof.s1_id.nunique(); oof = oof[oof.p > 0.5]
r = pd.read_parquet("artifacts/train/records.parquet", columns=["name_clean", "addr_clean"])
nm, ad = r.name_clean.astype(object).fillna("").to_numpy(), r.addr_clean.astype(object).fillna("").to_numpy()
ys = set(map(tuple, load_gt("student_resource/dataset/train/train_ground_truth.tsv")[["s1", "other"]].to_numpy()))
cnt, tru = Counter(), Counter()
for a, b, s, o in zip(oof.a, oof.b, oof.s1_id, oof.o_id):
    k = (ncat(nm[a], nm[b]), acat(ad[a], ad[b])); cnt[k] += 1; tru[k] += (s, o) in ys
T = pd.DataFrame([(k[0], k[1], v / n1 * 1000, tru[k] / v) for k, v in cnt.items()], columns=["name", "addr", "train_per1k", "train_true"])
# test
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "addr_clean", "country"])
nm, ad, ct = rt.name_clean.astype(object).fillna("").to_numpy(), rt.addr_clean.astype(object).fillna("").to_numpy(), rt.country.to_numpy()
idx = pd.Series(rt.index, index=rt.entity_id.astype(object))
m = pd.read_csv("output_v22.tsv", sep="\t", dtype=str, keep_default_na=False).sample(frac=0.35, random_state=0)
C, NS = Counter(), Counter()
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    a = idx[s]; c = ct[a]; NS[c] += 1
    if not l: continue
    for o in l.split(","):
        b = idx[o]; C[(c, ncat(nm[a], nm[b]), acat(ad[a], ad[b]))] += 1
for c in NS:
    T[c] = [C[(c, x, y)] / NS[c] * 1000 for x, y in zip(T.name, T.addr)]
T["FR/US"] = T.France / T.US.clip(lower=0.5)
pd.set_option("display.width", 200)
print(T[T.train_per1k.ge(2) | T.France.ge(2)].sort_values("France", ascending=False).round(3).to_string(index=False))
