import sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from struct_mix_lib import core, acat, NOISE
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "business_name", "addr_clean", "country", "src"])
nm, ad, ct, raw = rt.name_clean.astype(object).fillna("").to_numpy(), rt.addr_clean.astype(object).fillna("").to_numpy(), rt.country.to_numpy(), rt.business_name.astype(object).to_numpy()
idx = pd.Series(rt.index, index=rt.entity_id.astype(object))
F = rt[(rt.country == "France") & (rt.src == 1)]
sdf = Counter(t for n in F.name_clean.astype(object) for t in set(core(n))); N = len(F)
m = pd.read_csv("output_v22.tsv", sep="\t", dtype=str, keep_default_na=False)
m = m[m.source1_entity_id.isin(set(F.entity_id.astype(object)))]
W = defaultdict(list)
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    if not l: continue
    a = idx[s]; ta = core(nm[a])
    for o in l.split(","):
        b = idx[o]; tb = core(nm[b])
        if ta & tb and not (ta - tb) and len(tb - ta) == 1 and not (tb - ta) <= NOISE:
            w = next(iter(tb - ta)); W[w].append((acat(ad[a], ad[b]), raw[a], raw[b], len(l.split(","))))
tot = sum(map(len, W.values())); print("France pure-insert non-noise links:", tot)
top = sorted(W, key=lambda w: -len(W[w]))[:20]
for w in top:
    v = W[w]; print(f"  {w:14s} n={len(v):5d} S1df={sdf[w]/N:.4f} same-addr={np.mean([x[0]=='same' for x in v]):.2f}  e.g. {v[0][1]!r} -> {v[0][2]!r}")
cat = [w for w in W if sdf[w] / N >= 2e-3]
print("links whose inserted word is a common S1 word (S1df>=0.2%):", sum(len(W[w]) for w in cat))
