import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from street_mix import street
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
NATIVE_FILL = {"centre", "compagnie", "societe", "fils", "freres"}
TRUE_FILL = {"cie", "france", "groupe", "services", "and", "et", "associes", "service", "developpement"}
rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == "France")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "src", "name_clean", "business_address", "addr_clean"], filters=[("country", "==", "France")]); r.index = rows
sdf = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
odf = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
ratio = (odf + 1e-4) / (sdf.reindex(odf.index).fillna(0) + 1e-4)
isword = lambda t: sdf.get(t, 0) >= 1e-3 and ratio.get(t, 1) < 2
m = pd.read_csv("output_v18.tsv", sep="\t", dtype=str, keep_default_na=False)
idx = pd.Series(r.index, index=r.entity_id.astype(object)); s1 = set(r.entity_id[r.src == 1].astype(object))
L = [(idx[s], idx[o]) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) if s in s1 for o in l.split(",") if o]
cnt, ex = {}, {}
for a, b in L:
    ta, tb = toks(r.name_clean.loc[a]), toks(r.name_clean.loc[b]); D, A = ta - tb, tb - ta
    if not (ta & tb) or r.addr_clean.loc[b] == "": continue
    sa, sb = street(r.business_address.loc[a]), street(r.business_address.loc[b])
    if not (sa[0] == sb[0] and sa[0] != "" and fuzz.ratio(sa[1], sb[1]) >= 85): continue
    k = None
    if len(D) == 1 and len(A) == 1:
        u, v = next(iter(D)), next(iter(A))
        if isword(u) and v in NATIVE_FILL: k = f"swap -> native-filler ({v})"
        elif isword(u) and v in {"cie", "france"}: k = f"swap -> {v}"
    elif not D and A:
        cats = [v for v in A if isword(v) and v not in TRUE_FILL and v not in NATIVE_FILL]
        if cats and len(A) == 1: k = "adds one category word"
    if k: cnt[k] = cnt.get(k, 0) + 1; ex.setdefault(k, []).append((a, b))
print(dict(sorted(cnt.items(), key=lambda x: -x[1])))
for k in ("adds one category word",):
    for a, b in ex.get(k, [])[:5]: print(f"   S1 {r.name_clean.loc[a]} | REC {r.name_clean.loc[b]}")
