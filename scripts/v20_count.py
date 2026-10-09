"""Broader category-change patterns among v18's French links (after v18's demotion)."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from street_mix import street
FILL_LIKE = {"cie", "france", "compagnie", "societe", "fils", "freres", "centre", "services", "groupe", "and", "et", "associes", "service", "developpement"}
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def abbrev(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if s[0] != l[0]: return False
    i = 0
    for ch in l:
        if i < len(s) and ch == s[i]: i += 1
    return i == len(s)
rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == "France")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "src", "name_clean", "business_address", "addr_clean"], filters=[("country", "==", "France")]); r.index = rows
sdf = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
odf = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
ratio = (odf + 1e-4) / (sdf.reindex(odf.index).fillna(0) + 1e-4)
cat = lambda t: sdf.get(t, 0) >= 1e-3 and ratio.get(t, 1) < 2 and t not in FILL_LIKE
real = lambda t: sdf.get(t, 0) >= 1e-3 and ratio.get(t, 1) < 2
m = pd.read_csv("~/amlc/output_v18.tsv", sep="\t", dtype=str, keep_default_na=False)
idx = pd.Series(r.index, index=r.entity_id.astype(object))
fr = set(r.entity_id[r.src == 1].astype(object))
L = [(idx[s], idx[o]) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) if s in fr for o in l.split(",") if o]
cnt = {}
ex = {}
for a, b in L:
    ta, tb = toks(r.name_clean.loc[a]), toks(r.name_clean.loc[b]); D, A = ta - tb, tb - ta
    if not (ta & tb): continue
    hit = [(u, v) for u in D for v in A if real(u) and cat(v) and not abbrev(u, v) and fuzz.ratio(u, v) < 70]
    if not hit: continue
    one = len(D) == 1 and len(A) == 1
    ab = r.addr_clean.loc[b]
    if ab == "": k = "X2 swap, record address-less" + (" (pure)" if one else " (+noise)")
    else:
        sa, sb = street(r.business_address.loc[a]), street(r.business_address.loc[b])
        samest = fuzz.ratio(sa[1], sb[1]) >= 85; samenum = sa[0] == sb[0] and sa[0] != ""
        if samest and samenum: k = "X1 swap+noise, same address" if not one else "v18-pattern (should be ~0)"
        elif samest: k = "X3 swap, same street, diff/no number"
        else: k = "X4 swap, different street"
    cnt[k] = cnt.get(k, 0) + 1; ex.setdefault(k, []).append((a, b))
print({k: v for k, v in sorted(cnt.items())}, "of", len(L), "French links")
rng = np.random.default_rng(0)
for k, v in sorted(ex.items()):
    print("==", k)
    for i in rng.choice(len(v), min(4, len(v)), replace=False):
        a, b = v[i]; print(f"   S1 {r.name_clean.loc[a]} | {r.business_address.loc[a]}\n   REC {r.name_clean.loc[b]} | {r.business_address.loc[b]}")
