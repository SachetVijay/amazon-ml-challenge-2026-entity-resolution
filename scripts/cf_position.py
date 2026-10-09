"""Swaps INTO cie/france/fils at the same address in v20's French links: is the new word appended AFTER the legal form
(generator noise signature) or does it take the category word's place (clean replacement)?"""
import re, sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL, to_ascii
from street_mix import street
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == "France")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "src", "name_clean", "business_name", "business_address", "addr_clean"], filters=[("country", "==", "France")]); r.index = rows
sdf = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
m = pd.read_csv("output_v20.tsv", sep="\t", dtype=str, keep_default_na=False)
idx = pd.Series(r.index, index=r.entity_id.astype(object)); s1 = set(r.entity_id[r.src == 1].astype(object))
L = [(idx[s], idx[o]) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) if s in s1 for o in l.split(",") if o]
def appended(raw, v):
    t = [w.lower() for w in re.findall(r"[A-Za-z]+", to_ascii(raw))]
    if v not in t: return False
    i = t.index(v); legal_before = any(w in LEGAL for w in t[:i])
    return legal_before or i == 0          # after a legal form, or moved to the front ("Cie Bourgbarre ...")
res = {}
for a, b in L:
    ta, tb = toks(r.name_clean.loc[a]), toks(r.name_clean.loc[b]); D, A = ta - tb, tb - ta
    if len(D) != 1 or len(A) != 1 or not (ta & tb) or r.addr_clean.loc[b] == "": continue
    u, v = next(iter(D)), next(iter(A))
    if v not in {"cie", "france", "fils"} or sdf.get(u, 0) < 1e-3: continue
    sa, sb = street(r.business_address.loc[a]), street(r.business_address.loc[b])
    if not (sa[0] == sb[0] and sa[0] != "" and fuzz.ratio(sa[1], sb[1]) >= 85): continue
    k = (v, "appended/moved" if appended(r.business_name.loc[b], v) else "in place")
    res.setdefault(k, []).append((a, b))
for k in sorted(res): print(k, len(res[k]))
pd.to_pickle(res, "/tmp/cf_pos.pkl")
for k in sorted(res):
    print("==", k)
    for a, b in res[k][:3]: print(f"   S1 {r.business_name.loc[a]}\n   REC {r.business_name.loc[b]}")
