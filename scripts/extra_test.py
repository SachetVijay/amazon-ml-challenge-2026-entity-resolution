"""Label-free distractor test: a planted distractor is an EXTRA record on its S1, so S1s holding one have ~1 more link
than comparable S1s holding a true noisy copy. Groups defined on v16 France links (before any demotion)."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from street_mix import street
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
isword = lambda t: sdf.get(t, 0) >= 2e-4 and ratio.get(t, 1) < 2
m = pd.read_csv("output_v16/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
idx = pd.Series(r.index, index=r.entity_id.astype(object)); s1 = set(r.entity_id[r.src == 1].astype(object))
links = {s: [idx[o] for o in l.split(",") if o] for s, l in zip(m.source1_entity_id, m.matched_entity_ids) if s in s1}
NATIVE = {"compagnie", "centre", "societe"}; CF = {"cie", "france"}; TRUEFILL = {"groupe", "developpement", "participations", "holding", "distribution", "associes", "international", "services"}
grp = {}
for s, bs in links.items():
    a = idx[s]; ta = toks(r.name_clean.loc[a]); kinds = set()
    for b in bs:
        tb = toks(r.name_clean.loc[b]); D, A = ta - tb, tb - ta
        if not (ta & tb): continue
        if len(D) == 1 and len(A) == 1:
            u, v = next(iter(D)), next(iter(A))
            if isword(u) and not abbrev(u, v) and fuzz.ratio(u, v) < 70:
                if v in CF: kinds.add("swap->cie/france (v25 target)")
                elif v in NATIVE: kinds.add("swap->compagnie/centre/societe (v20: false)")
                elif isword(v) and v not in TRUEFILL and v not in {"fils", "freres"}: kinds.add("category->category (v18: false)")
        elif not D and A and A <= TRUEFILL:
            kinds.add("adds true filler (control: true)")
    for k in kinds: grp.setdefault(k, []).append(len(bs))
allc = [len(v) for v in links.values() if v]
print(f"all French S1 with >=1 link: mean links {np.mean(allc):.3f} (n={len(allc)})")
for k in sorted(grp): print(f"{k:45s}: n={len(grp[k]):6d} mean links {np.mean(grp[k]):.3f}  | share with >=5 links {np.mean(np.array(grp[k]) >= 5):.3f}")
