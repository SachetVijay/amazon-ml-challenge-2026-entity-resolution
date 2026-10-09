"""v18 = v16 with French native<->native category-word swaps at the same address demoted (planted distractors per the
v16->v17c leaderboard delta). Abbreviations (subsequence with same initial) and near-identical variants are kept."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from er.decide import exclusive, expected_f_fast
from street_mix import street
base_dir, out = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "count"
DFMIN = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
FILL_LIKE = {"cie", "france", "compagnie", "societe", "fils", "freres", "centre", "services", "groupe"}
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
P = pd.read_parquet(f"{base_dir}/pred_s2_France.parquet")
C = P[P.p > 0.05].copy()
A, B = r.name_clean.astype(object).loc[C.a].tolist(), r.name_clean.astype(object).loc[C.b].tolist()
cat = np.zeros(len(C), bool)
for i, (x, y) in enumerate(zip(A, B)):
    ta, tb = toks(x), toks(y); dx, dy = ta - tb, tb - ta
    if len(dx) == 1 and len(dy) == 1 and ta & tb:
        u, v = next(iter(dx)), next(iter(dy))
        if sdf.get(u, 0) >= DFMIN and sdf.get(v, 0) >= DFMIN and ratio.get(v, 1) < 2 and not abbrev(u, v) and fuzz.ratio(u, v) < 70 \
                and v not in FILL_LIKE:            # record swaps in a CATEGORY word (the planted-distractor recipe)
            cat[i] = True
C = C[cat]
sa = [street(x) for x in r.business_address.astype(object).loc[C.a]]; sb = [street(x) for x in r.business_address.astype(object).loc[C.b]]
st = cpdist([x[1] for x in sa], [x[1] for x in sb], scorer=fuzz.ratio, workers=-1)
same = np.array([(x[0] == y[0] and x[0] != "") for x, y in zip(sa, sb)]) & (st >= 85)
C = C[same]
ex = exclusive(P); sel = expected_f_fast(ex)
selk = set(zip(sel.a, sel.b))
insel = np.array([(a, b) in selk for a, b in zip(C.a, C.b)])
print(f"category-swap same-address pairs (p>0.05): {len(C)}; currently LINKED: {insel.sum()} ({insel.sum()/len(sel):.2%} of French links); p quantiles of linked {C.p[insel].quantile([.1,.5,.9]).round(3).tolist()}")
pairs = pd.Series([f"{sorted(toks(r.name_clean.loc[a]) - toks(r.name_clean.loc[b]))[0]}->{sorted(toks(r.name_clean.loc[b]) - toks(r.name_clean.loc[a]))[0]}" for a, b in zip(C.a[insel], C.b[insel])]).value_counts()
print("top swaps among linked:", pairs.head(15).to_dict())
if mode == "write":
    key = set(zip(C.a, C.b))
    m = np.array([(a, b) in key for a, b in zip(P.a, P.b)])
    P2 = P.assign(p=np.where(m, np.minimum(P.p, 0.01), P.p))
    ex2 = exclusive(P2); sel2 = expected_f_fast(ex2)
    ids = r.entity_id.astype(object)
    fr = {}
    for a, b in zip(sel2.a, sel2.b): fr.setdefault(ids[a], []).append(ids[b])
    s1_fr = set(ids[r.index[r.src == 1]])
    base = pd.read_csv(sys.argv[4], sep="\t", dtype=str, keep_default_na=False)
    new = [",".join(fr.get(s, [])) if s in s1_fr else l for s, l in zip(base.source1_entity_id, base.matched_entity_ids)]
    base.assign(matched_entity_ids=new).to_csv(out, sep="\t", index=False)
    print(f"wrote {out}: France links {len(sel)} -> {len(sel2)}")
