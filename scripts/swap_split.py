import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.decide import exclusive, expected_f_fast
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def abbrev(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if s[0] != l[0]: return False
    i = 0
    for ch in l:
        if i < len(s) and ch == s[i]: i += 1
    return i == len(s)
rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == "France")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["src", "name_clean"], filters=[("country", "==", "France")]); r.index = rows
sdf = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
odf = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
ratio = (odf + 1e-4) / (sdf.reindex(odf.index).fillna(0) + 1e-4)
sel = lambda d: set(map(tuple, expected_f_fast(exclusive(pd.read_parquet(f"{d}/pred_s2_France.parquet")))[["a", "b"]].to_numpy()))
v14, v16, v17 = sel("backup_pred_s2_v14"), sel("backup_pred_s2_v16"), sel("backup_pred_s2_output_v17c")
FILL_LIKE = {"cie", "france", "compagnie", "societe", "fils", "freres", "centre", "services", "groupe"}
def kinds(S):
    out = []
    for a, b in S:
        ta, tb = toks(r.name_clean.loc[a]), toks(r.name_clean.loc[b]); dx, dy = ta - tb, tb - ta
        if len(dx) == 1 and len(dy) == 1 and ta & tb:
            u, v = next(iter(dx)), next(iter(dy))
            if sdf.get(u, 0) >= 1e-3 and sdf.get(v, 0) >= 1e-3 and ratio.get(v, 1) < 2 and not abbrev(u, v) and fuzz.ratio(u, v) < 70:
                out.append("->cie/france/filler-like" if v in FILL_LIKE else ("filler-like->cat" if u in FILL_LIKE else "category->category"))
    return pd.Series(out).value_counts().to_dict()
for nm, S in (("v16 added vs v14 (LB +)", v16 - v14), ("v16 removed vs v14", v14 - v16), ("v17c added vs v16 (LB -)", v17 - v16), ("v17c removed vs v16", v16 - v17), ("ALL v16 links", v16)):
    print(f"{nm:26s}: {kinds(S)}")
