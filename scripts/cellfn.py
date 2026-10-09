"""Cell-wise transfer check. Cell = name relation x number relation x street agreement (+address-less).
Validation (US+India labels): truth rate per cell. Test: pairs/S1 and accepted/S1 per cell per country.
Implied France shortfall = pairs/S1 * (truth - accepted rate) per cell (assumes per-cell truth transfers)."""
import re, sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from street_mix import street
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def ratios(r):
    s = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
    o = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
    return (o + 1e-4) / (s.reindex(o.index).fillna(0) + 1e-4)
def cells(split, c, P):
    art = f"artifacts/{split}"
    rows = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["src", "name_clean", "business_address", "addr_clean"], filters=[("country", "==", c)]); r.index = rows
    ra = ratios(r)
    A, B = r.name_clean.astype(object).loc[P.a].tolist(), r.name_clean.astype(object).loc[P.b].tolist()
    nm = []
    for x, y in zip(A, B):
        ta, tb = toks(x), toks(y); add, drop = tb - ta, ta - tb
        if not ta & tb: nm.append("renamed"); continue
        if not add and not drop: nm.append("same"); continue
        if add and not drop: nm.append("adds:" + ("qual" if max(ra.get(t, 1) for t in add) >= 20 else ("fill" if min(ra.get(t, 1) for t in add) >= 2 else "word"))); continue
        if drop and not add: nm.append("drops"); continue
        nm.append("swap1" if len(add) == 1 and len(drop) == 1 else "multi")
    sa = [street(x) for x in r.business_address.astype(object).loc[P.a]]; sb = [street(x) for x in r.business_address.astype(object).loc[P.b]]
    st = cpdist([x[1] for x in sa], [x[1] for x in sb], scorer=fuzz.ratio, workers=-1)
    ab = r.addr_clean.astype(object).loc[P.b].to_numpy()
    out = []
    for n, (xa, _), (xb, _), s, e in zip(nm, sa, sb, st, ab):
        if e == "": out.append(f"{n} | no-address"); continue
        num = "no-num" if not xb or not xa else ("num=" if xa == xb else ("num-drop" if (xa.endswith(xb) or xa.startswith(xb) or xb.endswith(xa) or xb.startswith(xa)) else "num≠"))
        out.append(f"{n} | {num} | {'st=' if s >= 85 else 'st≠'}")
    return np.array(out), (r.src == 1).sum()
