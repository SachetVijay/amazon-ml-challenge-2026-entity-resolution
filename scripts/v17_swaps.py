import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.decide import exclusive, expected_f_fast
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == "France")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["src", "name_clean"], filters=[("country", "==", "France")]); r.index = rows
sdf = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
odf = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
ratio = (odf + 1e-4) / (sdf.reindex(odf.index).fillna(0) + 1e-4)
sel = lambda f: set(map(tuple, expected_f_fast(exclusive(pd.read_parquet(f)))[["a", "b"]].to_numpy()))
v14, v16, v17 = (sel(f"{d}/pred_s2_France.parquet") for d in ("backup_pred_s2_v14", "backup_pred_s2_v16", "backup_pred_s2_output_v17c"))
def swaps(S):
    out = []
    for a, b in S:
        ta, tb = toks(r.name_clean.loc[a]), toks(r.name_clean.loc[b]); dx, dy = ta - tb, tb - ta
        if len(dx) == 1 and len(dy) == 1 and ta & tb:
            u, v = next(iter(dx)), next(iter(dy))
            kind = ("rare/typo" if sdf.get(u, 0) < 1e-3 or sdf.get(v, 0) < 1e-3 else ("->filler" if ratio.get(v, 1) >= 2 else "native<->native"))
            out.append(kind)
    return pd.Series(out).value_counts(normalize=True).round(3).to_dict(), len(out)
for nm, S in (("v16 added vs v14 (LB +0.00117)", v16 - v14), ("v17c added vs v16", v17 - v16), ("v17c removed vs v16", v16 - v17)):
    d, n = swaps(S); print(f"{nm}: {len(S)} links, of which one-word swaps {n}: {d}")
