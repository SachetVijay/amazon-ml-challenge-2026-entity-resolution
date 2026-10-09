import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == "France")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "src", "business_name", "business_address", "name_clean", "addr_clean", "addr_nums"], filters=[("country", "==", "France")]); r.index = rows
s1 = r[r.src == 1]
df1 = pd.Series([t for n in s1.name_clean.astype(object) for t in {x for x in n.split() if x not in LEGAL}]).value_counts() / len(s1)
dfo = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in {x for x in n.split() if x not in LEGAL}]).value_counts() / (r.src != 1).sum()
ratio = (dfo.reindex(df1.index.union(dfo.index)).fillna(0) + 1e-4) / (df1.reindex(df1.index.union(dfo.index)).fillna(0) + 1e-4)
P = pd.read_parquet("artifacts/test/pred_s2_France.parquet")
from er.decide import exclusive, expected_f_fast
ex = exclusive(P); sel = expected_f_fast(ex)
S = sel.merge(ex[["a", "b"]], on=["a", "b"])
A = r.name_clean.astype(object).loc[S.a].tolist(); B = r.name_clean.astype(object).loc[S.b].tolist()
na = r.addr_nums.astype(object).loc[S.a].tolist(); nb = r.addr_nums.astype(object).loc[S.b].tolist()
kind, wp = [], []
for x, y, u1, u2 in zip(A, B, na, nb):
    ta = {t for t in x.split() if t not in LEGAL}; tb = {t for t in y.split() if t not in LEGAL}
    dx, dy = ta - tb, tb - ta
    if len(dx) == 1 and len(dy) == 1 and ta & tb:
        u, v = next(iter(dx)), next(iter(dy))
        if df1.get(u, 0) >= 0.001 and df1.get(v, 0) >= 0.001 and ratio.get(v, 1) < 2:
            kind.append(True); wp.append(f"{u}->{v}"); continue
    kind.append(False); wp.append("")
S["swap"], S["wp"] = kind, wp
W = S[S.swap]
print(f"v13 France selected links {len(S)}, swap->native {len(W)} ({len(W)/len(S):.2%}); p quantiles {W.p.quantile([.1,.25,.5,.75]).round(3).tolist()}")
print(W.groupby("wp").p.agg(["size", "median"]).sort_values("size", ascending=False).head(20).round(3).to_string())
# does the record's exact clean name exist as ANOTHER S1 in France?
name_to_s1 = s1.groupby("name_clean").size()
W = W.assign(rec_name_exists=[name_to_s1.get(n, 0) > 0 for n in r.name_clean.astype(object).loc[W.b]])
print(f"\nrecord's exact clean name also exists as another S1: {W.rec_name_exists.mean():.1%}")
for _, x in W.sample(14, random_state=5).iterrows():
    a, b = r.loc[int(x.a)], r.loc[int(x.b)]
    print(f"p={x.p:.3f} S1 {a.business_name} | {a.business_address}\n        S{b.src} {b.business_name} | {b.business_address}")
