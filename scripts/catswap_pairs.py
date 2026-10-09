import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
def toks(s): return [t for t in s.split() if t not in LEGAL]
Q = pd.read_parquet("/tmp/fr_catswap.parquet")
rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == "France")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["business_name", "business_address", "name_clean", "src"], filters=[("country", "==", "France")]); r.index = rows
A = r.name_clean.astype(object).loc[Q.a].tolist(); B = r.name_clean.astype(object).loc[Q.b].tolist()
pairs = []
for x, y in zip(A, B):
    tx, ty = set(toks(x)), set(toks(y)); pairs.append(f"{next(iter(tx - ty))} -> {next(iter(ty - tx))}")
Q = Q.assign(sw=pairs)
g = Q.groupby("sw").p.agg(["size", "median", lambda s: (s > .5).mean()]).sort_values("size", ascending=False)
g.columns = ["n", "median_p", "share_p>0.5"]
print(g.head(30).round(3).to_string())
print("\nexamples:")
for _, x in Q.sample(10, random_state=0).iterrows():
    a, b = r.loc[int(x.a)], r.loc[int(x.b)]
    print(f"p={x.p:.3f} S1 {a.business_name} | {a.business_address}\n        S{b.src} {b.business_name} | {b.business_address}")
