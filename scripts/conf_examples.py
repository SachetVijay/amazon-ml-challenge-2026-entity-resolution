import numpy as np, pandas as pd
S = pd.read_parquet("/tmp/conf_scored.parquet")
art = "artifacts/test"
rows = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
for c in ("France", "US"):
    idx = np.flatnonzero(rows == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["src", "business_name", "business_address"], filters=[("country", "==", c)]); r.index = idx
    g = S[(S.country == c) & (S.ce < -2)]
    print(f"===== {c}: {len(g)} strong disagreements")
    for _, x in g.sample(min(12, len(g)), random_state=0).iterrows():
        a, b = r.loc[int(x.a)], r.loc[int(x.b)]
        print(f"ce={x.ce:.1f} p1={x.p:.4f} pf={x.pf:.3f}\n   S1 {a.business_name} | {a.business_address}\n   S{b.src} {b.business_name} | {b.business_address}")
