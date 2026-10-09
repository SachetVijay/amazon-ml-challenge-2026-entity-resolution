import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
art = "artifacts/test"; c = "France"
idx = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "business_name", "business_address", "name_core", "addr_clean", "addr_first_num"], filters=[("country", "==", c)]); r.index = idx
cand = pd.read_parquet(f"{art}/pred_hyb_{c}.cand.parquet", columns=["a", "b"])
cand = cand[r.addr_clean.astype(object).loc[cand.b].to_numpy() != ""]
A, B = cand.a.to_numpy(), cand.b.to_numpy()
nm = cpdist(r.name_core.astype(object).loc[A].tolist(), r.name_core.astype(object).loc[B].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
strip = lambda s: " ".join(t for t in s.split() if not t.isdigit())
st = cpdist([strip(x) for x in r.addr_clean.astype(object).loc[A]], [strip(x) for x in r.addr_clean.astype(object).loc[B]], scorer=fuzz.token_set_ratio, workers=-1)
na, nb = r.addr_first_num.astype(object).loc[A].to_numpy(), r.addr_first_num.astype(object).loc[B].to_numpy()
samenum = (na == nb) & (nb != "")
X = cand[(nm >= 90) & (st >= 80) & ~samenum & (nb != "")]
Y = cand[(nm < 50) & (st >= 90) & samenum]
both = np.intersect1d(X.b.unique(), Y.b.unique())
P = pd.read_parquet(f"{art}/pred_s2_{c}.parquet").set_index(["a", "b"]).p
Xb = X[X.b.isin(both)].copy(); Xb["p"] = P.reindex(pd.MultiIndex.from_arrays([Xb.a, Xb.b])).fillna(0).to_numpy()
Yb = Y[Y.b.isin(both)].copy(); Yb["p"] = P.reindex(pd.MultiIndex.from_arrays([Yb.a, Yb.b])).fillna(0).to_numpy()
bx = Xb.groupby("b").p.max()
print(f"conflict records {len(both)}; name-match candidates per record {len(Xb)/len(both):.2f}; best name-match p quantiles "
      f"{bx.quantile([.1,.25,.5,.75,.9]).round(3).tolist()}; share with p1<=1e-3 (dropped before stage 2) {np.mean(bx==0):.1%}")
print(f"address-match Y p quantiles {Yb.groupby('b').p.max().quantile([.5,.9]).round(3).tolist()}")
for b in np.random.default_rng(7).choice(both, 8, replace=False):
    x = r.loc[b]; print(f"\nREC S{x.src} {x.business_name} | {x.business_address}")
    for _, y in Xb[Xb.b == b].iterrows(): s = r.loc[int(y.a)]; print(f"   X p={y.p:.3f} {s.business_name} | {s.business_address}")
    for _, y in Yb[Yb.b == b].iterrows(): s = r.loc[int(y.a)]; print(f"   Y p={y.p:.3f} {s.business_name} | {s.business_address}")
