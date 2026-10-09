"""France after v13: uncertain exclusive winners, with the record's competing S1s. What kind of ambiguity is it?"""
import numpy as np, pandas as pd
P = pd.read_parquet("artifacts/test/pred_s2_France.parquet")                  # a b p  (v13 final probabilities)
cols = ["entity_id", "src", "business_name", "business_address", "name_core", "name_clean", "addr_clean", "addr_nums"]
r = pd.read_parquet("artifacts/test/records.parquet", columns=cols, filters=[("country", "==", "France")])
rows = pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).reset_index(drop=True)
r.index = np.flatnonzero(rows.country.to_numpy() == "France")
win = P.loc[P.groupby("b").p.idxmax()]
unc = win[(win.p > 0.2) & (win.p < 0.8)]
print(f"France: {len(win)} records with a candidate, uncertain winners (0.2<p<0.8): {len(unc)} ({len(unc)/r[r.src==1].shape[0]:.3f} per S1)")
B = r.loc[unc.b]
print(f"  of which record address-less: {np.mean(B.addr_clean.to_numpy()==''):.1%}")
cnt = r[r.src == 1].groupby("name_core").size()
k = cnt.reindex(B.name_core).fillna(0).to_numpy()
print(f"  record's core name held by >=2 S1: {np.mean(k>=2):.1%}")
nc_cnt = r[r.src == 1].groupby("name_clean").size()
print(f"  record's full clean name held by >=2 S1: {np.mean(nc_cnt.reindex(B.name_clean).fillna(0).to_numpy()>=2):.1%}")
rng = np.random.default_rng(3)
for b in rng.choice(unc.b.to_numpy(), 12, replace=False):
    x = r.loc[b]
    c = P[P.b == b].sort_values("p", ascending=False).head(4)
    print(f"\nREC S{x.src} | {x.business_name} | {x.business_address}")
    for _, y in c.iterrows():
        s = r.loc[int(y.a)]
        print(f"   p={y.p:.2f} S1 | {s.business_name} | {s.business_address}")
