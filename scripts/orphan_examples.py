import sys
import numpy as np, pandas as pd
c = sys.argv[1]; art = "artifacts/test"
m = pd.read_csv("output_v14/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
link = {o: s for s, l in zip(m.source1_entity_id, m.matched_entity_ids) for o in l.split(",") if o}
r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "business_name", "business_address", "name_core", "addr_clean", "addr_first_num"], filters=[("country", "==", c)])
r = r.astype({k: object for k in r.columns if k != "src"})
s1 = r[r.src == 1]; S = s1.set_index("entity_id")
o = r[r.src != 1].copy(); o["X"] = o.entity_id.map(link); o["linked"] = o.X.notna()
o["key"] = o.name_core + "|" + o.addr_first_num
s1key = set(s1.name_core + "|" + s1.addr_first_num)
ul = o[~o.linked & (o.addr_first_num != "")]
g = ul.groupby("key").size(); orphan_keys = set(g[g >= 2].index) - s1key
okn = pd.Series(list(orphan_keys)); okn = okn.groupby(okn.str.split("|").str[0]).apply(list)
kvis = s1.groupby("name_core").size()
A = o[o.linked & (o.addr_clean == "")].copy()
A["k_vis"] = A.name_core.map(kvis).fillna(0); A = A[(A.k_vis == 1) & A.name_core.isin(okn.index)]
P = pd.read_parquet(f"{art}/pred_s2_{c}.parquet"); ids = r.entity_id.to_numpy()
rows = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
pos = dict(zip(ids, rows))
pp = P.set_index(["a", "b"]).p
A["p"] = [pp.get((pos[x], pos[e]), np.nan) for x, e in zip(A.X, A.entity_id)]
print(f"(A) {c}: {len(A)} links; p quantiles {A.p.quantile([.1,.25,.5,.75]).round(3).tolist()}")
for _, x in A.sample(6, random_state=1).iterrows():
    s = S.loc[x.X]
    print(f"\n p={x.p:.3f} REC S{x.src} {x.business_name} | (no address)\n   linked S1: {s.business_name} | {s.business_address}")
    for k in okn[x.name_core][:2]:
        grp = ul[ul.key == k]
        for _, y in grp.head(3).iterrows(): print(f"   orphan grp S{y.src}: {y.business_name} | {y.business_address}")
