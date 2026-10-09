import sys
import numpy as np, pandas as pd
c = sys.argv[1]; art = "artifacts/test"
m = pd.read_csv("output_v14/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
link = {o: s for s, l in zip(m.source1_entity_id, m.matched_entity_ids) for o in l.split(",") if o}
r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "business_name", "business_address", "name_core", "addr_first_num"], filters=[("country", "==", c)])
r = r.astype({k: object for k in r.columns if k != "src"})
s1 = r[r.src == 1]; S = s1.set_index("entity_id"); o = r[r.src != 1].copy()
o["X"] = o.entity_id.map(link); o["linked"] = o.X.notna(); o["key"] = o.name_core + "|" + o.addr_first_num
s1key = set(s1.name_core + "|" + s1.addr_first_num)
wa = o[o.addr_first_num != ""]; ulk = set(wa.key[~wa.linked])
B = wa[wa.linked & wa.key.isin(ulk) & ~wa.key.isin(s1key)]
print(f"(B) {c}: {len(B)}")
for _, x in B.sample(8, random_state=2).iterrows():
    s = S.loc[x.X]
    print(f"\nREC S{x.src} {x.business_name} | {x.business_address}\n  linked S1: {s.business_name} | {s.business_address}")
    for _, y in wa[(wa.key == x.key) & ~wa.linked].head(2).iterrows(): print(f"  unlinked twin S{y.src}: {y.business_name} | {y.business_address}")
