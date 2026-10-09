import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from struct_mix_lib import ncat, acat
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "business_name", "addr_clean", "country", "src"])
nm, ad, ct, raw = rt.name_clean.astype(object).fillna("").to_numpy(), rt.addr_clean.astype(object).fillna("").to_numpy(), rt.country.to_numpy(), rt.business_name.astype(object).to_numpy()
idx = pd.Series(rt.index, index=rt.entity_id.astype(object))
m = pd.read_csv("output_v22.tsv", sep="\t", dtype=str, keep_default_na=False).sample(frac=0.2, random_state=1)
ex = {"France": [], "US": []}; per = {}
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    a = idx[s]; c = ct[a]
    if c not in ex or not l: continue
    os_ = l.split(",")
    for o in os_:
        b = idx[o]
        if ncat(nm[a], nm[b]) == "disjoint" and acat(ad[a], ad[b]) == "same":
            ex[c].append((raw[a], raw[b], len(os_)))
for c in ex:
    print(c, len(ex[c]))
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ex[c]), 18, replace=False): print("   ", ex[c][i])
