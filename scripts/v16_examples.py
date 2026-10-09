import numpy as np, pandas as pd
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
fr = set(c.entity_id[c.country == "France"])
def load(p):
    m = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False); m = m[m.source1_entity_id.isin(fr)]
    return {(s, o) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) for o in l.split(",") if o}
a, b = load("output_v14/matching_results.tsv"), load("output_v16/matching_results.tsv")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "business_name", "business_address"], filters=[("country", "==", "France")]).set_index("entity_id")
rng = np.random.default_rng(11)
for nm, S in (("ADDED by v16", list(b - a)), ("REMOVED by v16", list(a - b))):
    print(f"===== {nm}: {len(S)}")
    for i in rng.choice(len(S), 9, replace=False):
        s, o = S[i]; x, y = r.loc[s], r.loc[o]
        print(f"  S1 {x.business_name} | {x.business_address}\n  {o[:2]} {y.business_name} | {y.business_address}")
