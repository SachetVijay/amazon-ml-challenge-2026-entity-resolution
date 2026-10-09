import sys
import numpy as np, pandas as pd
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
fr = set(c.entity_id[c.country == "France"])
for p in sys.argv[1:]:
    m = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False); m = m[m.source1_entity_id.isin(fr)]
    n = np.array([len(l.split(",")) if l else 0 for l in m.matched_entity_ids])
    print(f"{p:40s} France mean links {n.mean():.4f} empty {np.mean(n==0):.4f} one {np.mean(n==1):.4f} total {n.sum()}")
