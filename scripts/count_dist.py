import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.prep import load_gt
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "country", "src"])
cty = pd.Series(rt.country.to_numpy(), index=rt.entity_id.astype(object))
m = pd.read_csv("output_v22.tsv", sep="\t", dtype=str, keep_default_na=False)
m["n"] = [len(l.split(",")) if l else 0 for l in m.matched_entity_ids]; m["c"] = cty.reindex(m.source1_entity_id).to_numpy()
tab = pd.crosstab(m.c, m.n.clip(upper=8), normalize="index").round(4)
tab["mean"] = m.groupby("c").n.mean(); tab["S1"] = m.groupby("c").size()
# records per S1 in test (S2+S3 count per country / S1 count)
rec = rt[rt.src != 1].groupby("country").size(); tab["rec_per_S1"] = rec / tab.S1
print("TEST v22 predicted links per S1:"); print(tab.to_string())
tr = pd.read_parquet("artifacts/train/records.parquet", columns=["entity_id", "country", "src"])
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
s1 = tr[tr.src == 1]; k = gt.groupby("s1").size()
s1 = s1.assign(n=k.reindex(s1.entity_id.astype(object)).fillna(0).to_numpy())
t2 = pd.crosstab(s1.country, s1.n.clip(upper=8), normalize="index").round(4); t2["mean"] = s1.groupby("country").n.mean()
t2["rec_per_S1"] = tr[tr.src != 1].groupby("country").size() / s1.groupby("country").size()
print("TRAIN ground-truth links per S1:"); print(t2.to_string())
