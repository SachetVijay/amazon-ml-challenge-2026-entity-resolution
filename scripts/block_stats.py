import collections
import pandas as pd
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
country = dict(zip(c.entity_id, c.country))
n_other = collections.Counter()
for s in (2, 3):
    d = pd.read_csv(f"student_resource/dataset/test/test_source{s}.tsv", sep="\t", usecols=["country"], dtype=str)
    n_other.update(d.country.value_counts().to_dict())
tot, n1, sizes = collections.Counter(), collections.Counter(), collections.defaultdict(list)
with open("output_v9/candidate_pairs.tsv") as f:
    next(f)
    for line in f:
        s, lst = line.rstrip("\n").split("\t")
        k = len(lst.split(",")) if lst else 0
        tot[country[s]] += k; n1[country[s]] += 1; sizes[country[s]].append(k)
import numpy as np
allk = np.concatenate([np.array(v) for v in sizes.values()])
print(f"TOTAL: {allk.sum():,} candidate pairs | mean {allk.mean():.1f}/S1 | median {np.median(allk):.0f} | p95 {np.percentile(allk, 95):.0f} | max {allk.max()} | empty {(allk == 0).sum()}")
for k in n1:
    v = np.array(sizes[k]); full = n1[k] * n_other[k]
    print(f"{k:6s}: {tot[k]:,} pairs, mean {v.mean():.1f}/S1, median {np.median(v):.0f}, p95 {np.percentile(v, 95):.0f} | "
          f"all-pairs {full:,.0f} -> reduction ratio {1 - tot[k] / full:.6f}")
