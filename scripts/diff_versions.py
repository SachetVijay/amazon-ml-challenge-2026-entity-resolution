import sys
import pandas as pd
a, b = sys.argv[1], sys.argv[2]
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
country = dict(zip(c.entity_id, c.country))
def load(p):
    m = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    return {s: set(x for x in l.split(",") if x) for s, l in zip(m.source1_entity_id, m.matched_entity_ids)}
A, B = load(a), load(b)
stats = {}
for s in A:
    k = country[s]
    d = stats.setdefault(k, [0, 0, 0, 0])
    add, rem = B[s] - A[s], A[s] - B[s]
    d[0] += len(add); d[1] += len(rem); d[2] += bool(add or rem); d[3] += 1
for k, (ad, rm, ch, n) in stats.items():
    print(f"{k:6s}: links added {ad:6d}, removed {rm:6d}, entities changed {ch:6d} ({ch / n:.2%} of {n})")
