"""v24 (= v17a probabilities + v22 demotion) vs v22: what changes, by swap type / cell."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from cellfn import cells
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
fr = set(c.entity_id[c.country == "France"])
def load(p):
    m = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False); m = m[m.source1_entity_id.isin(fr)]
    return {(s, o) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) for o in l.split(",") if o}
a, b = load("output_v22.tsv"), load("/tmp/v24.tsv")
rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == "France")
r = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id"], filters=[("country", "==", "France")]); r.index = rows
idx = pd.Series(r.index, index=r.entity_id.astype(object))
for nm, S in (("ADDED by v24", b - a), ("REMOVED by v24", a - b)):
    P = pd.DataFrame([(idx[s], idx[o]) for s, o in S], columns=["a", "b"])
    k, _ = cells("test", "France", P)
    print(f"{nm}: {len(S)} links; top cells {pd.Series(k).value_counts().head(8).to_dict()}", flush=True)
