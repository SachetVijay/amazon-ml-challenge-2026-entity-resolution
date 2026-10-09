"""Train: position of the appended noise word in TRUE drop+append pairs (does noise ever put it mid-name?).
Test: v22 French links where france/cie replaces a word strictly mid-name (not last)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
def core(n): return [t for t in n.split() if t not in LEGAL and len(t) > 1]
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "s1_id", "o_id"]); oof = oof[oof.p > 0.05]
r = pd.read_parquet("artifacts/train/records.parquet", columns=["name_clean"]); nm = r.name_clean.astype(object).to_numpy()
ys = set(map(tuple, load_gt("student_resource/dataset/train/train_ground_truth.tsv")[["s1", "other"]].to_numpy()))
last = mid = midT = 0
for a, b, s, o in zip(oof.a, oof.b, oof.s1_id, oof.o_id):
    ca, cb = core(nm[a]), core(nm[b])
    A = [t for t in cb if t not in ca]
    if len(A) == 1 and A[0] in ("center", "services", "service", "partners") and len(set(ca) - set(cb)) == 1:
        y = (s, o) in ys
        if cb[-1] == A[0]: last += 1
        else: mid += 1; midT += y
print(f"TRAIN drop+noise-word: word last {last}, word mid-name {mid} (true {midT})")
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "country", "src"])
F = rt[rt.country == "France"]; idx = pd.Series(F.index, index=F.entity_id.astype(object)); nm = rt.name_clean.astype(object).to_numpy()
s1 = set(F.entity_id[F.src == 1].astype(object))
m = pd.read_csv("output_v22.tsv", sep="\t", dtype=str, keep_default_na=False)
cnt = {"france": [0, 0], "cie": [0, 0], "fils": [0, 0], "services": [0, 0]}
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    if s not in s1 or not l: continue
    ca = core(nm[idx[s]])
    for o in l.split(","):
        cb = core(nm[idx[o]]); A = [t for t in cb if t not in ca]
        if len(A) == 1 and A[0] in cnt and len(set(ca) - set(cb)) == 1:
            cnt[A[0]][cb[-1] != A[0]] += 1
print("TEST France v22 links drop+word  [last, mid-name]:", cnt)
