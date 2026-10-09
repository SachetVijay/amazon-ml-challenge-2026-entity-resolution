"""Profile each appended suffix word: pure append (nothing dropped) share, same-address share, extra-record share.
Train (labelled) vs France test (v16 links) -> which French words behave like train's distractor suffixes."""
import sys
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
def prof(na, nb):
    ta = set(na.split()); w = nb.split()
    if not w or w[-1] in ta or w[-1] in LEGAL or len(w[-1]) < 2: return None
    rest = set(w[:-1]); drop = {t for t in ta - rest if t not in LEGAL and len(t) > 1}
    return w[-1], not drop
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "s1_id", "o_id", "country"]); oof = oof[oof.p > 0.05]
r = pd.read_parquet("artifacts/train/records.parquet", columns=["name_clean", "addr_clean"])
ys = set(map(tuple, load_gt("student_resource/dataset/train/train_ground_truth.tsv")[["s1", "other"]].to_numpy()))
nm, ad = r.name_clean.astype(object).to_numpy(), r.addr_clean.astype(object).to_numpy()
W = ["center", "services", "service", "partners", "group", "enterprises", "holdings"]
st = defaultdict(list)
for a, b, s, o, p in zip(oof.a, oof.b, oof.s1_id, oof.o_id, oof.p):
    x = prof(nm[a], nm[b])
    if x and x[0] in W: st[x[0]].append(((s, o) in ys, x[1], ad[a] == ad[b] and ad[a] != "", p))
print("TRAIN word: n true | pure-append | same-addr | mean p1 | true rate among pure / among drop+append")
for w in W:
    A = np.array(st[w], float)
    pu = A[:, 1] == 1
    print(f"  {w:12s} {len(A):6d} {A[:,0].mean():.2f} | {pu.mean():.2f} | {A[:,2].mean():.2f} | {A[:,3].mean():.2f} | "
          f"{A[pu,0].mean() if pu.any() else float('nan'):.2f} / {A[~pu,0].mean() if (~pu).any() else float('nan'):.2f}")
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "addr_clean", "country", "src"])
F = rt[rt.country == "France"]; idx = pd.Series(F.index, index=F.entity_id.astype(object))
nm, ad = rt.name_clean.astype(object).to_numpy(), rt.addr_clean.astype(object).to_numpy()
ps = pd.read_parquet("artifacts/test/pred_s2_France.parquet", columns=["a", "b", "p"]).set_index(["a", "b"]).p
m = pd.read_csv("output_v16/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
s1 = set(F.entity_id[F.src == 1].astype(object))
WF = ["fils", "associes", "services", "cie", "developpement", "groupe", "france"]
st = defaultdict(list)
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    if s not in s1 or not l: continue
    a = idx[s]
    for o in l.split(","):
        b = idx[o]; x = prof(nm[a], nm[b])
        if x and x[0] in WF: st[x[0]].append((x[1], ad[a] == ad[b] and ad[a] != "", ps.get((a, b), np.nan)))
print("TEST France word: n | pure-append | same-addr | mean final p")
for w in WF:
    A = np.array(st[w], float)
    print(f"  {w:14s} {len(A):6d} | {A[:,0].mean():.2f} | {A[:,1].mean():.2f} | {np.nanmean(A[:,2]):.2f}")
