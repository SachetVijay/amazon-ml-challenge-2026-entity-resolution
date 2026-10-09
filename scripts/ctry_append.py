"""Is 'country name appended' a TRUE-copy noise style? Train: pairs where the other adds only the country token (no drop).
Test France (v16 links): same + position split of the v25 swap targets."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "s1_id", "o_id", "country"]); oof = oof[oof.p > 0.05]
r = pd.read_parquet("artifacts/train/records.parquet", columns=["name_clean", "business_name"])
ys = set(map(tuple, load_gt("student_resource/dataset/train/train_ground_truth.tsv")[["s1", "other"]].to_numpy()))
nm, raw = r.name_clean.astype(object).to_numpy(), r.business_name.astype(object).to_numpy()
for c, w in (("India", "india"), ("US", "usa")):
    P = oof[oof.country == c]; n = t = 0; ex = []
    for a, b, s, o in zip(P.a, P.b, P.s1_id, P.o_id):
        ta, tb = toks(nm[a]), toks(nm[b])
        if not (ta - tb) and (tb - ta) == {w}:
            n += 1; y = (s, o) in ys; t += y
            if len(ex) < 4: ex.append((y, raw[a], raw[b]))
    print(f"TRAIN {c}: pure '+{w}' pairs n={n} true={t}", ex)
# test France
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "business_name", "country", "src"])
F = rt[rt.country == "France"]; idx = pd.Series(F.index, index=F.entity_id.astype(object))
m = pd.read_csv("output_v16/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
s1 = set(F.entity_id[F.src == 1].astype(object)); nm, raw = rt.name_clean.astype(object).to_numpy(), rt.business_name.astype(object).to_numpy()
pure = swp = br = inpl = 0; ex = []
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    if s not in s1 or not l: continue
    a = idx[s]; ta = toks(nm[a])
    for o in l.split(","):
        b = idx[o]; tb = toks(nm[b]); D, A = ta - tb, tb - ta
        if not D and A == {"france"}: pure += 1
        if len(D) == 1 and A == {"france"}:
            swp += 1; bb = "(France)" in raw[b] or "[France]" in raw[b]; br += bb
            if len(ex) < 8: ex.append((raw[a], raw[b]))
print(f"TEST France v16 links: pure '+france' (true-copy suffix noise if common) = {pure}; drop+france = {swp} (bracketed {br})")
for e in ex: print("  ", e)
