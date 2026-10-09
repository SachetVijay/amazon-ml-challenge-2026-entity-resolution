"""Pattern mix of accepted links per country (test v13) and pattern precision on validation (v12 OOF decisions)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.decide import exclusive, expected_f_fast
def pat(A, B):
    out = []
    for (na, aa, ma), (nb, ab, mb) in zip(A, B):
        ta = {x for x in na.split() if x not in LEGAL}; tb = {x for x in nb.split() if x not in LEGAL}
        if ab == "": out.append("address-less record"); continue
        same = ma.split()[:1] == mb.split()[:1] and mb != ""
        if not (ta & tb): out.append("renamed/coined" + (" | same no." if same else " | diff no.")); continue
        if ta == tb: out.append("same name words" + (" | same no." if same else " | diff no.")); continue
        out.append("name differs" + (" | same no." if same else " | diff no."))
    return out
def recs(split, c):
    rows = np.flatnonzero(pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["entity_id", "name_clean", "addr_clean", "addr_nums"], filters=[("country", "==", c)])
    r.index = rows
    return r
def triples(r, idx): return list(zip(r.name_clean.astype(object).loc[idx], r.addr_clean.astype(object).loc[idx], r.addr_nums.astype(object).loc[idx]))
cols = {}
for c in ("US", "India", "France"):
    P = pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet"); ex = exclusive(P); S = expected_f_fast(ex)
    r = recs("test", c)
    cols[f"test {c}"] = pd.Series(pat(triples(r, S.a), triples(r, S.b))).value_counts(normalize=True)
T = pd.DataFrame(cols).fillna(0) * 100
from er.prep import load_gt
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a")
full = full.merge(oof, on="a")
for c in ("US", "India"):
    F = full[full.country == c]; ex = exclusive(F[["a", "b", "s1_id", "o_id", "p"]]); S = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    r = recs("train", c)
    S["pat"] = pat(triples(r, S.a), triples(r, S.b))
    S["y"] = [(s, o) in ys for s, o in zip(S.s1_id, S.o_id)]
    g = S.groupby("pat").y.agg(["mean", "size"])
    T[f"val {c} share"] = g["size"] / g["size"].sum() * 100
    T[f"val {c} precision"] = g["mean"] * 100
print(T.round(2).to_string())
