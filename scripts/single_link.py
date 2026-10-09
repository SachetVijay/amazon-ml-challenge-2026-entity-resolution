"""S1 entities with exactly ONE predicted link: link pattern mix (test per country) and precision per pattern (validation)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from street_mix import classify
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
res = {}
for c in ("US", "France"):
    rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet("artifacts/test/records.parquet", columns=["business_address", "name_core", "addr_clean"], filters=[("country", "==", c)]); r.index = rows
    P = pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet"); S = expected_f_fast(exclusive(P))
    S1 = S[S.groupby("a").b.transform("size") == 1]
    k = pd.Series(classify(r, S1.a, S1.b))
    res[f"test {c} %"] = k.value_counts(normalize=True) * 100
    res[f"test {c} p<0.9 %"] = pd.Series(np.array(classify(r, S1.a, S1.b))[S1.p.to_numpy() < 0.9]).value_counts() / len(S1) * 100
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a"); full = full.merge(oof, on="a")
F = full[full.country == "US"]; ex = exclusive(F[["a", "b", "s1_id", "o_id", "p"]]); S = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
S1 = S[S.groupby("a").b.transform("size") == 1]
rows = np.flatnonzero(pd.read_parquet("artifacts/train/records.parquet", columns=["country"]).country.to_numpy() == "US")
r = pd.read_parquet("artifacts/train/records.parquet", columns=["business_address", "name_core", "addr_clean"], filters=[("country", "==", "US")]); r.index = rows
S1 = S1.assign(k=classify(r, S1.a, S1.b), y=[(s, o) in ys for s, o in zip(S1.s1_id, S1.o_id)])
g = S1.groupby("k").y.agg(["size", "mean"]); res["val US %"] = g["size"] / g["size"].sum() * 100; res["val US prec"] = g["mean"] * 100
print((pd.DataFrame(res).fillna(0).sort_values("test France %", ascending=False)).round(2).head(12).to_string())
print(f"1-link S1 share: val US {len(S1)/F.s1_id.nunique():.4f}")
