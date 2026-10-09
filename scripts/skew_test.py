import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "src")
def bctx(d, col):
    gb = d.groupby("b")[col]
    d["rank_in_b"] = gb.rank(ascending=False, method="first")
    d["b_max"] = gb.transform("max")
    s = d.sort_values(col, ascending=False).groupby("b")[col].nth(1)
    d["b_second"] = d.b.map(pd.Series(s.values, index=d.loc[s.index, "b"].values)).fillna(0)
    d["b_margin"] = np.where(d.rank_in_b == 1, d[col] - d.b_second, d[col] - d.b_max)
    return d
def summ(d, nm):
    print(f"{nm:14s} band n={len(d):8d} | rank_in_b>=2 {np.mean(d.rank_in_b >= 2):.1%} | b_second>0.05 {np.mean(d.b_second > .05):.1%} "
          f"| b_second>0.5 {np.mean(d.b_second > .5):.1%} | b_margin<0 {np.mean(d.b_margin < 0):.1%}", flush=True)
print(pq.ParquetFile("artifacts/test/pred_hyb_US.parquet").schema.names)
for c in ("US", "India", "France"):
    d = pd.read_parquet(f"artifacts/test/pred_hyb_{c}.parquet", columns=["a", "b", "p"])
    d = bctx(d, "p")
    summ(d[(d.p > 0.005) & (d.p < 0.995)], f"test {c}")
    del d
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "country"])
for c in ("US", "India"):
    d = bctx(oof[oof.country == c].copy(), "p")
    summ(d[(d.p > 0.005) & (d.p < 0.995)], f"train {c}")
