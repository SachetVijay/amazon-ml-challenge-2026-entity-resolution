import numpy as np, pandas as pd, sys
sys.path.insert(0, "src")
from er.predict import country_rows
bins = [0, .05, .2, .5, .8, .95, 1.0001]
O = pd.read_parquet("artifacts/train/oof_stage2_hyb5_ce_v4.parquet", columns=["a", "b", "country", "p2", "y"])
for c, g in O.groupby("country"):
    n = g.a.nunique()
    best = g.loc[g.groupby("b").p2.idxmax()]
    h = np.histogram(best.p2, bins)[0] / n
    hn = np.histogram(best.p2[~best.y.to_numpy()], bins)[0] / n
    print(f"OOF  {c:6s} winners/S1 by p-bin {np.round(h, 3)} | of which NON-matches {np.round(hn, 3)}")
src = pd.read_parquet("artifacts/test/records.parquet", columns=["src"]).src.to_numpy()
for c, rows in country_rows("artifacts/test/records.parquet").items():
    P = pd.read_parquet(f"artifacts/test/v4_backup/pred_s2_{c}.parquet")
    n = (src[rows] == 1).sum()
    best = P.loc[P.groupby("b").p.idxmax()]
    print(f"TEST {c:6s} winners/S1 by p-bin {np.round(np.histogram(best.p, bins)[0] / n, 3)}")
print("bins:", bins[:-1])
