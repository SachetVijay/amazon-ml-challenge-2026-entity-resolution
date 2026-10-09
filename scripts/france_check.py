import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.predict import country_rows
src = pd.read_parquet("artifacts/test/records.parquet", columns=["src"]).src.to_numpy()
rows = country_rows("artifacts/test/records.parquet")
for c in ("France", "US"):
    n = (src[rows[c]] == 1).sum()
    for tag, f in (("v6", f"artifacts/test/v6_backup/pred_hyb_{c}.parquet"), ("v7", f"artifacts/test/pred_hyb_{c}.parquet")):
        try:
            P = pd.read_parquet(f, columns=["b", "p"])
        except Exception as e:
            print(c, tag, "n/a"); continue
        best = P.groupby("b").p.max()
        mid = ((best > 0.2) & (best < 0.95)).sum() / n
        print(f"{c:6s} {tag} stage-1 winners/S1: p>0.99 {(best > .99).sum() / n:.3f} | p>0.5 {(best > .5).sum() / n:.3f} | uncertain(0.2-0.95) {mid:.3f}")
        del P, best
