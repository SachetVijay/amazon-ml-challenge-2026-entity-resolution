import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from er.predict import country_rows
art = "artifacts/test"
src = pd.read_parquet(f"{art}/records.parquet", columns=["src"]).src.to_numpy()
for country, rows in country_rows(f"{art}/records.parquet").items():
    n_s1 = (src[rows] == 1).sum()
    P = pd.read_parquet(f"{art}/pred_s2_{country}.parquet")
    best = P.loc[P.groupby("b").p.idxmax()]
    h = np.histogram(best.p, [0, .05, .2, .5, .8, .95, 1.0001])[0] / n_s1
    print(f"{country:7s} per S1: winners by p-bin [0-.05 .05-.2 .2-.5 .5-.8 .8-.95 .95-1] = {np.round(h, 3)}  "
          f"expected #matches (sum p of winners) = {best.p.sum() / n_s1:.3f}")
