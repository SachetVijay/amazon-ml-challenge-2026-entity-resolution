import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from er.predict import country_rows, read_country
art, country = "artifacts/test", sys.argv[1]
rows = country_rows(f"{art}/records.parquet")[country]
sub = read_country(f"{art}/records.parquet", country, rows)
P = pd.read_parquet(f"{art}/pred_s2_{country}.parquet")
best = P.loc[P.groupby("b").p.idxmax()]
print(country, "exclusive winners p-distribution:", np.histogram(best.p, [0, .05, .2, .5, .8, .95, 1])[0])
s1n = (sub.src.to_numpy() == 1).sum()
print("S2/S3 records whose best p>0.5 per S1:", round((best.p > .5).sum() / s1n, 3))
pd.set_option("display.width", 250)
unc = best[(best.p > 0.2) & (best.p < 0.8)].sample(14, random_state=1)
for r in unc.itertuples():
    A, B = sub.loc[r.a], sub.loc[r.b]
    print(f"\n p={r.p:.2f}  S1: {A.business_name} | {A.business_address}\n         rec: {B.business_name} | {B.business_address}")
