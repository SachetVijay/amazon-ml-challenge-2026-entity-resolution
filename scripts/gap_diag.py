import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast
from er.predict import country_rows
# 1) OOF: predicted matches per S1 vs true matches per S1 (same decision rule as test)
D = pd.read_parquet("artifacts/train/oof_stage2_hyb5_v6.parquet")
ex = exclusive(D.rename(columns={"p2": "p"})[["a", "b", "p"]])
sel = expected_f_fast(ex)
n_s1 = D.a.nunique()
print(f"OOF: predicted {len(sel) / n_s1:.3f}/S1 (among S1 with candidates), true {D.y.sum() / n_s1:.3f}/S1, "
      f"true positives chosen {D.set_index(['a','b']).loc[list(zip(sel.a, sel.b))].y.sum() / n_s1:.3f}/S1")
best = D.loc[D.groupby("b").p2.idxmax()]
print("OOF winners/S1 with p>0.5:", round((best.p2 > .5).sum() / n_s1, 3), " p>0.99:", round((best.p2 > .99).sum() / n_s1, 3))
# 2) TEST, same statistics per country
src = pd.read_parquet("artifacts/test/records.parquet", columns=["src"]).src.to_numpy()
for c, rows in country_rows("artifacts/test/records.parquet").items():
    P = pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet")
    n = (src[rows] == 1).sum()
    b = P.loc[P.groupby("b").p.idxmax()]
    s = expected_f_fast(exclusive(P))
    print(f"TEST {c:6s}: predicted {len(s) / n:.3f}/S1 | winners p>0.5 {(b.p > .5).sum() / n:.3f} | p>0.99 {(b.p > .99).sum() / n:.3f}")
