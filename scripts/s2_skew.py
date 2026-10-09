"""Stage 2 was also trained on thin (bucket-0) competition. Outside the band its p2 is final. How often does it
overturn a confident stage-1 match (p1 >= 0.995) at test time, and do those pairs have strong rivals?"""
import numpy as np, pandas as pd
for c in ("US", "India", "France"):
    p1 = pd.read_parquet(f"artifacts/test/pred_hyb_{c}.parquet", columns=["a", "b", "p"])
    p1 = p1[p1.p > 1e-3]
    s2 = pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet", columns=["a", "b", "p"]).rename(columns={"p": "pf"})
    d = p1.merge(s2, on=["a", "b"])
    gb = d.groupby("b").p
    d["n_conf_b"] = gb.transform(lambda x: (x >= 0.995).sum())
    hi = d[d.p >= 0.995]
    low = hi[hi.pf < 0.5]
    print(f"{c:6s}: p1>=0.995 pairs {len(hi):8d}; stage 2 drops {len(low):6d} below 0.5 ({len(low)/len(hi):.2%}); "
          f"of those, record has another p1>=0.995 S1: {np.mean(low.n_conf_b >= 2):.1%} "
          f"(vs {np.mean(hi.n_conf_b >= 2):.1%} of all confident pairs)")
