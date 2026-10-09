"""Confident stage-1 pairs (p1 >= 0.995) never reach the cross-encoders. Score a sample of such LINKED pairs with the
e5-base CE and compare disagreement rates across countries."""
import sys
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.cross_encoder import score
art = "artifacts/test"
parts = []
for c in ("US", "India", "France"):
    p1 = pd.read_parquet(f"{art}/pred_hyb_{c}.parquet", columns=["a", "b", "p"])
    fin = pd.read_parquet(f"{art}/pred_s2_{c}.parquet").rename(columns={"p": "pf"})
    d = p1[p1.p >= 0.995].merge(fin, on=["a", "b"])
    d = d[d.pf > 0.5].sample(15000, random_state=0).assign(country=c)
    parts.append(d)
S = pd.concat(parts, ignore_index=True)
S[["a", "b"]].to_parquet("/tmp/conf_sample.parquet")
score(SimpleNamespace(pairs="/tmp/conf_sample.parquet", band=None, art=art, model="models/ce_base", out="/tmp/conf_ce.parquet", max_len=128, view="raw"))
S = S.merge(pd.read_parquet("/tmp/conf_ce.parquet"), on=["a", "b"])
S.to_parquet("/tmp/conf_scored.parquet")
for c, g in S.groupby("country"):
    print(f"{c:7s}: confident links sampled {len(g)} | CE logit <0: {np.mean(g.ce < 0):.2%} | < -2: {np.mean(g.ce < -2):.2%} | median {g.ce.median():.2f}", flush=True)
