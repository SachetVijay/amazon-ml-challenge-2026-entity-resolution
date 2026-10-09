"""Pseudo-label check of adapted cross-encoders: reject rate on confident true links, accept rate on pseudo-negatives,
per country, original vs adapted.  usage: adapt_eval.py orig_dir:adapted_dir[:view] ..."""
import sys
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.cross_encoder import score
A = pd.read_parquet("/tmp/ce_rob_scored.parquet")[["a", "b", "country", "kind"]]
H = pd.read_parquet("kaggle_frad/held_pos.parquet").assign(country="France", kind="heldpos")
P = pd.concat([H, A], ignore_index=True); P[["a", "b"]].drop_duplicates().to_parquet("/tmp/ae_pairs.parquet")
for spec in sys.argv[1:]:
    o, n = spec.split(":")[:2]; v = spec.split(":")[2] if spec.count(":") > 1 else "raw"
    Q = P.copy()
    for tag, m in (("orig", o), ("adapt", n)):
        out = f"/tmp/ae_{m.replace('/', '_')}.parquet"
        score(SimpleNamespace(pairs="/tmp/ae_pairs.parquet", band=None, art="artifacts/test", model=m, out=out, max_len=128, view=v))
        Q = Q.merge(pd.read_parquet(out).rename(columns={"ce": tag}), on=["a", "b"], how="left")
    rows = []
    for (c, k), g in Q.groupby(["country", "kind"]):
        f = (lambda s: np.mean(s > 0)) if k == "neg" else (lambda s: np.mean(s < 0))
        rows.append({"model": n, "country": c, "set": k, "orig %": round(100 * f(g.orig), 2), "adapted %": round(100 * f(g.adapt), 2)})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
