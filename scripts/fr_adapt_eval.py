"""Compare ce_base2 vs adapted ce_base2_fr on pseudo-labels: held-out French positives, US/India/France confident
positives (conf sample) and pseudo-negatives (ce_rob sample)."""
import sys
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.cross_encoder import score
art = "artifacts/test"
H = pd.read_parquet("kaggle_frad/held_pos.parquet").assign(country="France", kind="heldpos")
A = pd.read_parquet("/tmp/ce_rob_scored.parquet")[["a", "b", "country", "kind", "ce1"]]
P = pd.concat([H, A[["a", "b", "country", "kind"]]], ignore_index=True)
P[["a", "b"]].drop_duplicates().to_parquet("/tmp/frad_eval_pairs.parquet")
for tag, m in (("orig", "models/ce_base2"), ("adapt", "models/ce_base2_fr")):
    score(SimpleNamespace(pairs="/tmp/frad_eval_pairs.parquet", band=None, art=art, model=m, out=f"/tmp/frad_{tag}.parquet", max_len=128, view="raw"))
    P = P.merge(pd.read_parquet(f"/tmp/frad_{tag}.parquet").rename(columns={"ce": tag}), on=["a", "b"], how="left")
rows = []
for (c, k), g in P.groupby(["country", "kind"]):
    f = (lambda s: np.mean(s < 0)) if k != "neg" else (lambda s: np.mean(s > 0))
    rows.append({"country": c, "set": k + (" (reject rate)" if k != "neg" else " (accept rate)"), "n": len(g),
                 "orig %": round(100 * f(g.orig), 2), "adapted %": round(100 * f(g.adapt), 2)})
print(pd.DataFrame(rows).to_string(index=False))
