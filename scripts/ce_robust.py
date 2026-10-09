"""Per-CE disagreement on confident stage-1 links (pseudo-positives) and on confident negatives, per country.
A French-robust CE behaves in France as in US/India."""
import sys
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.cross_encoder import score
art = "artifacts/test"
S = pd.read_parquet("/tmp/conf_scored.parquet")[["a", "b", "country"]]
# pseudo-negatives: band pairs whose final p < 0.01 and stage-1 p1 < 0.05 (clear rejects), sampled
neg = []
for c in ("US", "India", "France"):
    p1 = pd.read_parquet(f"{art}/pred_hyb_{c}.parquet", columns=["a", "b", "p"])
    fin = pd.read_parquet(f"{art}/pred_s2_{c}.parquet").rename(columns={"p": "pf"})
    d = p1[(p1.p > 0.005) & (p1.p < 0.05)].merge(fin, on=["a", "b"]); d = d[d.pf < 0.01].sample(10000, random_state=0)
    neg.append(d[["a", "b"]].assign(country=c))
N = pd.concat(neg); N["kind"] = "neg"; S["kind"] = "pos"
A = pd.concat([S, N], ignore_index=True); A[["a", "b"]].drop_duplicates().to_parquet("/tmp/ce_rob_pairs.parquet")
for j, m, v in ((0, "ce_base", "raw"), (1, "ce_base2", "raw"), (2, "ce_mdeberta", "raw"), (3, "ce_norm", "norm"), (4, "ce_hard", "raw"), (5, "ce_large", "raw")):
    out = f"/tmp/ce_rob_{j}.parquet"
    score(SimpleNamespace(pairs="/tmp/ce_rob_pairs.parquet", band=None, art=art, model=f"models/{m}", out=out, max_len=128, view=v))
    A = A.merge(pd.read_parquet(out).rename(columns={"ce": f"ce{j}"}), on=["a", "b"], how="left")
A.to_parquet("/tmp/ce_rob_scored.parquet")
names = ["base", "base2", "mdeberta", "norm", "hard", "large"]
rows = []
for j, nm in enumerate(names):
    row = {"CE": f"ce{j} {nm}"}
    for c in ("US", "India", "France"):
        pos = A[(A.kind == "pos") & (A.country == c)][f"ce{j}"]; ng = A[(A.kind == "neg") & (A.country == c)][f"ce{j}"]
        row[f"{c} rejects true %"] = round(100 * np.mean(pos < 0), 2); row[f"{c} accepts false %"] = round(100 * np.mean(ng > 0), 2)
    rows.append(row)
print(pd.DataFrame(rows).set_index("CE").to_string())
