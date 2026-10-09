"""v16 vs v14 France link changes by cell, with US+India validation truth rate per cell (numshift relation for num≠)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from cellfn import cells
from er.decide import exclusive, expected_f_fast
v14 = pd.read_parquet("backup_pred_s2_v14/pred_s2_France.parquet"); v16 = pd.read_parquet("backup_pred_s2_v16/pred_s2_France.parquet")
s14 = expected_f_fast(exclusive(v14))[["a", "b"]]; s16 = expected_f_fast(exclusive(v16))[["a", "b"]]
k14 = set(map(tuple, s14.to_numpy())); k16 = set(map(tuple, s16.to_numpy()))
add = pd.DataFrame(list(k16 - k14), columns=["a", "b"]); rem = pd.DataFrame(list(k14 - k16), columns=["a", "b"])
ca, _ = cells("test", "France", add); cr, _ = cells("test", "France", rem)
T = pd.read_parquet("/tmp/cell_train.parquet").groupby("cell").y.mean()
A = pd.Series(ca).value_counts(); R = pd.Series(cr).value_counts()
D = pd.DataFrame({"added": A, "removed": R}).fillna(0).astype(int); D["US/IN val truth"] = T.reindex(D.index).round(3)
D["net"] = D.added - D.removed
D = D.reindex(D.added.add(D.removed).sort_values(ascending=False).index)
print(D.head(22).to_string())
