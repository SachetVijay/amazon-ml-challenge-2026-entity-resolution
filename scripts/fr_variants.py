"""Compare France predictions of variants (pred_s2 files) on key cells: acceptance (p>0.5) per cell + total links."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from cellfn import cells
from er.decide import exclusive, expected_f_fast
files = dict(v.split("=") for v in sys.argv[1:])
base = pd.read_parquet(list(files.values())[0])
k, n1 = cells("test", "France", base)
K = pd.DataFrame({"a": base.a, "b": base.b, "cell": k})
show = ["adds:qual | num= | st=", "adds:fill | num= | st=", "adds:word | num= | st=", "swap1 | num= | st=", "renamed | num= | st=",
        "same | num= | st=", "same | num≠ | st=", "adds:qual | num≠ | st=", "adds:fill | num≠ | st=", "swap1 | num≠ | st=", "same | no-address", "renamed | num= | st≠"]
rows = {}
for name, f in files.items():
    P = pd.read_parquet(f).merge(K, on=["a", "b"])
    acc = P.groupby("cell").p.apply(lambda s: np.mean(s > .5))
    sel = expected_f_fast(exclusive(P[["a", "b", "p"]]))
    rows[name] = pd.concat([acc.reindex(show), pd.Series({"LINKS/S1": len(sel) / n1, "E[links]/S1": exclusive(P[["a", "b", "p"]]).p.sum() / n1})])
print(pd.DataFrame(rows).round(4).to_string())
