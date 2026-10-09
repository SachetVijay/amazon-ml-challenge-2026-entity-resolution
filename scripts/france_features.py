import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.features import NEEDED, pair_features
from er.predict import country_rows
path = "artifacts/test/records.parquet"
cols = sorted(set(NEEDED + ["entity_id", "business_name", "business_address"]))
out = {}
for c in ("France", "US"):
    rows = country_rows(path)[c]
    sub = pd.read_parquet(path, columns=cols, filters=[("country", "==", c)], dtype_backend="pyarrow")
    sub.index = rows
    P = pd.read_parquet(f"artifacts/test/v4_backup/pred_s2_{c}.parquet")
    best = P.loc[P.groupby("b").p.idxmax()]
    for band, (lo, hi) in {"uncertain": (0.2, 0.95), "confident": (0.95, 1.01)}.items():
        s = best[(best.p >= lo) & (best.p < hi)].sample(6000, random_state=0)
        F = pair_features(sub, s[["a", "b"]].reset_index(drop=True), workers=8)
        out[(c, band)] = F.drop(columns=["a", "b"]).median()
    del sub
T = pd.DataFrame(out)
keep = ["nc_ratio", "nc_tset", "nk_ratio", "nk_jw", "nsp_ratio", "skel_ratio", "name_jac", "name_cont", "addr_ratio",
        "addr_tset", "addr_jac", "addr_cont", "addrskel_jac", "num_jac", "firstnum_eq", "num_partial", "num_conflict",
        "legal_jac", "acronym", "lenA", "lenB", "addrB_empty", "nameA_freq", "nameB_s1freq"]
print(T.loc[[k for k in keep if k in T.index]].round(3).to_string())
