"""Cell-wise transfer check. Cell = name relation x number relation x street agreement (+address-less).
Validation (US+India labels): truth rate per cell. Test: pairs/S1 and accepted/S1 per cell per country.
Implied France shortfall = pairs/S1 * (truth - accepted rate) per cell (assumes per-cell truth transfers)."""
import re, sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from street_mix import street
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def ratios(r):
    s = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
    o = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
    return (o + 1e-4) / (s.reindex(o.index).fillna(0) + 1e-4)
def cells(split, c, P):
    art = f"artifacts/{split}"
    rows = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["src", "name_clean", "business_address", "addr_clean"], filters=[("country", "==", c)]); r.index = rows
    ra = ratios(r)
    A, B = r.name_clean.astype(object).loc[P.a].tolist(), r.name_clean.astype(object).loc[P.b].tolist()
    nm = []
    for x, y in zip(A, B):
        ta, tb = toks(x), toks(y); add, drop = tb - ta, ta - tb
        if not ta & tb: nm.append("renamed"); continue
        if not add and not drop: nm.append("same"); continue
        if add and not drop: nm.append("adds:" + ("qual" if max(ra.get(t, 1) for t in add) >= 20 else ("fill" if min(ra.get(t, 1) for t in add) >= 2 else "word"))); continue
        if drop and not add: nm.append("drops"); continue
        nm.append("swap1" if len(add) == 1 and len(drop) == 1 else "multi")
    sa = [street(x) for x in r.business_address.astype(object).loc[P.a]]; sb = [street(x) for x in r.business_address.astype(object).loc[P.b]]
    st = cpdist([x[1] for x in sa], [x[1] for x in sb], scorer=fuzz.ratio, workers=-1)
    ab = r.addr_clean.astype(object).loc[P.b].to_numpy()
    out = []
    for n, (xa, _), (xb, _), s, e in zip(nm, sa, sb, st, ab):
        if e == "": out.append(f"{n} | no-address"); continue
        num = "no-num" if not xb or not xa else ("num=" if xa == xb else ("num-drop" if (xa.endswith(xb) or xa.startswith(xb) or xb.endswith(xa) or xb.startswith(xa)) else "num≠"))
        out.append(f"{n} | {num} | {'st=' if s >= 85 else 'st≠'}")
    return np.array(out), (r.src == 1).sum()
split = sys.argv[1]
if split == "train":
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
    oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "country"]).rename(columns={"p": "p1"}); full = full.merge(oof, on=["a", "b"])
    parts = []
    for c in ("US", "India"):
        P = full[(full.country == c) & (full.p1 > 1e-3)].reset_index(drop=True)
        k, _ = cells("train", c, P)
        parts.append(pd.DataFrame({"cell": k, "y": [(a, b) in ys for a, b in zip(P.s1_id, P.o_id)], "acc": P.p.to_numpy() > .5, "c": c}))
    D = pd.concat(parts); D.to_parquet("/tmp/cell_train.parquet")
else:
    parts = []
    for c in ("US", "India", "France"):
        P = pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet")
        k, n1 = cells("test", c, P)
        parts.append(pd.DataFrame({"cell": k, "acc": P.p.to_numpy() > .5, "c": c, "n1": n1}))
    D = pd.concat(parts); D.to_parquet("/tmp/cell_test.parquet")
    T = pd.read_parquet("/tmp/cell_train.parquet")
    tr = T.groupby("cell").agg(val_n=("y", "size"), val_true=("y", "mean"), val_acc=("acc", "mean"))
    rows = {}
    for c, g in D.groupby("c"):
        n1 = g.n1.iloc[0]
        rows[f"{c} pairs/S1"] = g.groupby("cell").size() / n1
        rows[f"{c} acc"] = g.groupby("cell").acc.mean()
    R = tr.join(pd.DataFrame(rows), how="outer").fillna(0)
    R["FR shortfall/S1"] = R["France pairs/S1"] * (R.val_true - R["France acc"])
    R["US shortfall/S1"] = R["US pairs/S1"] * (R.val_true - R["US acc"])
    R = R[R.val_n >= 200].sort_values("FR shortfall/S1")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    cols = ["val_n", "val_true", "val_acc", "US pairs/S1", "US acc", "France pairs/S1", "France acc", "US shortfall/S1", "FR shortfall/S1"]
    print(R[cols].round(4).to_string())
    print("\nFrance: sum of positive shortfall (implied misses/S1)", round(R["FR shortfall/S1"].clip(lower=0).sum(), 4),
          "| sum of negative (implied extra false accepts/S1)", round(R["FR shortfall/S1"].clip(upper=0).sum(), 4))
    print("US    : sum positive", round(R["US shortfall/S1"].clip(lower=0).sum(), 4), "| negative", round(R["US shortfall/S1"].clip(upper=0).sum(), 4))
