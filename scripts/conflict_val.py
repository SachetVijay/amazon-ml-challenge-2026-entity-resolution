"""Validation (labels, v12 OOF): name-match pairs X (bucket 0) of records that ALSO have an address-match rival Y among
their full-density candidates. Truth rate vs model p; F effect of raising p(X) to tau."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src")
from er.prep import load_gt
from er.decide import exclusive, expected_f_fast, f05
art = "artifacts/train"
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full = pd.read_parquet(f"{art}/oof_band_v12.parquet")
oof = pd.read_parquet(f"{art}/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a")
full = full.merge(oof, on="a")
rows_all = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
strip = lambda s: " ".join(t for t in s.split() if not t.isdigit())
Xall = []
for c in ("US", "India"):
    idx = np.flatnonzero(rows_all == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["name_core", "addr_clean", "addr_first_num"], filters=[("country", "==", c)]); r.index = idx
    F = full[(full.country == c)]
    F = F[r.addr_clean.astype(object).loc[F.b].to_numpy() != ""]
    nm = cpdist(r.name_core.astype(object).loc[F.a].tolist(), r.name_core.astype(object).loc[F.b].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    st = cpdist([strip(x) for x in r.addr_clean.astype(object).loc[F.a]], [strip(x) for x in r.addr_clean.astype(object).loc[F.b]], scorer=fuzz.token_set_ratio, workers=-1)
    na, nb = r.addr_first_num.astype(object).loc[F.a].to_numpy(), r.addr_first_num.astype(object).loc[F.b].to_numpy()
    X = F[(nm >= 90) & (st >= 80) & (na != nb) & (nb != "")]
    # rivals Y at full density: sparse + dense candidate files, any S1
    bl = list(map(int, X.b.unique()))
    s = pd.read_parquet(f"{art}/cand_{c}_k5_5.parquet", columns=["a", "b"], filters=[("b", "in", bl)])
    d = pd.read_parquet(f"{art}/dense_{c}_k10_10.parquet", columns=["a", "b"], filters=[[("b", "in", bl), ("dense_rank_b2a", "<=", 5)], [("b", "in", bl), ("dense_rank_a2b", "<=", 5)]])
    C = pd.concat([s, d]).drop_duplicates()
    nm2 = cpdist(r.name_core.astype(object).loc[C.a].tolist(), r.name_core.astype(object).loc[C.b].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    st2 = cpdist([strip(x) for x in r.addr_clean.astype(object).loc[C.a]], [strip(x) for x in r.addr_clean.astype(object).loc[C.b]], scorer=fuzz.token_set_ratio, workers=-1)
    same2 = (r.addr_first_num.astype(object).loc[C.a].to_numpy() == r.addr_first_num.astype(object).loc[C.b].to_numpy())
    Yb = set(C.b[(nm2 < 50) & (st2 >= 90) & same2])
    X = X.assign(conflict=X.b.isin(Yb), y=[(s_, o_) in ys for s_, o_ in zip(X.s1_id, X.o_id)])
    for k, g in X.groupby("conflict"):
        print(f"{c} name-match pairs, conflict={k}: n={len(g)} TRUE rate {g.y.mean():.3f} | model p>0.5 {np.mean(g.p > .5):.3f}, p median {g.p.median():.3f}", flush=True)
    Xall.append(X[X.conflict])
X = pd.concat(Xall)
samp = set(full.s1_id); truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict(); allS = samp | set(truth)
def macro(P):
    ex = exclusive(P[["a", "b", "s1_id", "o_id", "p"]]); sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
    return np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in allS])
key = pd.MultiIndex.from_arrays([full.a, full.b]); xk = pd.MultiIndex.from_arrays([X.a, X.b])
base = macro(full)
print(f"baseline {base:.5f}")
for tau in (0.5, 0.6, 0.75):
    p = full.p.to_numpy().copy(); m = key.isin(xk); p[m] = np.maximum(p[m], tau)
    print(f"raise conflict name-matches to >= {tau}: {macro(full.assign(p=p)):.5f} ({m.sum()} pairs)", flush=True)
