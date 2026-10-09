"""Language-agnostic category-word swap: names (minus legal forms) differ by exactly one token on each side, both
differing tokens are FREQUENT in this country's S1 names (df >= q), >=1 shared token, and the addresses agree.
Validation: positive rate + model p.  Test: prevalence + model p per country."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src")
from er.normalize import LEGAL
DF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
def toks(s): return [t for t in s.split() if t not in LEGAL]
def run(art, P, country, rows, label=None):
    r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "name_clean", "addr_clean"], filters=[("country", "==", country)])
    r.index = rows
    s1 = r[r.src == 1]
    df = pd.Series([t for n in s1.name_clean.astype(object) for t in set(toks(n))]).value_counts() / len(s1)
    freq = set(df[df >= DF].index)
    A = r.name_clean.astype(object).loc[P.a].tolist(); B = r.name_clean.astype(object).loc[P.b].tolist()
    sw = np.zeros(len(P), bool)
    for i, (x, y) in enumerate(zip(A, B)):
        tx, ty = set(toks(x)), set(toks(y))
        dx, dy = tx - ty, ty - tx
        if len(dx) == 1 and len(dy) == 1 and tx & ty and next(iter(dx)) in freq and next(iter(dy)) in freq:
            sw[i] = True
    addr = cpdist(r.addr_clean.astype(object).loc[P.a[sw]].tolist(), r.addr_clean.astype(object).loc[P.b[sw]].tolist(),
                  scorer=fuzz.token_set_ratio, workers=-1)
    m = np.zeros(len(P), bool); m[np.flatnonzero(sw)[addr >= 90]] = True
    Q = P[m]
    msg = (f"{country:7s}: {len(freq)} frequent name tokens | category-swap same-address pairs {m.sum()} "
           f"({m.sum() / (r.src == 1).sum():.4f}/S1) | model p: median {Q.p.median():.3f}, p>0.5 {np.mean(Q.p > .5):.1%}")
    if label is not None: msg += f" | TRUE rate {label[m].mean():.3f}"
    print(msg, flush=True)
    return Q
rows_all = {s: pd.read_parquet(f"artifacts/{s}/records.parquet", columns=["country"]).country.to_numpy() for s in ("train", "test")}
if sys.argv[1] == "val":
    from er.prep import load_gt
    full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
    oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a")
    full = full.merge(oof, on="a")
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    for c in ("US", "India"):
        F = full[(full.country == c) & (full.p > 1e-3)].reset_index(drop=True)
        y = np.fromiter(((s, o) in ys for s, o in zip(F.s1_id, F.o_id)), bool, len(F))
        run("artifacts/train", F, c, np.flatnonzero(rows_all["train"] == c), y)
else:
    for c in ("US", "India", "France"):
        P = pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet")
        Q = run("artifacts/test", P, c, np.flatnonzero(rows_all["test"] == c))
        if c == "France": Q.to_parquet("/tmp/fr_catswap.parquet")
