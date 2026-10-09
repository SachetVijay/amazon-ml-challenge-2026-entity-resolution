"""Ground-truth label rate of single-word swaps in TRAIN (bucket-0 OOF pairs), by type of the added word."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "s1_id", "o_id", "country"])
oof = oof[oof.p > 0.05]
r = pd.read_parquet("artifacts/train/records.parquet", columns=["name_clean", "country", "src"])
ys = set(map(tuple, load_gt("student_resource/dataset/train/train_ground_truth.tsv")[["s1", "other"]].to_numpy()))
oof["y"] = [(s, o) in ys for s, o in zip(oof.s1_id, oof.o_id)]
nm = r.name_clean.astype(object).to_numpy()
for c in ("US", "India"):
    s1 = r[(r.country == c) & (r.src == 1)].name_clean.astype(object)
    sdf = pd.Series([t for n in s1 for t in toks(n)]).value_counts() / len(s1)
    CTRY = {"usa", "us", "america", "american", "india", "indian", "bharat", "united", "states"}
    FILL = {"co", "company", "cie", "group", "groupe", "services", "holdings", "holding", "international", "enterprises", "and"}
    P = oof[oof.country == c]
    rows = []
    for a, b, y, p in zip(P.a, P.b, P.y, P.p):
        ta, tb = toks(nm[a]), toks(nm[b]); D, A = ta - tb, tb - ta
        if len(D) != 1 or len(A) != 1 or not (ta & tb): continue
        u, v = next(iter(D)), next(iter(A))
        if sdf.get(u, 0) < 2e-4: continue
        k = "->country" if v in CTRY else "->filler" if v in FILL else "->category" if sdf.get(v, 0) >= 2e-4 else "->rare"
        rows.append((k, y, p, nm[a], nm[b]))
    D = pd.DataFrame(rows, columns=["k", "y", "p", "s1", "o"])
    print(f"== {c}")
    print(D.groupby("k").agg(n=("y", "size"), true_rate=("y", "mean"), mean_p=("p", "mean")).to_string())
    for k in ("->country", "->category"):
        print(k, "examples:"); print(D[D.k == k].sample(min(6, (D.k == k).sum()), random_state=0)[["y", "s1", "o"]].to_string(index=False))
