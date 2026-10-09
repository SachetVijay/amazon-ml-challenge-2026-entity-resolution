"""Name-vs-address conflicts. For a record with an address: X = candidate S1 with (near-)same core name, same street words,
DIFFERENT house number; Y = candidate S1 with the SAME house number + street, clearly different name.
train (full-density candidates of bucket-0 band records + rivals are not all scored -> use the raw candidate files + GT):
    which is the true owner when both exist?
test: how often does each country have such conflicts, and which does v13 pick (exclusive winner)?"""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
split = sys.argv[1]; art = f"artifacts/{split}"
rows_all = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
countries = ("US", "India") if split == "train" else ("US", "India", "France")
if split == "train":
    sys.path.insert(0, "src"); from er.prep import load_gt
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); owner = dict(zip(gt.other, gt.s1))
for c in countries:
    idx = np.flatnonzero(rows_all == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "name_core", "addr_clean", "addr_first_num"], filters=[("country", "==", c)])
    r.index = idx
    cand = pd.read_parquet(f"{art}/cand_{c}_k5_5.parquet", columns=["a", "b"]) if split == "train" else \
           pd.read_parquet(f"{art}/pred_hyb_{c}.cand.parquet", columns=["a", "b"])
    if split == "train":   # sample records to keep it light
        bs = np.random.default_rng(0).choice(cand.b.unique(), 400000, replace=False); cand = cand[cand.b.isin(bs)]
    cand = cand[r.addr_clean.astype(object).loc[cand.b].to_numpy() != ""]
    A, B = cand.a.to_numpy(), cand.b.to_numpy()
    nm = cpdist(r.name_core.astype(object).loc[A].tolist(), r.name_core.astype(object).loc[B].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    strip = lambda s: " ".join(t for t in s.split() if not t.isdigit())
    st = cpdist([strip(x) for x in r.addr_clean.astype(object).loc[A]], [strip(x) for x in r.addr_clean.astype(object).loc[B]], scorer=fuzz.token_set_ratio, workers=-1)
    na, nb = r.addr_first_num.astype(object).loc[A].to_numpy(), r.addr_first_num.astype(object).loc[B].to_numpy()
    samenum = (na == nb) & (nb != "")
    X = cand[(nm >= 90) & (st >= 80) & ~samenum & (nb != "")]
    Y = cand[(nm < 50) & (st >= 90) & samenum]
    both = np.intersect1d(X.b.unique(), Y.b.unique())
    msg = f"{split} {c:7s}: records with a conflict {len(both)} ({len(both)/max(len(np.unique(B)),1):.3%} of records with address)"
    if split == "train":
        ids = r.entity_id.astype(object)
        Xo = X[X.b.isin(both)]; Yo = Y[Y.b.isin(both)]
        own = {b: owner.get(ids[b]) for b in both}
        tx = np.mean([any(ids[a] == own[b] for a in Xo.a[Xo.b == b]) for b in both])
        ty = np.mean([any(ids[a] == own[b] for a in Yo.a[Yo.b == b]) for b in both])
        tn = np.mean([own[b] is None for b in both])
        msg += f" | TRUE owner is the name-match X {tx:.1%}, the address-match Y {ty:.1%}, record unmatched {tn:.1%}"
    else:
        P = pd.read_parquet(f"{art}/pred_s2_{c}.parquet")
        win = P.loc[P.groupby("b").p.idxmax()].set_index("b")
        w = win.reindex(both)
        xs = set(zip(X.a, X.b)); ys_ = set(zip(Y.a, Y.b))
        px = np.mean([(a, b) in xs and p > 0.5 for b, a, p in zip(both, w.a, w.p)])
        py = np.mean([(a, b) in ys_ and p > 0.5 for b, a, p in zip(both, w.a, w.p)])
        msg += f" | v13 links it to name-match X {px:.1%}, to address-match Y {py:.1%}, other/none {1-px-py:.1%}"
    print(msg, flush=True)
