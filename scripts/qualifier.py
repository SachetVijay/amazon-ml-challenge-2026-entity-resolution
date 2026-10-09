"""Record ADDS tokens to the S1 name (legal forms ignored; nothing dropped). Classify added tokens by per-country filler score
(df_records/df_S1): 'qualifier' if any added token >= 20, 'filler' if all added in [2,20), else 'other'.
Validation: truth rate by class x same/different number. Test: counts per S1, share confident (p1>=0.995), accepted."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL, GENERIC
from street_mix import street
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def ratios(r):
    s = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
    o = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
    return (o + 1e-4) / (s.reindex(o.index).fillna(0) + 1e-4)
def run(split, c, P, ys=None):
    art = f"artifacts/{split}"
    rows = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["src", "name_clean", "business_address"], filters=[("country", "==", c)]); r.index = rows
    ra = ratios(r); n1 = (r.src == 1).sum() if ys is None else P.s1_id.nunique()
    A, B = r.name_clean.astype(object).loc[P.a].tolist(), r.name_clean.astype(object).loc[P.b].tolist()
    cls = []
    for x, y in zip(A, B):
        ta, tb = toks(x), toks(y); add = tb - ta
        if not add or (ta - tb) or not (ta & tb): cls.append(""); continue
        m = max(ra.get(t, 1) for t in add)
        cls.append("qualifier(>=20)" if m >= 20 else ("filler[2,20)" if min(ra.get(t, 1) for t in add) >= 2 else "other"))
    P = P.assign(cls=cls); P = P[P.cls != ""].copy()
    na = [street(x)[0] for x in r.business_address.astype(object).loc[P.a]]; nb = [street(x)[0] for x in r.business_address.astype(object).loc[P.b]]
    P["num"] = np.where([x == y and x != "" for x, y in zip(na, nb)], "same no.", "diff/none")
    agg = dict(pairs_per_S1=("p", lambda s: len(s) / n1), confident=("p1", lambda s: np.mean(s >= 0.995)), accepted=("p", lambda s: np.mean(s > .5)))
    if ys is not None:
        P["y"] = [(a, b) in ys for a, b in zip(P.s1_id, P.o_id)]; agg["TRUE"] = ("y", "mean")
    g = P.groupby(["cls", "num"]).agg(**agg)
    top = sorted({t for x, y in zip(r.name_clean.astype(object).loc[P.a[P.cls == "qualifier(>=20)"]], r.name_clean.astype(object).loc[P.b[P.cls == "qualifier(>=20)"]]) for t in toks(y) - toks(x)} , key=lambda t: -ra.get(t, 0))[:10]
    print(f"== {split} {c}  (qualifier words e.g. {top}; in GENERIC: {[t for t in top if t in GENERIC]})\n" + g.round(3).to_string(), flush=True)
split = sys.argv[1]
if split == "train":
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
    oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "b", "p", "country"]).rename(columns={"p": "p1"}); full = full.merge(oof, on=["a", "b"])
    for c in ("US", "India"): run("train", c, full[(full.country == c) & (full.p1 > 1e-3)].reset_index(drop=True), ys)
else:
    for c in ("US", "India", "France"):
        P = pd.read_parquet(f"artifacts/test/pred_hyb_{c}.parquet", columns=["a", "b", "p"]).rename(columns={"p": "p1"})
        P = P[P.p1 > 1e-3].merge(pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet"), on=["a", "b"]).reset_index(drop=True)
        run("test", c, P)
