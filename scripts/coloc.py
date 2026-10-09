"""Renamed/coined record (no shared name word with S1) linked at the same house number: truth rate by k_addr = number of
S1s sharing the record's (street, number) in the country (full density). Validation labels + test France/US/India."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from street_mix import street
from er.normalize import LEGAL
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def analyse(split, c, S, ys=None):
    art = f"artifacts/{split}"
    rows = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["src", "business_address", "name_clean"], filters=[("country", "==", c)]); r.index = rows
    s1 = r[r.src == 1]
    k1 = pd.Series([street(x) for x in s1.business_address.astype(object)], index=s1.index)
    kk = k1[k1.map(lambda t: t[0] != "")].map(lambda t: t[0] + "|" + t[1])
    kcount = kk.value_counts()
    na, nb = r.name_clean.astype(object).loc[S.a].tolist(), r.name_clean.astype(object).loc[S.b].tolist()
    coined = np.array([not (toks(x) & toks(y)) for x, y in zip(na, nb)])
    S = S[coined].copy()
    sb = [street(x) for x in r.business_address.astype(object).loc[S.b]]
    sa = kk.reindex(S.a).to_numpy()
    same = np.array([isinstance(ka, str) and y[0] != "" and ka.split("|")[0] == y[0] for ka, y in zip(sa, sb)])
    S = S[same]; S["k_addr"] = kcount.reindex(kk.reindex(S.a)).fillna(1).clip(upper=4).to_numpy()
    n1 = (r.src == 1).sum() if ys is None else None
    agg = {"links": S.groupby("k_addr").size(), "p_median": S.groupby("k_addr").p.median()}
    if ys is not None:
        S["y"] = [(a, b) in ys for a, b in zip(S.s1_id, S.o_id)]; agg["TRUE rate"] = S.groupby("k_addr").y.mean()
    print(f"== {split} {c} renamed-record links at the same number, by # S1 at that street+number (4 = 4+)\n" + pd.DataFrame(agg).round(3).to_string(), flush=True)
split = sys.argv[1]
if split == "train":
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
    oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a"); full = full.merge(oof, on="a")
    for c in ("US", "India"):
        F = full[full.country == c]; ex = exclusive(F[["a", "b", "s1_id", "o_id", "p"]])
        S = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
        analyse("train", c, S, ys)
else:
    for c in ("US", "India", "France"):
        P = pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet"); S = expected_f_fast(exclusive(P))
        analyse("test", c, S)
