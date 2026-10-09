"""Same name + same STREET + different house number: truth rate (train, labels) and model p, by number length.
Street = the comma component holding the house number, normalised (ascii, lower, abbreviations, no digits/punct)."""
import re, sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src")
from er.normalize import to_ascii, ADDR_ABBR
split = sys.argv[1]; art = f"artifacts/{split}"
rows_all = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
def street(addr):
    for comp in (addr or "").split(","):
        s = to_ascii(comp).lower()
        m = re.search(r"\d+", s)
        if m:
            words = [ADDR_ABBR.get(t, t) for t in re.findall(r"[a-z]+", s)]
            words = [w for w in words if w not in {"no", "number", "n", "deg", "bis", "ter", "de", "du", "des", "la", "le", "les", "d", "l"}]
            return int(m.group()), " ".join(words)
    return None, ""
if split == "train":
    from er.prep import load_gt
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")[["a", "b", "s1_id", "o_id", "p"]]
    oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a")
    full = full.merge(oof, on="a")
countries = ("US", "India") if split == "train" else ("US", "India", "France")
for c in countries:
    idx = np.flatnonzero(rows_all == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["business_address", "name_core"], filters=[("country", "==", c)]); r.index = idx
    P = full[full.country == c] if split == "train" else pd.read_parquet(f"{art}/pred_hyb_{c}.cand.parquet", columns=["a", "b"])
    nm = cpdist(r.name_core.astype(object).loc[P.a].tolist(), r.name_core.astype(object).loc[P.b].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    P = P[nm >= 90].copy()
    sa = [street(x) for x in r.business_address.astype(object).loc[P.a]]; sb = [street(x) for x in r.business_address.astype(object).loc[P.b]]
    P["na"] = [x[0] for x in sa]; P["nb"] = [x[0] for x in sb]
    ss = cpdist([x[1] for x in sa], [x[1] for x in sb], scorer=fuzz.ratio, workers=-1)
    P = P[(ss >= 90) & P.na.notna() & P.nb.notna() & (P.na != P.nb)].copy()
    P["len"] = np.where(np.maximum(P.na, P.nb) < 100, "1-2 digits", np.where(np.maximum(P.na, P.nb) < 1000, "3 digits", "4+ digits"))
    if split == "test":
        pp = pd.read_parquet(f"{art}/pred_s2_{c}.parquet").set_index(["a", "b"]).p
        P["p"] = pp.reindex(pd.MultiIndex.from_arrays([P.a, P.b])).fillna(0).to_numpy()
    else:
        P["y"] = [(s, o) in ys for s, o in zip(P.s1_id, P.o_id)]
    agg = dict(n=("p", "size"), p_median=("p", "median"), p_gt_half=("p", lambda s: (s > .5).mean()))
    if split == "train": agg["TRUE_rate"] = ("y", "mean")
    g = P.groupby("len").agg(**agg)
    print(f"{split} {c}: same name + same street + different number\n{g.round(3).to_string()}\n", flush=True)
