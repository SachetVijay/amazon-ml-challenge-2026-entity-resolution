"""Accepted links by (name agreement, street agreement, number agreement); test shares per country and validation
precision (US/India labels, v12 OOF decisions). Street = words of the comma part holding the house number."""
import re, sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src")
from er.normalize import to_ascii, ADDR_ABBR
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
STOP = {"no", "number", "n", "deg", "bis", "ter", "de", "du", "des", "la", "le", "les", "d", "l", "unit", "apt", "suite"}
def street(addr):
    for comp in (addr or "").split(","):
        s = to_ascii(comp).lower(); m = re.search(r"\d+", s)
        if m: return m.group().lstrip("0"), " ".join(w for w in (ADDR_ABBR.get(t, t) for t in re.findall(r"[a-z]+", s)) if w not in STOP)
    return "", ""
def classify(r, A, B):
    na = cpdist(r.name_core.astype(object).loc[A].tolist(), r.name_core.astype(object).loc[B].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    sa = [street(x) for x in r.business_address.astype(object).loc[A]]; sb = [street(x) for x in r.business_address.astype(object).loc[B]]
    st = cpdist([x[1] for x in sa], [x[1] for x in sb], scorer=fuzz.ratio, workers=-1)
    out = []
    for n, s, x, y, ab in zip(na, st, sa, sb, r.addr_clean.astype(object).loc[B]):
        if ab == "": out.append("address-less"); continue
        nm = "name=" if n >= 90 else ("name~" if n >= 60 else "name≠")
        if not y[1]: out.append(nm + " | no street/number"); continue
        sm = "street=" if s >= 85 else ("street~" if s >= 60 else "street≠")
        nb = "num=" if x[0] == y[0] else "num≠"
        out.append(f"{nm} {sm} {nb}")
    return out
if __name__ == "__main__":
    res = {}
    for c in ("US", "India", "France"):
        rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == c)
        r = pd.read_parquet("artifacts/test/records.parquet", columns=["business_address", "name_core", "addr_clean"], filters=[("country", "==", c)]); r.index = rows
        P = pd.read_parquet(f"artifacts/test/pred_s2_{c}.parquet"); S = expected_f_fast(exclusive(P))
        res[f"test {c} %"] = pd.Series(classify(r, S.a, S.b)).value_counts(normalize=True) * 100
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
    full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
    oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a"); full = full.merge(oof, on="a")
    for c in ("US", "India"):
        rows = np.flatnonzero(pd.read_parquet("artifacts/train/records.parquet", columns=["country"]).country.to_numpy() == c)
        r = pd.read_parquet("artifacts/train/records.parquet", columns=["business_address", "name_core", "addr_clean"], filters=[("country", "==", c)]); r.index = rows
        F = full[full.country == c]; ex = exclusive(F[["a", "b", "s1_id", "o_id", "p"]]); S = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
        S["k"] = classify(r, S.a, S.b); S["y"] = [(s, o) in ys for s, o in zip(S.s1_id, S.o_id)]
        g = S.groupby("k").y.agg(["size", "mean"])
        res[f"val {c} %"] = g["size"] / g["size"].sum() * 100; res[f"val {c} prec"] = g["mean"] * 100
    T = pd.DataFrame(res).fillna(0).sort_values("test France %", ascending=False)
    pd.set_option("display.width", 200); print(T.round(2).to_string())
