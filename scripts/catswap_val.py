"""Same category-swap definition as v18, on US/India validation (labels) and test link counts."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
from street_mix import street
FILL_LIKE = {"cie", "france", "compagnie", "societe", "fils", "freres", "centre", "services", "groupe", "center", "service", "group", "company", "co", "and"}
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def abbrev(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if s[0] != l[0]: return False
    i = 0
    for ch in l:
        if i < len(s) and ch == s[i]: i += 1
    return i == len(s)
def mark(split, c, S):
    rows = np.flatnonzero(pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["src", "name_clean", "business_address"], filters=[("country", "==", c)]); r.index = rows
    sdf = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
    odf = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
    ratio = (odf + 1e-4) / (sdf.reindex(odf.index).fillna(0) + 1e-4)
    m = np.zeros(len(S), bool)
    for i, (a, b) in enumerate(zip(S.a, S.b)):
        ta, tb = toks(r.name_clean.loc[a]), toks(r.name_clean.loc[b]); dx, dy = ta - tb, tb - ta
        if len(dx) == 1 and len(dy) == 1 and ta & tb:
            u, v = next(iter(dx)), next(iter(dy))
            if sdf.get(u, 0) >= 1e-3 and sdf.get(v, 0) >= 1e-3 and ratio.get(v, 1) < 2 and not abbrev(u, v) and fuzz.ratio(u, v) < 70 and v not in FILL_LIKE:
                sa, sb = street(r.business_address.loc[a]), street(r.business_address.loc[b])
                m[i] = sa[0] == sb[0] and sa[0] != "" and fuzz.ratio(sa[1], sb[1]) >= 85
    return m
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a"); full = full.merge(oof, on="a")
for c in ("US", "India"):
    F = full[full.country == c]; ex = exclusive(F[["a", "b", "s1_id", "o_id", "p"]]); S = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    m = mark("train", c, S)
    y = np.array([(s, o) in ys for s, o in zip(S.s1_id[m], S.o_id[m])])
    print(f"validation {c}: category-swap LINKS {m.sum()} ({m.mean():.3%} of links), precision {y.mean() if len(y) else float('nan'):.3f}", flush=True)
for c in ("US", "India"):
    P = pd.read_parquet(f"backup_pred_s2_v14/pred_s2_{c}.parquet"); S = expected_f_fast(exclusive(P))
    m = mark("test", c, S.reset_index(drop=True)); print(f"test {c}: category-swap links {m.sum()} ({m.mean():.3%})", flush=True)
