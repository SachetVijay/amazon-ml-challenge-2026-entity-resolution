import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from street_mix import street
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def rel(x, y):
    if x.startswith(y) or x.endswith(y) or y.startswith(x) or y.endswith(x): return "drop/insert"
    try: return "adjacent<=3" if abs(int(x) - int(y)) <= 3 else "other"
    except ValueError: return "other"
def scan(split, c, L, P=None, ys=None, ids=None):
    rows = np.flatnonzero(pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["entity_id", "src", "name_clean", "business_address"], filters=[("country", "==", c)]); r.index = rows
    out = {}
    for a, b in L:
        ta, tb = toks(r.name_clean.loc[a]), toks(r.name_clean.loc[b])
        if not ta or tb - ta: continue          # record name adds nothing to the S1 name
        sa, sb = street(r.business_address.loc[a]), street(r.business_address.loc[b])
        if sa[0] and sb[0] and sa[0] != sb[0] and fuzz.ratio(sa[1], sb[1]) >= 85:
            k = rel(sa[0], sb[0])
            out.setdefault(k, []).append((a, b))
    n1 = (r.src == 1).sum() if ys is None else len(set(a for a, _ in L))
    msg = f"{split} {c:7s}: " + " | ".join(f"{k} {len(v)} ({len(v)/n1:.4f}/S1)" for k, v in sorted(out.items()))
    if ys is not None:
        idv = r.entity_id.astype(object)
        msg += " || precision " + ", ".join(f"{k} {np.mean([(idv[a], idv[b]) in ys for a, b in v]):.3f}" for k, v in sorted(out.items()))
    print(msg, flush=True)
    return out, r
m = pd.read_csv("output_v18.tsv", sep="\t", dtype=str, keep_default_na=False)
for c in ("France", "US"):
    rows = np.flatnonzero(pd.read_parquet("artifacts/test/records.parquet", columns=["country"]).country.to_numpy() == c)
    ent = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "src"], filters=[("country", "==", c)]); ent.index = rows
    idx = pd.Series(ent.index, index=ent.entity_id.astype(object)); s1 = set(ent.entity_id[ent.src == 1].astype(object))
    L = [(idx[s], idx[o]) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) if s in s1 for o in l.split(",") if o]
    out, r = scan("test", c, L)
    if c == "France":
        pd.to_pickle(out, "/tmp/fr_shadow_links.pkl")
        for _, (a, b) in zip(range(6), out.get("other", [])[:6]):
            print(f"   S1 {r.name_clean.loc[a]} | {r.business_address.loc[a]}\n   REC {r.name_clean.loc[b]} | {r.business_address.loc[b]}")
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a"); full = full.merge(oof, on="a")
for c in ("US", "India"):
    F = full[full.country == c]; L = list(map(tuple, expected_f_fast(exclusive(F[["a", "b", "p"]]))[["a", "b"]].to_numpy()))
    scan("train", c, L, ys=ys)
