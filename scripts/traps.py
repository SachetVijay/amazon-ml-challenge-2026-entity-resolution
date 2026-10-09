"""Singleton traps: S1 with exactly ONE link, the link being a renamed record (no shared name word) at the same
street+number. Test US/France (v18 France links) per S1; validation US/India precision (labels)."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.normalize import LEGAL
from street_mix import street
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
def classify(r, a, b, P):
    ta, tb = toks(r.name_clean.loc[a]), toks(r.name_clean.loc[b])
    if ta & tb or r.addr_clean.loc[b] == "": return None
    sa, sb = street(r.business_address.loc[a]), street(r.business_address.loc[b])
    if not (sa[0] == sb[0] and sa[0] != "" and fuzz.ratio(sa[1], sb[1]) >= 85): return None
    return "renamed@same-address"
def load(split, c):
    rows = np.flatnonzero(pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["entity_id", "src", "name_clean", "business_address", "addr_clean"], filters=[("country", "==", c)]); r.index = rows
    return r
def report(tag, r, links, P, n1, truth=None):
    g = links.groupby("a").b.apply(list)
    one = g[g.str.len() == 1]
    cand = P[P.p > 0.1].groupby("a").size()
    hits, nosib, y = 0, 0, []
    for a, bs in one.items():
        if classify(r, a, bs[0], P):
            hits += 1
            if cand.get(a, 0) <= 1: nosib += 1
            if truth is not None: y.append(truth(a, bs[0]))
    msg = f"{tag:10s}: 1-link S1 {len(one)/n1:.4f}/S1 | ...with renamed@same-address link {hits/n1:.4f}/S1 ({hits}) | of which no other candidate p>0.1: {nosib/n1:.4f}/S1"
    if y: msg += f" | precision {np.mean(y):.3f}"
    print(msg, flush=True)
for c in ("US", "France"):
    r = load("test", c)
    if c == "France":
        m = pd.read_csv("output_v18.tsv", sep="\t", dtype=str, keep_default_na=False)
        idx = pd.Series(r.index, index=r.entity_id.astype(object)); fr = set(r.entity_id[r.src == 1].astype(object))
        L = pd.DataFrame([(idx[s], idx[o]) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) if s in fr for o in l.split(",") if o], columns=["a", "b"])
        P = pd.read_parquet("backup_pred_s2_v16/pred_s2_France.parquet")
    else:
        P = pd.read_parquet(f"backup_pred_s2_v14/pred_s2_{c}.parquet"); L = expected_f_fast(exclusive(P))[["a", "b"]]
    report(f"test {c}", r, L, P, (r.src == 1).sum())
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a"); full = full.merge(oof, on="a")
for c in ("US", "India"):
    r = load("train", c); F = full[full.country == c]
    L = expected_f_fast(exclusive(F[["a", "b", "p"]]))[["a", "b"]]
    ids = r.entity_id.astype(object)
    report(f"val {c}", r, L, F, F.s1_id.nunique(), truth=lambda a, b: (ids[a], ids[b]) in ys)
