"""Per-S1 true F0.5 vs model-expected F0.5 (MC) on validation, grouped by S1 name multiplicity k (full density).
Then: predicted France F = sum over k of France share(k) * (validation true F given k)  [and with E[F]-bias correction]."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import BETA2, exclusive, expected_f_fast, f05
from er.prep import load_gt
rng = np.random.default_rng(0)
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
samp = set(full.s1_id); truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
ex = exclusive(full[["a", "b", "s1_id", "o_id", "p"]]); sel = expected_f_fast(ex)
ex["chosen"] = ex.set_index(["a", "b"]).index.isin(sel.set_index(["a", "b"]).index)
pred = ex[ex.chosen].groupby("s1_id").o_id.agg(set).to_dict()
ef = {}
for s, g in ex.groupby("s1_id"):
    p, ch = g.p.to_numpy(), g.chosen.to_numpy()
    Y = rng.random((64, len(p))) < p; nt, m = Y.sum(1), ch.sum()
    if m == 0: f = (nt == 0).astype(float)
    else:
        tp = Y[:, ch].sum(1); prec = tp / m; rec = np.divide(tp, nt, out=np.zeros(64), where=nt > 0)
        f = np.where(tp > 0, (1 + BETA2) * prec * rec / (BETA2 * prec + rec + 1e-12), 0.0)
    ef[s] = f.mean()
allS = sorted(samp | set(truth))
T = pd.DataFrame({"s1": allS})
T["F"] = [f05(pred.get(s, set()), truth.get(s, set())) for s in allS]
T["EF"] = T.s1.map(ef).fillna(1.0)
# name multiplicity (full density) per country
rec = pd.read_parquet("artifacts/train/records.parquet", columns=["entity_id", "src", "country", "name_core"]); rec = rec[rec.src == 1]
k = rec.groupby(["country", "name_core"]).entity_id.transform("size"); kmap = dict(zip(rec.entity_id, k))
T["k"] = T.s1.map(kmap).fillna(1).clip(upper=5).astype(int)
g = T.groupby("k").agg(n=("F", "size"), trueF=("F", "mean"), expF=("EF", "mean")); g["bias"] = g.expF - g.trueF
print(g.round(4).to_string())
tst = pd.read_parquet("artifacts/test/records.parquet", columns=["src", "country", "name_core"]); tst = tst[tst.src == 1]
tst["k"] = tst.groupby(["country", "name_core"]).name_core.transform("size").clip(upper=5)
for c, h in tst.groupby("country"):
    w = h.k.value_counts(normalize=True).reindex(g.index).fillna(0)
    print(f"{c:7s}: k-share {w.round(3).to_dict()} -> predicted F (validation true-F by k, reweighted) {float((w * g.trueF).sum()):.4f}")
# ---- address co-location: S1s sharing the S1's (number, street) key, full density
sys.path.insert(0, "scripts")
from street_mix import street
def addr_k(split, ids=None):
    r = pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["entity_id", "src", "country", "business_address"]); r = r[r.src == 1]
    kk = [street(x) for x in r.business_address.astype(object)]
    r["key"] = [f"{a}|{b}" if a else None for a, b in kk]
    r["ka"] = r.groupby(["country", "key"]).entity_id.transform("size").fillna(1)
    r.loc[r.key.isna(), "ka"] = 1
    return r
tr = addr_k("train"); T["ka"] = T.s1.map(dict(zip(tr.entity_id, tr.ka))).fillna(1).clip(upper=4).astype(int)
g2 = T.groupby(["k", "ka"]).agg(n=("F", "size"), trueF=("F", "mean"))
print(T.groupby("ka").agg(n=("F", "size"), trueF=("F", "mean"), expF=("EF", "mean")).round(4).to_string())
te = addr_k("test")
tn = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "src", "country", "name_core"]); tn = tn[tn.src == 1]
tn["k"] = tn.groupby(["country", "name_core"]).name_core.transform("size").clip(upper=5)
te = te.merge(tn[["entity_id", "k"]], on="entity_id"); te["ka"] = te.ka.clip(upper=4).astype(int)
for c, h in te.groupby("country"):
    w = h.groupby(["k", "ka"]).size() / len(h)
    pf = float((w * g2.trueF.reindex(w.index)).dropna().sum() / w[g2.trueF.reindex(w.index).notna()].sum())
    print(f"{c:7s}: co-location share {h.ka.value_counts(normalize=True).sort_index().round(3).to_dict()} -> predicted F by (k, co-location) {pf:.4f}")
