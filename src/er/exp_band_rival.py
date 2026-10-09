"""Second pass of the band specialist: record-side competition measured with cross-encoder evidence.

v12 measures how strongly a rival S1 claims the same record by its stage-1 probability p1 (band AUC 0.965). Here a
light 'rival scorer' q = LightGBM(p1, record-side p1 context, cross-encoder logits) -- nearly as sharp as the band
specialist, but free of S1-side context, which is unavailable for rivals -- scores every band pair of a record:
bucket-0 pairs out-of-fold, and rival pairs (S1s outside bucket 0, er.build_competitors) using ONLY the
cross-encoders that never saw the rival's bucket (at test time nothing is in-sample, so all are used). The
record-side context is then recomputed from q (outside the band, p1 stands in for q) and appended to the v12
features for a second-pass specialist.

  python -m er.exp_band_rival --art artifacts/train --gt .../train_ground_truth.tsv --s2-tag _v9 \
      --ce <OOF CE files in test order> --rival-ce artifacts/train/rce{0..5}.parquet --save-tag _v13
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .exp_band import band_features
from .exp_band_fd import CTXC, PARAMS
from .exp_band_robust import macro
from .predict import country_rows, read_country
from .prep import load_gt
from .stage2 import context_features

# cross-encoder feature index (test order: base, base2, mdeberta, norm, hard, large) -> entity buckets it trained on
# (incl. its initialisation: base2 and hard both start from base)
INSAMPLE = {0: {2, 3}, 1: {2, 3, 4, 5}, 2: {6, 7}, 3: {2, 3}, 4: {2, 3, 4, 5}, 5: {6, 7}}
BCTX = ["rank_in_b", "b_max", "b_second", "b_margin"]
QCTX = ["q", "q_rank_in_b", "q_b_max", "q_b_second", "q_b_margin", "q_b_sum_other"]
QPARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=20)


def q_feats(ces):
    return ["p1"] + BCTX + [f"ce{j}" for j in ces]


def q_context(d: pd.DataFrame) -> pd.DataFrame:
    """d: a, b, v (q inside the band, p1 outside) for every pair of the records concerned -> record-side context."""
    d = d[["a", "b", "v"]].copy()
    gb = d.groupby("b")["v"]
    d["q_rank_in_b"] = gb.rank(ascending=False, method="first").astype(np.float32)
    d["q_b_max"] = gb.transform("max").astype(np.float32)
    s = d.sort_values("v", ascending=False).groupby("b")["v"].nth(1)
    d["q_b_second"] = d.b.map(pd.Series(s.values, index=d.loc[s.index, "b"].values)).fillna(0).astype(np.float32)
    d["q_b_margin"] = np.where(d.q_rank_in_b == 1, d.v - d.q_b_second, d.v - d.q_b_max).astype(np.float32)
    d["q_b_sum_other"] = (gb.transform("sum") - d.v).astype(np.float32)
    return d.rename(columns={"v": "q"})


SLOT = ["slot_same_src", "slot_other_src", "slot_same_mass"]


def slot_features(d: pd.DataFrame) -> pd.DataFrame:
    """d: a, b, v (q inside the band, p1 outside), src (source of b). For each pair: how many OTHER confident records
    the S1 already has from the record's own source / the other source (almost every entity has >=1 of each)."""
    d = d[["a", "b", "v", "src"]].copy()
    conf = (d.v > 0.5).astype(np.float32)
    g_same = d.assign(c=conf, m=d.v).groupby(["a", "src"])
    d["slot_same_src"] = (g_same.c.transform("sum") - conf).astype(np.float32)
    d["slot_same_mass"] = (g_same.m.transform("sum") - d.v).astype(np.float32)
    d["slot_other_src"] = (d.assign(c=conf).groupby("a").c.transform("sum") - conf - d.slot_same_src).astype(np.float32)
    return d[["a", "b"] + SLOT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", default="_hyb5")
    ap.add_argument("--s2-tag", default="_v9")
    ap.add_argument("--ce", nargs="+", required=True, help="bucket-0 OOF CE files, feature names ce0..")
    ap.add_argument("--ce-index", type=int, nargs="+", default=None, help="test-order index of each --ce file")
    ap.add_argument("--rival-ce", nargs="+", required=True, help="rival CE files rce<j>.parquet, j = test index")
    ap.add_argument("--comp-tag", default="fd")
    ap.add_argument("--save-tag", default="_v13")
    ap.add_argument("--band", type=float, nargs=2, default=[0.005, 0.995])
    ap.add_argument("--min-p", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--with-v12", action="store_true", help="also report the v12 feature set (control)")
    ap.add_argument("--pick-best", action="store_true", help="save whichever second-pass variant validates best")
    a = ap.parse_args()
    idx = a.ce_index or list(range(len(a.ce)))
    art = Path(a.art)
    oof = pd.read_parquet(art / f"oof{a.tag}.parquet")
    s2 = pd.read_parquet(art / f"oof_stage2{a.tag}{a.s2_tag}.parquet", columns=["a", "b", "p2"])
    path = art / "records.parquet"
    # 1. test-like p1 context over bucket-0 pairs + rival pairs (as er.exp_band_fd), keeping rival rows too
    band0, rivb, allrows = [], [], []
    for country, rows in country_rows(path).items():
        P = oof[oof.country == country][["a", "b", "p"]]
        if not len(P):
            continue
        sub = read_country(path, country, rows)
        ids = pd.Series(sub.entity_id.to_numpy(dtype=object), index=sub.index)
        comp = pd.read_parquet(art / f"comp_{a.comp_tag}_{country}.parquet", columns=["a", "b", "p"])
        Pn = pd.concat([P.assign(r=False), comp.assign(r=True)], ignore_index=True)
        Pn = Pn[Pn.p > a.min_p].reset_index(drop=True)
        C = context_features(Pn[["a", "b", "p"]], sub[["name_core", "addr_clean"]])
        C["rival"] = Pn.r.to_numpy()
        C["bucket"] = np.where(C.rival, pd.util.hash_array(ids.loc[C.a].to_numpy()) % 8, 0)
        inb = (C.p1 > a.band[0]) & (C.p1 < a.band[1])
        band0.append(C[inb & ~C.rival])
        rivb.append(C[inb & C.rival])
        allrows.append(C[["a", "b", "p1"]].assign(inband=inb.to_numpy(), src=sub.src.to_numpy()[np.searchsorted(rows, C.b.to_numpy())]))
        print(f"{country}: {inb.sum()} band pairs ({(inb & C.rival).sum()} rival)", flush=True)
    Cn, R, A = (pd.concat(x, ignore_index=True) for x in (band0, rivb, allrows))
    # 2. bucket-0 band rows with the full v12 feature set
    F = band_features(a.art, a.tag, Cn)
    D = Cn.drop(columns=["rival", "bucket"]).merge(
        F.drop(columns=[c for c in ("y", "country", "s1_id", "o_id") if c in F]), on=["a", "b"])
    D = D.merge(oof[["a", "b", "s1_id", "o_id", "country"]], on=["a", "b"])
    for j, p in zip(idx, a.ce):
        D = D.merge(pd.read_parquet(p).rename(columns={"ce": f"ce{j}"}), on=["a", "b"], how="left")
    ys = set(map(tuple, load_gt(a.gt)[["s1", "other"]].to_numpy()))
    D["y"] = np.fromiter(((s, o) in ys for s, o in zip(D.s1_id, D.o_id)), bool, len(D))
    for j, p in zip(idx, a.rival_ce):
        assert Path(p).name == f"rce{j}.parquet", (j, p)
        R = R.merge(pd.read_parquet(p).rename(columns={"ce": f"ce{j}"}), on=["a", "b"], how="left")
    fold = (pd.util.hash_array(D.s1_id.to_numpy(dtype=object)) // 7 % a.folds).astype(int)
    y = D.y.to_numpy()
    # 3. rival scorer q: out-of-fold on bucket 0; rivals scored with their bucket's clean cross-encoders only
    qf = q_feats(idx)
    D["q"] = np.nan
    its = []
    for k in range(a.folds):
        tr, va = fold != k, fold == k
        m = lgb.train(QPARAMS, lgb.Dataset(D.loc[tr, qf].to_numpy(np.float32), y[tr], feature_name=qf), 3000,
                      valid_sets=[lgb.Dataset(D.loc[va, qf].to_numpy(np.float32), y[va], feature_name=qf)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        D.loc[va, "q"] = m.predict(D.loc[va, qf].to_numpy(np.float32), num_iteration=m.best_iteration)
        its.append(m.best_iteration)
    n_it = int(np.mean(its) * 1.1)
    from sklearn.metrics import roc_auc_score
    print(f"rival scorer q ({len(qf)} feats): OOF band AUC {roc_auc_score(y, D.q):.4f} (p1 {roc_auc_score(y, D.p1):.4f})",
          flush=True)
    q_full = lgb.train(QPARAMS, lgb.Dataset(D[qf].to_numpy(np.float32), y, feature_name=qf), n_it)
    q_full.save_model(str(art / f"q_model{a.save_tag}.txt"))
    R["q"] = np.nan
    for buckets in ({1}, {2, 3}, {4, 5}, {6, 7}):
        clean = [j for j in idx if not (INSAMPLE[j] & buckets)]
        f = q_feats(clean)
        mq = q_full if len(clean) == len(idx) else lgb.train(
            QPARAMS, lgb.Dataset(D[f].to_numpy(np.float32), y, feature_name=f), n_it)
        sel = R.bucket.isin(buckets).to_numpy()
        R.loc[sel, "q"] = mq.predict(R.loc[sel, f].to_numpy(np.float32))
        print(f"  rivals in buckets {sorted(buckets)}: {sel.sum()} pairs scored with CEs {clean}", flush=True)
    # 4. q-context over every pair of the concerned records (q inside the band, p1 outside)
    qv = pd.concat([D[["a", "b", "q"]], R[["a", "b", "q"]]], ignore_index=True)
    A = A.merge(qv, on=["a", "b"], how="left")
    A["v"] = np.where(A.inband & A.q.notna(), A.q, A.p1)
    Q = q_context(A[A.b.isin(set(D.b))])
    D = D.drop(columns=["q"]).merge(Q[["a", "b"] + QCTX], on=["a", "b"], how="left")
    assert not D[QCTX].isna().any().any()
    Sf = slot_features(A[A.a.isin(set(D.a))])
    D = D.merge(Sf, on=["a", "b"], how="left")
    # 5. second-pass specialist: v12 features + q-context, same folds
    base = [c for c in D.columns if c not in ("a", "b", "s1_id", "o_id", "country", "y") and c not in QCTX + SLOT]
    gt = load_gt(a.gt)
    samp = set(oof.s1_id)
    truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
    all_s1 = samp | set(truth)
    s1_country = oof.drop_duplicates("s1_id").set_index("s1_id").country
    by_c = {c: {s for s in all_s1 if s1_country.get(s) == c} for c in D.country.unique()}
    res, final_its = {}, {}
    variants = [("+ q-context", base + QCTX), ("+ q-context + slots", base + QCTX + SLOT)]
    if a.with_v12: variants.insert(0, ("v12 features", base))
    for name, feats in variants:
        X = D[feats].to_numpy(np.float32)
        pb, it2 = np.zeros(len(D), np.float32), []
        for k in range(a.folds):
            tr, va = fold != k, fold == k
            m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=feats), 5000,
                          valid_sets=[lgb.Dataset(X[va], y[va], feature_name=feats)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            pb[va] = m.predict(X[va], num_iteration=m.best_iteration)
            it2.append(m.best_iteration)
        keys = D[["a", "b"]].assign(pb=pb)
        res[name] = {"all": macro(keys, oof, s2, truth, all_s1),
                     **{c: macro(keys[(D.country == c).to_numpy()], oof[oof.country == c], s2, truth, by_c[c])
                        for c in ("US", "India")}}
        final_its[name] = it2
        print(f"{name}: macro F0.5 {res[name]['all']:.5f} | US {res[name]['US']:.5f} | India {res[name]['India']:.5f}"
              f" | AUC {roc_auc_score(y, pb):.4f}", flush=True)
        if name.startswith("+ q-context"):
            imp = pd.Series(m.feature_importance("gain"), feats)
            print("top:", (imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_dict(), flush=True)
    json.dump(res, open(art / f"band_rival_metrics{a.save_tag}.json", "w"), indent=1)
    best = max(("+ q-context", "+ q-context + slots"), key=lambda k: res[k]["all"]) if a.pick_best else "+ q-context"
    feats = base + QCTX + (SLOT if best.endswith("slots") else [])
    print(f"final model: {best}", flush=True)
    m = lgb.train(PARAMS, lgb.Dataset(D[feats].to_numpy(np.float32), y, feature_name=feats),
                  int(np.mean(final_its[best]) * 1.1))
    m.save_model(str(art / f"band_model{a.save_tag}.txt"))
    print("saved second-pass band model + q model", flush=True)


if __name__ == "__main__":
    main()
