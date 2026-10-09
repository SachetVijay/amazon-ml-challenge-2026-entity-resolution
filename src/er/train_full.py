"""Full-density training: block the WHOLE train split exactly like test, then featurise/label only a
hash-sampled subset of S1 entities. Features (ranks, score gaps, name frequency) then have the same
distribution as at test time, and the OOF F0.5 on the sample is an honest estimate of leaderboard F0.5.

  python -m er.train_full --split-dir student_resource/dataset/train --prefix train --art artifacts/train \
      --gt student_resource/dataset/train/train_ground_truth.tsv --sample-mod 8
"""
import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decide import exclusive, expected_f_fast, macro_f05
from .candidates import HYBRID_CTX, hybrid
from .features import pair_features
from .predict import CTX, country_candidates, country_rows, read_country
from .prep import load_gt, load_split

def build(a):
    """Per-country: block the full split, keep pairs whose S1 is in the hash sample, featurise + label.
    Returns the pieces needed for evaluation (sampled S1 ids per country, ground-truth pairs)."""
    path = Path(a.art) / "records.parquet"
    if not path.exists():
        load_split(a.split_dir, a.prefix, a.art, workers=22, arrow=True)
    gt_all = load_gt(a.gt)
    gt_all = gt_all[(pd.util.hash_array(gt_all.s1.to_numpy(dtype=object)) % a.sample_mod) == 0]  # sampled S1 only
    samp_ids = {}
    for country, rows in country_rows(path).items():
        sub = read_country(path, country, rows)
        ids = pd.Series(sub.entity_id.to_numpy(dtype=object), index=sub.index)
        s1 = ids[sub.src.to_numpy() == 1]
        samp_ids[country] = s1[(pd.util.hash_array(s1.to_numpy()) % a.sample_mod) == 0]
        fp = Path(a.art) / f"feats{a.tag}_{country}.parquet"
        if fp.exists():
            continue
        T = time.time()
        keep = list(map(int, samp_ids[country].index))
        cp = Path(a.art) / f"cand_{country}_k{a.k_other}_{a.k_s1}.parquet"
        if cp.exists():  # read only the sampled S1s' rows (context columns were computed on the full set)
            cand = pd.read_parquet(cp, filters=[("a", "in", keep)])
        else:
            cand = country_candidates(sub, country, a)
            cand = cand[cand.a.isin(keep)]
        if a.hybrid:
            cand = hybrid(cand, a.art, country, rows, a.dense_k, keep_a=keep)
        ctx = CTX + (HYBRID_CTX if a.hybrid else [])
        pos = set(map(tuple, gt_all[gt_all.s1.isin(set(samp_ids[country]))][["s1", "other"]].to_numpy()))
        X = pair_features(sub, cand, workers=22).merge(cand[ctx], on=["a", "b"])
        X["y"] = np.fromiter(((ids[i], ids[j]) in pos for i, j in zip(X.a.to_numpy(), X.b.to_numpy())), bool, len(X))
        X["country"] = country
        X["s1_id"] = ids.loc[X.a].to_numpy()
        X["o_id"] = ids.loc[X.b].to_numpy()
        X.to_parquet(fp)
        print(f"{country}: {len(X)} pairs, {X.y.sum()} pos, {time.time() - T:.0f}s", flush=True)
        del sub, cand, X, ids
        gc.collect()
    all_samp = set().union(*[set(v) for v in samp_ids.values()])
    gt = gt_all[gt_all.s1.isin(all_samp)]
    return samp_ids, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sample-mod", type=int, default=8)
    ap.add_argument("--k-other", type=int, default=5)
    ap.add_argument("--k-s1", type=int, default=5)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--drop", nargs="*", default=[], help="feature columns to exclude (ablations)")
    ap.add_argument("--hybrid", action="store_true", help="sparse ∪ bi-encoder candidates + dense features")
    ap.add_argument("--dense-k", type=int, default=5)
    ap.add_argument("--holdout-country", default=None, help="train without it, report F0.5 on it (unseen-country sim)")
    a = ap.parse_args()
    a.tag = f"_hyb{a.dense_k}" if a.hybrid else ""
    samp_ids, gt = build(a)
    n_pos_gt = len(gt)
    X = pd.concat([pd.read_parquet(Path(a.art) / f"feats{a.tag}_{c}.parquet") for c in samp_ids], ignore_index=True)
    print(f"pairs {len(X)} pos {X.y.sum()} blocking recall on sample {X.y.sum() / n_pos_gt:.4f}", flush=True)
    feat_cols = [c for c in X.columns if c not in ("a", "b", "y", "country", "s1_id", "o_id") and c not in a.drop]
    y = X.y.to_numpy()
    XM = X[feat_cols].to_numpy(np.float32)
    X = X[["a", "b", "y", "country", "s1_id", "o_id"]]  # free the feature columns (XM holds them)
    gc.collect()
    params = dict(objective="binary", learning_rate=0.06, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=22)
    if a.holdout_country:
        fold = np.where(X.country.to_numpy() == a.holdout_country, 0, 1)
        folds = [0]
    else:
        fold = (pd.util.hash_array(X.s1_id.to_numpy(dtype=object)) // 7 % a.folds).astype(int)
        folds = range(a.folds)
    oof = np.full(len(X), np.nan, np.float32)
    its = []
    for k in folds:
        tr, va = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(XM[tr], y[tr], feature_name=feat_cols), 3000,
                      valid_sets=[lgb.Dataset(XM[va], y[va], feature_name=feat_cols)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(XM[va], num_iteration=m.best_iteration)
        its.append(m.best_iteration)
        print(f"fold {k}: it {m.best_iteration}", flush=True)
    imp = pd.Series(m.feature_importance("gain"), feat_cols).sort_values(ascending=False)
    print("top features:", (imp / imp.sum()).head(12).round(3).to_dict())

    ev = X.loc[~np.isnan(oof), ["a", "b", "country", "s1_id", "o_id"]].assign(p=oof[~np.isnan(oof)])
    truth = {}
    for s, o in zip(gt.s1, gt.other):
        truth.setdefault(s, set()).add(o)
    s1_eval = np.concatenate([v.to_numpy() for c, v in samp_ids.items()
                              if not a.holdout_country or c == a.holdout_country])
    ex = exclusive(ev)
    lists = lambda d: {s: set(g) for s, g in d.groupby("s1_id")["o_id"]}  # noqa: E731
    for t in (0.5, 0.65, 0.75, 0.85):
        print(f"OOF exclusive thr {t}: F0.5 {macro_f05(lists(ex[ex.p >= t]), truth, s1_eval):.4f}")
    for w in (1.0, 1.5):
        sel = expected_f_fast(ex, empty_weight=w).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
        print(f"OOF exclusive+expectedF w={w}: F0.5 {macro_f05(lists(sel), truth, s1_eval):.4f}")
    for c, v in samp_ids.items():
        if not a.holdout_country or c == a.holdout_country:
            print(f"  {c}: thr .75 F0.5 {macro_f05(lists(ex[ex.p >= .75]), truth, v.to_numpy()):.4f}")
    ev.to_parquet(Path(a.art) / f"oof{a.tag}{'_ho_' + a.holdout_country if a.holdout_country else ''}.parquet")
    if not a.holdout_country:
        full = lgb.train(params, lgb.Dataset(XM, y, feature_name=feat_cols), int(np.mean(its) * 1.1))
        full.save_model(str(Path(a.art) / f"model{a.tag}.txt"))
        print("saved model")


if __name__ == "__main__":
    main()
