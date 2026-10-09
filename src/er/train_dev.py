"""End-to-end on a labelled split: block -> features -> LightGBM (entity-grouped CV) -> decide -> macro F0.5.

  python -m er.train_dev --art artifacts/dev10 --gt data/dev10/ground_truth.tsv --k-other 10 --k-s1 10
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import generate, pivot_scores
from .decide import exclusive, expected_f_select, macro_f05, to_lists
from .features import pair_features
from .prep import load_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--k-other", type=int, default=5)
    ap.add_argument("--k-s1", type=int, default=5)
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()
    T = time.time()
    rec = pd.read_parquet(f"{a.art}/records.parquet")
    ids = rec.entity_id.to_numpy()
    fp = f"{a.art}/feats_k{a.k_other}_{a.k_s1}.parquet"
    try:
        X = pd.read_parquet(fp)
    except FileNotFoundError:
        cand = pivot_scores(generate(rec, a.k_other, a.k_s1, threads=22, retrievers=("all",)))
        print(f"cand {len(cand)} pairs ({len(cand) / (rec.src == 1).sum():.1f}/S1) {time.time() - T:.0f}s", flush=True)
        X = pair_features(rec, cand, workers=22)
        # context features: how this pair's retrieval score compares with the b's / a's alternatives
        for key, nm in (("b", "b"), ("a", "a")):
            g = X.groupby(key)["score_all_b2a"]
            X[f"{nm}_ncand"] = X.groupby(key)["a"].transform("size").astype(np.float32)
            X[f"{nm}_score_gap"] = (X["score_all_b2a"].fillna(0) - g.transform("max").fillna(0)).astype(np.float32)
        X.to_parquet(fp)
        print(f"features {X.shape} {time.time() - T:.0f}s", flush=True)

    gt = load_gt(a.gt)
    g = set(zip(gt.s1, gt.other))
    y = np.fromiter(((ids[i], ids[j]) in g for i, j in zip(X.a.to_numpy(), X.b.to_numpy())), bool, len(X))
    print(f"pairs {len(X)} pos {y.sum()} recall ceiling {y.sum() / len(g):.4f}")
    feat_cols = [c for c in X.columns if c not in ("a", "b")]
    fold = (pd.util.hash_array(ids[X.a.to_numpy()].astype(object)) % a.folds).astype(int)
    oof = np.zeros(len(X), np.float32)
    XM = X[feat_cols].to_numpy(np.float32)  # float32 matrix: LightGBM won't upcast/copy to float64
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=22)
    for k in range(a.folds):
        tr, va = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(XM[tr], y[tr], feature_name=feat_cols, free_raw_data=True), 2000,
                      valid_sets=[lgb.Dataset(XM[va], y[va], feature_name=feat_cols)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(XM[va], num_iteration=m.best_iteration)
        print(f"fold {k} it {m.best_iteration} {time.time() - T:.0f}s", flush=True)
        its = getattr(main, "_its", []) + [m.best_iteration]
        main._its = its
    full = lgb.train(params, lgb.Dataset(XM, y, feature_name=feat_cols), int(np.mean(main._its) * 1.1))
    full.save_model(f"{a.art}/model.txt")
    imp = pd.Series(m.feature_importance("gain"), feat_cols).sort_values(ascending=False)
    print("top features:", (imp / imp.sum()).head(15).round(3).to_dict())
    X["p"] = oof
    X[["a", "b", "p"]].assign(y=y).to_parquet(f"{a.art}/oof.parquet")

    s1 = ids[(rec.src == 1).to_numpy()]
    truth = {}
    for s, o in zip(gt.s1, gt.other):
        truth.setdefault(s, set()).add(o)
    P = X[["a", "b", "p"]]
    for t in (0.3, 0.5, 0.7, 0.8, 0.9):
        print(f"thr {t}: F0.5 {macro_f05(to_lists(P[P.p >= t], ids), truth, s1):.4f}"
              f" | exclusive {macro_f05(to_lists(exclusive(P)[lambda d: d.p >= t], ids), truth, s1):.4f}")
    ex = exclusive(P)
    print(f"exclusive + expected-F: {macro_f05(to_lists(expected_f_select(ex), ids), truth, s1):.4f}")
    print(f"total {time.time() - T:.0f}s")


if __name__ == "__main__":
    main()
