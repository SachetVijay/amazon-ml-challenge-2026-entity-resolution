"""Stage-2 experiment: does cluster-aware re-scoring improve OOF macro F0.5 over stage 1?

Uses stage-1 OOF probabilities (artifacts/train/oof<tag>.parquet) and the SAME entity-hash folds, so
stage-2 is trained/evaluated without leakage.

  python -m er.exp_stage2 --art artifacts/train --gt .../train_ground_truth.tsv [--tag _hyb5]
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decide import exclusive, expected_f_fast, macro_f05
from .predict import country_rows, read_country
from .prep import load_gt
from .stage2 import context_features


def evaluate(ev, truth, s1_eval, col):
    ex = exclusive(ev, col)
    out = {}
    for t in (0.65, 0.75):
        sel = ex[ex[col] >= t]
        out[f"thr{t}"] = macro_f05({s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}, truth, s1_eval)
    sel = expected_f_fast(ex, col).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    out["expF"] = macro_f05({s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}, truth, s1_eval)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--ce", nargs="*", default=None, help="cross-encoder score parquets (a, b, ce); 2nd -> ce2")
    ap.add_argument("--consensus", action="store_true", help="add entity-consensus embedding features")
    ap.add_argument("--save-tag", default=None, help="suffix for saved stage-2 model/oof (default _ce)")
    a = ap.parse_args()
    oof = pd.read_parquet(f"{a.art}/oof{a.tag}.parquet")
    path = f"{a.art}/records.parquet"
    parts = []
    for country, rows in country_rows(path).items():
        P = oof[oof.country == country]
        if len(P) == 0:
            continue
        sub = read_country(path, country, rows)[["name_core", "addr_clean"]]
        ctx = context_features(P, sub)
        if a.consensus:
            from .biencoder import load_emb
            from .stage2 import consensus_features
            ctx = consensus_features(ctx, load_emb(a.art, country), rows, col="p1")
        parts.append(ctx.merge(P[["a", "b", "s1_id", "o_id", "country"]], on=["a", "b"]))
        del sub
    D = pd.concat(parts, ignore_index=True)
    for j, path_ce in enumerate(a.ce or []):
        ce = pd.read_parquet(path_ce).rename(columns={"ce": "ce" if j == 0 else f"ce{j + 1}"})
        D = D.merge(ce, on=["a", "b"], how="left")
        print(f"cross-encoder {j + 1} scores for {ce.shape[0]} pairs", flush=True)
    ys = {(s, o) for s, o in zip(*load_gt(a.gt)[["s1", "other"]].to_numpy().T)}
    D["y"] = np.fromiter(((s, o) in ys for s, o in zip(D.s1_id, D.o_id)), bool, len(D))
    feats = [c for c in D.columns if c not in ("a", "b", "s1_id", "o_id", "country", "y")]
    print("stage-2 features:", feats, flush=True)
    fold = (pd.util.hash_array(D.s1_id.to_numpy(dtype=object)) // 7 % a.folds).astype(int)  # = stage-1 folds
    X, y = D[feats].to_numpy(np.float32), D.y.to_numpy()
    p2 = np.zeros(len(D), np.float32)
    its = []
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=22)
    for k in range(a.folds):
        tr, va = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=feats), 2000,
                      valid_sets=[lgb.Dataset(X[va], y[va], feature_name=feats)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p2[va] = m.predict(X[va], num_iteration=m.best_iteration)
        its.append(m.best_iteration)
    imp = pd.Series(m.feature_importance("gain"), feats)
    print("stage-2 importance:", (imp / imp.sum()).sort_values(ascending=False).round(3).to_dict())
    D["p2"] = p2
    gt = load_gt(a.gt)
    gt = gt[gt.s1.isin(set(D.s1_id))]
    truth = gt.groupby("s1").other.agg(set).to_dict()
    # evaluation population = all sampled S1 of these countries (includes candidate-less singletons)
    s1_eval = np.array(sorted(set(oof.s1_id) | set(truth)))
    for nm, col in (("stage 1", "p1"), ("stage 2", "p2")):
        print(nm, {k: round(v, 4) for k, v in evaluate(D, truth, s1_eval, col).items()}, flush=True)
    D[["a", "b", "s1_id", "o_id", "country", "p1", "p2", "y"]].to_parquet(
        f"{a.art}/oof_stage2{a.tag}{a.save_tag or ('_ce' if a.ce else '')}.parquet")
    m_full = lgb.train(params, lgb.Dataset(X, y, feature_name=feats), int(np.mean(its) * 1.1))
    m_full.save_model(f"{a.art}/stage2{a.tag}{a.save_tag or ('_ce' if a.ce else '')}.txt")
    print("saved stage-2 model; features:", feats)


if __name__ == "__main__":
    main()
