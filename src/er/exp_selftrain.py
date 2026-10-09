"""Unseen-country adaptation by self-training, evaluated with a held-out TRAIN country as a proxy for France.

Protocol (no target labels are used for training, stopping or thresholds):
  1. train on source country S with a fixed number of rounds
  2. score target country T; keep confident pseudo-labels:
       positive = p >= hi and the pair is the record's best S1 (exclusive winner)
       negative = p <= lo
  3. retrain on S + pseudo-labelled T (target weight w), repeat for R rounds
  4. report T's macro F0.5 after each round (true labels used only for this report)

  python -m er.exp_selftrain --art artifacts/train --gt .../train_ground_truth.tsv --source US --target India
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decide import exclusive, expected_f_fast, macro_f05
from .prep import load_gt


def f05(ev, truth, s1_eval):
    ex = exclusive(ev)
    sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    lists = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
    return macro_f05(lists, truth, s1_eval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--source", default="US")
    ap.add_argument("--target", default="India")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--lo", type=float, default=0.03)
    ap.add_argument("--w", type=float, default=1.0)
    a = ap.parse_args()
    S = pd.read_parquet(f"{a.art}/feats_{a.source}.parquet")
    T = pd.read_parquet(f"{a.art}/feats_{a.target}.parquet")
    meta = ["a", "b", "y", "country", "s1_id", "o_id"]
    cols = [c for c in S.columns if c not in meta]
    XS, yS = S[cols].to_numpy(np.float32), S.y.to_numpy()
    XT = T[cols].to_numpy(np.float32)
    gt = load_gt(a.gt)
    tset = set(T.s1_id.unique())
    gt = gt[gt.s1.isin(tset)]
    truth = gt.groupby("s1").other.agg(set).to_dict()
    # evaluation population: every sampled target S1 that has candidates (singletons without candidates
    # score 1.0 under any rule and would only dilute differences)
    s1_eval = np.array(sorted(tset))
    params = dict(objective="binary", learning_rate=0.06, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=22)
    m = lgb.train(params, lgb.Dataset(XS, yS), a.iters)
    p = m.predict(XT)
    ev = T[["a", "b", "s1_id", "o_id"]].assign(p=p)
    print(f"round 0 (source only): {a.target} F0.5 {f05(ev, truth, s1_eval):.4f}", flush=True)
    for r in range(1, a.rounds + 1):
        win = np.zeros(len(T), bool)
        win[ev.groupby("b")["p"].idxmax().to_numpy()] = True
        pos, neg = (p >= a.hi) & win, p <= a.lo
        keep = pos | neg
        acc = (T.y.to_numpy()[keep] == pos[keep]).mean()   # diagnostic only
        print(f"  pseudo-labels: {pos.sum()} pos, {neg.sum()} neg ({keep.mean():.1%} of target pairs), "
              f"label accuracy {acc:.4f}", flush=True)
        X = np.vstack([XS, XT[keep]])
        y = np.concatenate([yS, pos[keep]])
        w = np.concatenate([np.ones(len(yS)), np.full(keep.sum(), a.w)])
        m = lgb.train(params, lgb.Dataset(X, y, weight=w), a.iters)
        p = m.predict(XT)
        ev = T[["a", "b", "s1_id", "o_id"]].assign(p=p)
        print(f"round {r}: {a.target} F0.5 {f05(ev, truth, s1_eval):.4f}", flush=True)


if __name__ == "__main__":
    main()
