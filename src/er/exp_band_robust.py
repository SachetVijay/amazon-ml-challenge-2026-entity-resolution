"""Domain-robust band specialist: monotone constraints + seed bagging, selected by leave-one-country-out.

France has no labels; its proxy is an unseen *training* country. For each variant we report:
  in-country OOF macro F0.5 (bucket-0 entity folds, both countries) and
  LOCO: train on US band pairs -> F0.5 on India, and India -> US.
Monotone constraints: a higher similarity / cross-encoder score can never lower the match probability, which
forbids brittle country-specific interactions (e.g. "high name similarity is bad when X").

  python -m er.exp_band_robust --art artifacts/train --gt .../train_ground_truth.tsv --s2-tag _v9 \
      --ce artifacts/train/ce0_v9_oof.parquet ... --save-tag _v11r
"""
import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decide import exclusive, expected_f_fast, f05
from .exp_band import band_features
from .predict import country_rows, read_country
from .prep import load_gt
from .stage2 import context_features

INC = re.compile(r"^(ce\d+|p1|dense_b2a|dense_a2b|dense_cos|b_margin|b_max|a_max|sib_name|sib_addr|sib_p|"
                 r"nc_.*|nk_.*|ns_.*|nsp_.*|skel_.*|addr_ratio|addr_tset|addr_pratio|addrw_tsort|name_jac|name_cont|"
                 r"skel_jac|addr_jac|addr_cont|addrskel_jac|num_jac|firstnum_eq|num_partial|name_nospace_eq|"
                 r"loc_name_jac|loc_name_cont|loc_name_eq|loc_name_ratio|loc_name_tsort|score_all_.*|"
                 r"dense_b_gap|dense_a_gap|cos_b_gap|cos_a_gap|b_score_gap|a_score_gap)$")
DEC = re.compile(r"^(num_conflict|rank_all_.*|dense_rank_.*|rank_in_a|rank_in_b|cos_a_rank|b_second)$")


def build(a):
    oof = pd.read_parquet(Path(a.art) / f"oof{a.tag}.parquet")
    path = Path(a.art) / "records.parquet"
    ctxs = []
    for country, rows in country_rows(path).items():
        P = oof[oof.country == country]
        if len(P):
            ctxs.append(context_features(P, read_country(path, country, rows)[["name_core", "addr_clean"]]))
    C = pd.concat(ctxs, ignore_index=True)
    band = C[(C.p1 > 0.005) & (C.p1 < 0.995)]
    F = band_features(a.art, a.tag, band)
    D = band.merge(F.drop(columns=[c for c in ("y", "country", "s1_id", "o_id") if c in F]), on=["a", "b"])
    D = D.merge(oof[["a", "b", "s1_id", "o_id", "country"]], on=["a", "b"])
    for j, p in enumerate(a.ce):
        D = D.merge(pd.read_parquet(p).rename(columns={"ce": f"ce{j}"}), on=["a", "b"], how="left")
    ys = set(map(tuple, load_gt(a.gt)[["s1", "other"]].to_numpy()))
    D["y"] = np.fromiter(((s, o) in ys for s, o in zip(D.s1_id, D.o_id)), bool, len(D))
    return oof, D


def fit_predict(Xtr, ytr, Xva, feats, monotone, seeds, n_iter=None, Xes=None, yes=None):
    params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
                  num_threads=20)
    if monotone:
        params["monotone_constraints"] = [1 if INC.match(f) else (-1 if DEC.match(f) else 0) for f in feats]
        params["monotone_constraints_method"] = "advanced"
    preds, its = [], []
    for s in range(seeds):
        p = dict(params, seed=17 + s, bagging_seed=17 + s, feature_fraction_seed=17 + s)
        if n_iter is None:
            m = lgb.train(p, lgb.Dataset(Xtr, ytr, feature_name=feats), 5000,
                          valid_sets=[lgb.Dataset(Xes, yes, feature_name=feats)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            its.append(m.best_iteration)
            preds.append(m.predict(Xva, num_iteration=m.best_iteration))
        else:
            m = lgb.train(p, lgb.Dataset(Xtr, ytr, feature_name=feats), n_iter)
            preds.append(m.predict(Xva))
    return np.mean(preds, 0), its, m


def macro(full_p, oof, s2, truth, s1_eval):
    full = oof[["a", "b", "s1_id", "o_id"]].merge(s2, on=["a", "b"]).merge(full_p, on=["a", "b"], how="left")
    full["p"] = full.pb.fillna(full.p2)
    ex = exclusive(full[["a", "b", "s1_id", "o_id", "p"]])
    sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
    return float(np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in s1_eval]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", default="_hyb5")
    ap.add_argument("--s2-tag", default="_v9")
    ap.add_argument("--ce", nargs="*", default=[])
    ap.add_argument("--save-tag", default="_v11r")
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()
    oof, D = build(a)
    s2 = pd.read_parquet(Path(a.art) / f"oof_stage2{a.tag}{a.s2_tag}.parquet", columns=["a", "b", "p2"])
    feats = [c for c in D.columns if c not in ("a", "b", "s1_id", "o_id", "country", "y")]
    X, y = D[feats].to_numpy(np.float32), D.y.to_numpy()
    fold = (pd.util.hash_array(D.s1_id.to_numpy(dtype=object)) // 7 % a.folds).astype(int)
    gt = load_gt(a.gt)
    samp = set(oof.s1_id)
    truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
    s1_country = oof.drop_duplicates("s1_id").set_index("s1_id").country
    all_s1 = samp | set(truth)
    by_c = {c: {s for s in all_s1 if s1_country.get(s, None) == c} for c in D.country.unique()}
    print(f"band pairs {len(D)}, features {len(feats)}, monotone: +{sum(bool(INC.match(f)) for f in feats)} "
          f"-{sum(bool(DEC.match(f)) for f in feats)}", flush=True)
    results = {}
    for name, mono, seeds in (("base", False, 1), ("mono", True, 1), ("bag5", False, 5), ("mono_bag5", True, 5)):
        pb = np.zeros(len(D))
        its_all = []
        for k in range(a.folds):
            tr, va = fold != k, fold == k
            p, its, _ = fit_predict(X[tr], y[tr], X[va], feats, mono, seeds, Xes=X[va], yes=y[va])
            pb[va] = p
            its_all += its
        f_in = macro(D[["a", "b"]].assign(pb=pb), oof, s2, truth, all_s1)
        n_it = int(np.mean(its_all))
        loco = {}
        for src, dst in (("US", "India"), ("India", "US")):
            tr, va = (D.country == src).to_numpy(), (D.country == dst).to_numpy()
            p, _, _ = fit_predict(X[tr], y[tr], X[va], feats, mono, seeds, n_iter=n_it)
            loco[f"{src}->{dst}"] = macro(D.loc[va, ["a", "b"]].assign(pb=p), oof[oof.country == dst], s2, truth, by_c[dst])
        results[name] = {"in_country": f_in, **loco, "iters": n_it}
        print(f"{name:10s} in-country {f_in:.5f} | " + " | ".join(f"{k} {v:.5f}" for k, v in loco.items()), flush=True)
    json.dump(results, open(Path(a.art) / f"band_robust{a.save_tag}.json", "w"), indent=1)
    # final models for the two most promising variants (all data, fixed iterations)
    for name in ("mono_bag5", "bag5"):
        mono, seeds = name.startswith("mono"), 5
        for s in range(seeds):
            params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=100,
                          feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
                          num_threads=20, seed=17 + s, bagging_seed=17 + s, feature_fraction_seed=17 + s)
            if mono:
                params["monotone_constraints"] = [1 if INC.match(f) else (-1 if DEC.match(f) else 0) for f in feats]
                params["monotone_constraints_method"] = "advanced"
            m = lgb.train(params, lgb.Dataset(X, y, feature_name=feats), int(results[name]["iters"] * 1.1))
            m.save_model(str(Path(a.art) / f"band_model{a.save_tag}_{name}_s{s}.txt"))
    print("saved bagged models", flush=True)


if __name__ == "__main__":
    main()
