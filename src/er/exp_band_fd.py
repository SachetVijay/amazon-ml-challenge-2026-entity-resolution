"""Band specialist on test-like (full-density) competition context.

The record-side context of a validation band pair (rank_in_b, b_max, b_second, b_margin) used to be computed from
bucket-0 S1s only, i.e. ~1/8 of the S1s that compete for the record at test time. er.build_competitors scores
the missing rivals with the final stage-1 model; here the context is recomputed over bucket-0 pairs + rivals,
with the same p > min_p filter predict_stage2 applies, so train/validation inputs match the test.

Reports 4-fold OOF macro F0.5 (bucket-0 S1s) for
  old->old  the protocol used so far (bucket-0-only context for training and evaluation)
  old->new  the CURRENT model facing test-like context: what the submitted model actually does at test time
  new->new  the model retrained on test-like context
plus leave-one-country-out (US -> India) for old->new vs new->new as the proxy for an unseen country.

  python -m er.exp_band_fd --art artifacts/train --gt .../train_ground_truth.tsv --s2-tag _v9 \
      --ce <OOF CE files in the final model's order> --comp-tag fd --save-tag _v12
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .exp_band import band_features
from .exp_band_robust import macro
from .predict import country_rows, read_country
from .prep import load_gt
from .stage2 import context_features

CTXC = ["p1", "rank_in_a", "rank_in_b", "a_max", "a_sum", "a_n50", "b_max", "b_second", "b_margin",
        "sib_name", "sib_addr", "sib_p"]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.6,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=20)


def contexts(a, oof):
    path = Path(a.art) / "records.parquet"
    old, new = [], []
    for country, rows in country_rows(path).items():
        P = oof[oof.country == country][["a", "b", "p"]]
        if not len(P):
            continue
        sub = read_country(path, country, rows)[["name_core", "addr_clean"]]
        C = context_features(P, sub)
        band = C[(C.p1 > a.band[0]) & (C.p1 < a.band[1])]
        old.append(band)
        comp = pd.read_parquet(Path(a.art) / f"comp_{a.comp_tag}_{country}.parquet", columns=["a", "b", "p"])
        Pn = pd.concat([P, comp], ignore_index=True)
        Cn = context_features(Pn[Pn.p > a.min_p].reset_index(drop=True), sub)
        new.append(Cn.merge(band[["a", "b"]], on=["a", "b"]))
        print(f"{country}: {len(band)} band pairs, {len(comp)} rival pairs; rank_in_b>=2 "
              f"{np.mean(band.rank_in_b >= 2):.1%} -> {np.mean(new[-1].rank_in_b >= 2):.1%}", flush=True)
    return pd.concat(old, ignore_index=True), pd.concat(new, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", default="_hyb5")
    ap.add_argument("--s2-tag", default="_v9")
    ap.add_argument("--ce", nargs="*", default=[])
    ap.add_argument("--comp-tag", default="fd")
    ap.add_argument("--save-tag", default="_v12")
    ap.add_argument("--band", type=float, nargs=2, default=[0.005, 0.995])
    ap.add_argument("--min-p", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()
    oof = pd.read_parquet(Path(a.art) / f"oof{a.tag}.parquet")
    s2 = pd.read_parquet(Path(a.art) / f"oof_stage2{a.tag}{a.s2_tag}.parquet", columns=["a", "b", "p2"])
    Cold, Cnew = contexts(a, oof)
    F = band_features(a.art, a.tag, Cold)
    D = Cold.merge(F.drop(columns=[c for c in ("y", "country", "s1_id", "o_id") if c in F]), on=["a", "b"])
    D = D.merge(oof[["a", "b", "s1_id", "o_id", "country"]], on=["a", "b"])
    for j, p in enumerate(a.ce):
        D = D.merge(pd.read_parquet(p).rename(columns={"ce": f"ce{j}"}), on=["a", "b"], how="left")
    ys = set(map(tuple, load_gt(a.gt)[["s1", "other"]].to_numpy()))
    D["y"] = np.fromiter(((s, o) in ys for s, o in zip(D.s1_id, D.o_id)), bool, len(D))
    Dn = D.copy()
    key = pd.MultiIndex.from_arrays([D.a.to_numpy(), D.b.to_numpy()])
    N = Cnew.set_index(["a", "b"])
    for c in CTXC:
        Dn[c] = N[c].reindex(key).to_numpy()
    assert not Dn[CTXC].isna().any().any(), "band pair missing from test-like context"
    feats = [c for c in D.columns if c not in ("a", "b", "s1_id", "o_id", "country", "y")]
    Xo, Xn, y = D[feats].to_numpy(np.float32), Dn[feats].to_numpy(np.float32), D.y.to_numpy()
    print(f"{len(D)} band pairs x {len(feats)} features; changed context values: "
          + ", ".join(f"{c} {np.mean(~np.isclose(D[c], Dn[c])):.1%}" for c in CTXC), flush=True)
    fold = (pd.util.hash_array(D.s1_id.to_numpy(dtype=object)) // 7 % a.folds).astype(int)
    gt = load_gt(a.gt)
    samp = set(oof.s1_id)
    truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
    all_s1 = samp | set(truth)
    s1_country = oof.drop_duplicates("s1_id").set_index("s1_id").country
    by_c = {c: {s for s in all_s1 if s1_country.get(s) == c} for c in D.country.unique()}
    res, oofs, its = {}, {}, {"old": [], "new": []}
    pb = {k: np.zeros(len(D), np.float32) for k in ("old->old", "old->new", "new->new")}
    for k in range(a.folds):
        tr, va = fold != k, fold == k
        for ctx, Xtr in (("old", Xo), ("new", Xn)):
            m = lgb.train(PARAMS, lgb.Dataset(Xtr[tr], y[tr], feature_name=feats), 5000,
                          valid_sets=[lgb.Dataset(Xtr[va], y[va], feature_name=feats)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            its[ctx].append(m.best_iteration)
            if ctx == "old":
                pb["old->old"][va] = m.predict(Xo[va], num_iteration=m.best_iteration)
                pb["old->new"][va] = m.predict(Xn[va], num_iteration=m.best_iteration)
            else:
                pb["new->new"][va] = m.predict(Xn[va], num_iteration=m.best_iteration)
        print(f"fold {k}: it old {its['old'][-1]} new {its['new'][-1]}", flush=True)
    for name, p in pb.items():
        keys = D[["a", "b"]].assign(pb=p)
        res[name] = {"all": macro(keys, oof, s2, truth, all_s1)}
        for c in ("US", "India"):
            res[name][c] = macro(keys[(D.country == c).to_numpy()], oof[oof.country == c], s2, truth, by_c[c])
        print(f"{name}: macro F0.5 {res[name]['all']:.5f} | US {res[name]['US']:.5f} | India {res[name]['India']:.5f}",
              flush=True)
    # unseen-country proxy: train on US only, evaluate India on test-like context
    tr, va = (D.country == "US").to_numpy(), (D.country == "India").to_numpy()
    for ctx, Xtr in (("old", Xo), ("new", Xn)):
        m = lgb.train(PARAMS, lgb.Dataset(Xtr[tr], y[tr], feature_name=feats), int(np.mean(its[ctx])))
        keys = D.loc[va, ["a", "b"]].assign(pb=m.predict(Xn[va]))
        res[f"LOCO US->India {ctx}->new"] = macro(keys, oof[oof.country == "India"], s2, truth, by_c["India"])
        print(f"LOCO US->India, {ctx}-context model on test-like India: {res[f'LOCO US->India {ctx}->new']:.5f}",
              flush=True)
    json.dump(res, open(Path(a.art) / f"band_fd_metrics{a.save_tag}.json", "w"), indent=1)
    oof[["a", "b", "s1_id", "o_id"]].merge(s2, on=["a", "b"]).merge(
        D[["a", "b"]].assign(pb=pb["new->new"]), on=["a", "b"], how="left").assign(
        p=lambda x: x.pb.fillna(x.p2))[["a", "b", "s1_id", "o_id", "p"]].to_parquet(
        Path(a.art) / f"oof_band{a.save_tag}.parquet")
    m = lgb.train(PARAMS, lgb.Dataset(Xn, y, feature_name=feats), int(np.mean(its["new"]) * 1.1))
    m.save_model(str(Path(a.art) / f"band_model{a.save_tag}.txt"))
    print("saved band model (test-like context)", flush=True)


if __name__ == "__main__":
    main()
