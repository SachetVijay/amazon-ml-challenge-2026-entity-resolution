"""Band specialist: a GBDT trained ONLY on the uncertain stage-1 band, with the full stage-1 feature vector +
cross-encoder logit(s) + cluster context. Stage 1 is trained on 12M mostly-easy pairs and stage 2 only sees
summary numbers; this model is specialised on the ~3 % of pairs that decide the score.

  python -m er.exp_band --art artifacts/train --gt .../train_ground_truth.tsv --ce artifacts/train/ce_v6_oof.parquet
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .decide import exclusive, expected_f_fast, f05
from .predict import country_rows, read_country
from .prep import load_gt
from .stage2 import context_features


def band_features(art: str, tag: str, keys: pd.DataFrame) -> pd.DataFrame:
    """Stream cached stage-1 feature files, keep only rows whose (a, b) is in `keys`."""
    out = []
    keyset = keys[["a", "b"]]
    for fp in sorted(Path(art).glob(f"feats{tag}_*.parquet")):
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=1_000_000):
            df = batch.to_pandas()
            out.append(df.merge(keyset, on=["a", "b"]))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tag", default="_hyb5")
    ap.add_argument("--ce", nargs="*", default=[])
    ap.add_argument("--band", type=float, nargs=2, default=[0.005, 0.995])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--s2-tag", default="_v6", help="stage-2 OOF file suffix")
    ap.add_argument("--extra-tag", default=None, help="add extra_band_<tag>_*.parquet rows to training only")
    ap.add_argument("--extra-ce", nargs="*", default=[], help="CE scores for the extra rows, same order as --ce")
    ap.add_argument("--save-tag", default="", help="suffix for band_model / oof_band / metrics files")
    a = ap.parse_args()
    oof = pd.read_parquet(Path(a.art) / f"oof{a.tag}.parquet")          # a b country s1_id o_id p (stage 1)
    s2 = pd.read_parquet(Path(a.art) / f"oof_stage2{a.tag}{a.s2_tag}.parquet", columns=["a", "b", "p2"])
    # cluster context on the full stage-1 output (per country), then keep the band
    path = Path(a.art) / "records.parquet"
    ctxs = []
    for country, rows in country_rows(path).items():
        P = oof[oof.country == country]
        if len(P):
            sub = read_country(path, country, rows)[["name_core", "addr_clean"]]
            ctxs.append(context_features(P, sub))
    C = pd.concat(ctxs, ignore_index=True)
    band = C[(C.p1 > a.band[0]) & (C.p1 < a.band[1])]
    print(f"band pairs: {len(band)} of {len(C)}", flush=True)
    F = band_features(a.art, a.tag, band)
    D = band.merge(F.drop(columns=[c for c in ("y", "country", "s1_id", "o_id") if c in F]), on=["a", "b"])
    D = D.merge(oof[["a", "b", "s1_id", "o_id", "country"]], on=["a", "b"])
    for j, p in enumerate(a.ce):
        D = D.merge(pd.read_parquet(p).rename(columns={"ce": f"ce{j}"}), on=["a", "b"], how="left")
    ys = set(map(tuple, load_gt(a.gt)[["s1", "other"]].to_numpy()))
    D["y"] = np.fromiter(((s, o) in ys for s, o in zip(D.s1_id, D.o_id)), bool, len(D))
    feats = [c for c in D.columns if c not in ("a", "b", "s1_id", "o_id", "country", "y")]
    print(f"{len(D)} band pairs x {len(feats)} features, pos rate {D.y.mean():.3f}", flush=True)
    X, y = D[feats].to_numpy(np.float32), D.y.to_numpy()
    XE = yE = None
    if a.extra_tag:   # extra TRAINING rows from other leak-free buckets (never evaluated on)
        E = pd.concat([pd.read_parquet(f) for f in sorted(Path(a.art).glob(f"extra_band_{a.extra_tag}_*.parquet"))
                       if not f.name.endswith("_pairs.parquet")], ignore_index=True)
        for j, p in enumerate(a.extra_ce):
            E = E.merge(pd.read_parquet(p).rename(columns={"ce": f"ce{j}"}), on=["a", "b"], how="left")
        missing = [f for f in feats if f not in E.columns]
        assert not missing, f"extra rows lack features: {missing}"
        XE, yE = E[feats].to_numpy(np.float32), E.y.to_numpy()
        print(f"extra training rows: {len(E)} (pos rate {yE.mean():.3f}) from buckets tag {a.extra_tag}", flush=True)
    fold = (pd.util.hash_array(D.s1_id.to_numpy(dtype=object)) // 7 % a.folds).astype(int)   # = stage-1 folds
    params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
                  num_threads=20)
    pb = np.zeros(len(D), np.float32)
    its = []
    for k in range(a.folds):
        tr, va = fold != k, fold == k
        Xtr, ytr = (X[tr], y[tr]) if XE is None else (np.vstack([X[tr], XE]), np.concatenate([y[tr], yE]))
        m = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=feats), 5000,
                      valid_sets=[lgb.Dataset(X[va], y[va], feature_name=feats)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pb[va] = m.predict(X[va], num_iteration=m.best_iteration)
        its.append(m.best_iteration)
        print(f"fold {k} it {m.best_iteration}", flush=True)
    imp = pd.Series(m.feature_importance("gain"), feats)
    print("top:", (imp / imp.sum()).sort_values(ascending=False).head(15).round(3).to_dict())
    from sklearn.metrics import roc_auc_score
    print(f"band AUC: p1 {roc_auc_score(y, D.p1):.4f} | stage2 {roc_auc_score(y, D.merge(s2, on=['a','b']).p2):.4f}"
          f" | specialist {roc_auc_score(y, pb):.4f}")
    # final probability: specialist inside the band, stage 2 outside
    full = oof[["a", "b", "s1_id", "o_id"]].merge(s2, on=["a", "b"]).merge(
        D[["a", "b"]].assign(pb=pb), on=["a", "b"], how="left")
    full["p"] = full.pb.fillna(full.p2)
    gt = load_gt(a.gt)
    samp = set(oof.s1_id)
    truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
    allS = samp | set(truth)
    scores = {}
    for col in ("p2", "p"):
        view = full.drop(columns=["p"]).rename(columns={"p2": "p"}) if col == "p2" else full
        ex = exclusive(view[["a", "b", "s1_id", "o_id", "p"]])
        sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
        pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
        scores[col] = float(np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in allS]))
        print(f"F0.5 using {'stage 2' if col == 'p2' else 'band specialist'}: {scores[col]:.5f}", flush=True)
    full[["a", "b", "s1_id", "o_id", "p"]].to_parquet(Path(a.art) / f"oof_band{a.save_tag}.parquet")
    import json
    json.dump({"stage2": scores["p2"], "band": scores["p"], "ce": a.ce, "features": feats},
              open(Path(a.art) / f"band_metrics{a.save_tag}.json", "w"))
    Xall, yall = (X, y) if XE is None else (np.vstack([X, XE]), np.concatenate([y, yE]))
    m_full = lgb.train(params, lgb.Dataset(Xall, yall, feature_name=feats), int(np.mean(its) * 1.1))
    m_full.save_model(str(Path(a.art) / f"band_model{a.save_tag}.txt"))
    print("saved band model", flush=True)


if __name__ == "__main__":
    main()
