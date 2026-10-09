"""Full-scale inference: records -> blocking -> features (chunked) -> model -> decision -> submission TSVs.

  python -m er.predict --split-dir student_resource/dataset/test --prefix test --art artifacts/test \
      --model artifacts/dev10/model.txt --out output
"""
import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import generate, pivot_scores
from .decide import exclusive, expected_f_fast
from .candidates import HYBRID_CTX, hybrid, record_best
from .features import pair_features
from .prep import load_split


def add_context(X: pd.DataFrame) -> pd.DataFrame:
    for key in ("b", "a"):
        g = X.groupby(key)["score_all_b2a"]
        X[f"{key}_ncand"] = X.groupby(key)["a"].transform("size").astype(np.float32)
        X[f"{key}_score_gap"] = (X["score_all_b2a"].fillna(0) - g.transform("max").fillna(0)).astype(np.float32)
    return X


def append_lists(f, s1_rows: np.ndarray, a: np.ndarray, b: np.ndarray, ids: pd.Series):
    """Append one TSV line per S1 row (empty list allowed). Groups on integer rows; strings built per line."""
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    starts = np.searchsorted(a, s1_rows, "left")
    ends = np.searchsorted(a, s1_rows, "right")
    idv = ids.to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids.index)
    s1p, bp = pos.loc[s1_rows].to_numpy(), pos.loc[b].to_numpy() if len(b) else np.array([], int)
    for r, st, en in zip(s1p, starts, ends):
        f.write(f"{idv[r]}\t{','.join(dict.fromkeys(idv[bp[st:en]]))}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--prefix", default="test")
    ap.add_argument("--art", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--k-other", type=int, default=5)
    ap.add_argument("--k-s1", type=int, default=5)
    ap.add_argument("--chunk-s1", type=int, default=100_000)
    ap.add_argument("--rule", default="thr", choices=["expected_f", "thr"])
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--empty-weight", type=float, default=1.0)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--save-feats", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action="store_true", help="reuse existing per-country prediction files")
    ap.add_argument("--dense-k", type=int, default=5)
    a = ap.parse_args()
    T = time.time()
    load_split(a.split_dir, a.prefix, a.art, workers=22, arrow=True) if not (Path(a.art) / "records.parquet").exists() else None
    booster = lgb.Booster(model_file=a.model)
    cols = booster.feature_name()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_by_country = country_rows(Path(a.art) / "records.parquet")
    fm = open(out / "matching_results.tsv", "w")
    fc = open(out / "candidate_pairs.tsv", "w")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    n_s1 = n_match = n_nonempty = 0
    for country, rows in rows_by_country.items():  # open label set: whatever countries the file contains
        sub = read_country(Path(a.art) / "records.parquet", country, rows)
        ids = pd.Series(sub.entity_id.to_numpy(dtype=object), index=sub.index)
        cp = Path(a.art) / f"cand_{country}_k{a.k_other}_{a.k_s1}.parquet"
        if not cp.exists():
            country_candidates(sub, country, a)  # builds + caches sparse candidates & context
        ctx = CTX + (HYBRID_CTX if a.hybrid else [])
        b_best = record_best(a.art, country) if a.hybrid else None
        s1_all_rows = np.sort(sub.index[sub.src.to_numpy() == 1].to_numpy())
        preds, cand_ab = [], []
        pred_path = Path(a.art) / f"pred{'_hyb' if a.hybrid else ''}_{country}.parquet"
        if a.resume and pred_path.exists():  # reuse a finished country (same model/candidates assumed)
            preds = [pd.read_parquet(pred_path)]
            cand_ab = [pd.read_parquet(pred_path.with_suffix(".cand.parquet"))]
            s1_iter = []
            print(f"  {country}: reusing {pred_path}", flush=True)
        else:
            s1_iter = range(0, len(s1_all_rows), a.chunk_s1)
        for i in s1_iter:
            keep = list(map(int, s1_all_rows[i:i + a.chunk_s1]))
            fcache = Path(a.art) / "feats_cache" / f"{country}_{i // a.chunk_s1:03d}.parquet"
            if a.resume and fcache.exists():   # exact: same features, computed earlier by this code
                X = pd.read_parquet(fcache)
                cand_ab.append(X[["a", "b"]].copy())
            else:
                part = pd.read_parquet(cp, filters=[("a", "in", keep)])  # only this chunk's S1 rows
                if a.hybrid:
                    part = hybrid(part, a.art, country, rows, a.dense_k, keep_a=keep, b_best=b_best)
                cand_ab.append(part[["a", "b"]].copy())
                X = pair_features(sub, part, workers=22).merge(part[ctx], on=["a", "b"])
                if a.save_feats:  # cache test features: later versions reuse them or add only new columns
                    fcache.parent.mkdir(exist_ok=True)
                    X.astype({c: np.float32 for c in X.columns if c not in ("a", "b")}).to_parquet(fcache)
            preds.append(pd.DataFrame({"a": X.a.values, "b": X.b.values,
                                       "p": booster.predict(X[cols].to_numpy(np.float32), num_threads=22).astype(np.float32)}))
            del X
            gc.collect()
            print(f"  {country} scored {min(i + a.chunk_s1, len(s1_all_rows))}/{len(s1_all_rows)} S1 {time.time() - T:.0f}s", flush=True)
        cand = pd.concat(cand_ab, ignore_index=True)
        P = pd.concat(preds, ignore_index=True)
        P.to_parquet(pred_path)
        cand.to_parquet(pred_path.with_suffix(".cand.parquet"))
        ex = exclusive(P)
        sel = expected_f_fast(ex, empty_weight=a.empty_weight) if a.rule == "expected_f" else ex[ex.p >= a.thr]
        s1_rows = s1_all_rows
        append_lists(fm, s1_rows, sel.a.to_numpy(), sel.b.to_numpy(), ids)
        append_lists(fc, s1_rows, cand.a.to_numpy(), cand.b.to_numpy(), ids)
        n_s1 += len(s1_rows)
        n_match += len(sel)
        n_nonempty += sel.a.nunique()
        print(f"{country}: {len(sel)} matches for {len(s1_rows)} S1 {time.time() - T:.0f}s", flush=True)
        del sub, cand, P, ex, sel, ids, preds, s1_rows
        gc.collect()
    fm.close()
    fc.close()
    print(f"wrote {out}: {n_match} matches, {n_nonempty} S1 with >=1 match ({n_nonempty / n_s1:.1%}), "
          f"mean {n_match / n_s1:.2f}/S1  {time.time() - T:.0f}s")


CTX = ["a", "b", "b_ncand", "b_score_gap", "a_ncand", "a_score_gap"]


def country_rows(path) -> dict:
    """Global row numbers of each country (row order of records.parquet)."""
    c = pd.read_parquet(path, columns=["country"]).country.to_numpy(dtype=object)
    return {k: np.flatnonzero(c == k) for k in pd.unique(c)}


def read_country(path, country, rows) -> pd.DataFrame:
    sub = pd.read_parquet(path, filters=[("country", "==", country)], dtype_backend="pyarrow")
    assert len(sub) == len(rows)
    sub.index = rows  # keep global row ids (candidate caches and a/b indices use them)
    return sub


def country_candidates(sub, country, a) -> pd.DataFrame:
    cp = Path(a.art) / f"cand_{country}_k{a.k_other}_{a.k_s1}.parquet"
    if cp.exists():
        return pd.read_parquet(cp)
    cand = add_context(pivot_scores(generate(sub, a.k_other, a.k_s1, threads=22, retrievers=("all",))))
    cand.to_parquet(cp)
    return cand

if __name__ == "__main__":
    main()
