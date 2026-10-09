"""Full-density competition for the band specialist's training pairs.

Validation samples S1 entities (bucket 0), so the rival S1s of a band record that live in other buckets were
never scored: its record-side context (rank_in_b, b_max, b_second, b_margin) saw ~1/8 of the competition,
while at test time every S1 competes (test band: 21 % of pairs have a better-scored rival vs 12 % in training;
France 44 %). For every record in the bucket-0 band, score all its candidate S1s OUTSIDE bucket 0 with the final
stage-1 model (never trained on them, exactly as at test time) and save (a, b, p) so the band specialist can be
trained and validated on test-like context.

  python -m er.build_competitors --art artifacts/train --model final/final_models/model.txt --tag fd
"""
import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import HYBRID_CTX, hybrid, record_best
from .features import pair_features
from .predict import CTX, country_rows, read_country


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--oof-tag", default="_hyb5")
    ap.add_argument("--tag", default="fd")
    ap.add_argument("--dense-k", type=int, default=5)
    ap.add_argument("--band", type=float, nargs=2, default=[0.005, 0.995])
    ap.add_argument("--chunk-s1", type=int, default=100_000)
    a = ap.parse_args()
    booster = lgb.Booster(model_file=a.model)
    cols = booster.feature_name()
    path = Path(a.art) / "records.parquet"
    oof = pd.read_parquet(Path(a.art) / f"oof{a.oof_tag}.parquet", columns=["a", "b", "p", "country"])
    for country, rows in country_rows(path).items():
        out = Path(a.art) / f"comp_{a.tag}_{country}.parquet"
        P = oof[oof.country == country]
        if out.exists() or not len(P):
            continue
        T = time.time()
        bb = np.unique(P.b[(P.p > a.band[0]) & (P.p < a.band[1])].to_numpy())
        bl = list(map(int, bb))
        cp = Path(a.art) / f"cand_{country}_k5_5.parquet"
        dp = Path(a.art) / f"dense_{country}_k10_10.parquet"
        s = pd.read_parquet(cp, columns=["a"], filters=[("b", "in", bl)])
        d = pd.read_parquet(dp, columns=["a"], filters=[[("b", "in", bl), ("dense_rank_b2a", "<=", a.dense_k)],
                                                        [("b", "in", bl), ("dense_rank_a2b", "<=", a.dense_k)]])
        sub = read_country(path, country, rows)
        ids = pd.Series(sub.entity_id.to_numpy(dtype=object), index=sub.index)
        a0 = P.a.unique()
        assert (pd.util.hash_array(ids.loc[a0].to_numpy()) % 8 == 0).all(), "bucket definition mismatch"
        cand_a = np.union1d(s.a.to_numpy(), d.a.to_numpy())
        rivals = cand_a[pd.util.hash_array(ids.loc[cand_a].to_numpy()) % 8 != 0]   # S1s validation never scored
        print(f"{country}: {len(bb)} band records, {len(cand_a)} candidate S1s, {len(rivals)} outside bucket 0",
              flush=True)
        del s, d
        b_best = record_best(a.art, country)
        bset = pd.Index(bb)
        outs = []
        for i in range(0, len(rivals), a.chunk_s1):
            keep = list(map(int, rivals[i:i + a.chunk_s1]))
            part = pd.read_parquet(cp, filters=[("a", "in", keep)])
            part = hybrid(part, a.art, country, rows, a.dense_k, keep_a=keep, b_best=b_best)  # a-side ctx complete
            part = part[part.b.isin(bset)].reset_index(drop=True)
            X = pair_features(sub, part, workers=20).merge(part[CTX + HYBRID_CTX], on=["a", "b"])
            outs.append(X[["a", "b"]].assign(p=booster.predict(X[cols].to_numpy(np.float32), num_threads=20)))
            del part, X
            gc.collect()
            print(f"  {country} {min(i + a.chunk_s1, len(rivals))}/{len(rivals)} rival S1 "
                  f"{sum(map(len, outs))} pairs {time.time() - T:.0f}s", flush=True)
        C = pd.concat(outs, ignore_index=True)
        C.to_parquet(out)
        print(f"{country}: {len(C)} rival pairs, p>0.05 {np.mean(C.p > .05):.1%}, {time.time() - T:.0f}s", flush=True)
        del sub, C, outs
        gc.collect()


if __name__ == "__main__":
    main()
