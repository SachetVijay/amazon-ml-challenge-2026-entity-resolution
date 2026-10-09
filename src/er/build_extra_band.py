"""Extra band-specialist training data from other entity buckets.

Only bucket 0 is unseen by *every* learned component, which caps the band specialist at ~400k pairs.
Buckets 4-5 are unseen by the stage-1 model (bucket 0), the bi-encoder (1), the e5-base CE (2-3), the
normalised CE (2-3), mDeBERTa (6-7) and e5-large (6-7) -- only the e5-base *continuation* (4-5) saw them,
so a band model that omits that one CE can train on buckets 0 + 4 + 5 without leakage.

For the chosen buckets, per country: hybrid candidates -> stage-1 features -> stage-1 probability (the final
stage-1 model, which never saw these entities) -> cluster context -> keep the band (0.005 < p1 < 0.995) ->
save features + label. Cross-encoder scores are added afterwards with er.cross_encoder score.

  python -m er.build_extra_band --art artifacts/train --gt .../train_ground_truth.tsv --model final/v9_models/model.txt \
      --buckets 4 5 --tag b45
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
from .prep import load_gt
from .stage2 import context_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--buckets", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dense-k", type=int, default=5)
    ap.add_argument("--band", type=float, nargs=2, default=[0.005, 0.995])
    ap.add_argument("--chunk-s1", type=int, default=60_000)
    a = ap.parse_args()
    booster = lgb.Booster(model_file=a.model)
    cols = booster.feature_name()
    path = Path(a.art) / "records.parquet"
    gt = load_gt(a.gt)
    for country, rows in country_rows(path).items():
        out = Path(a.art) / f"extra_band_{a.tag}_{country}.parquet"
        if out.exists():
            continue
        T = time.time()
        sub = read_country(path, country, rows)
        ids = pd.Series(sub.entity_id.to_numpy(dtype=object), index=sub.index)
        s1 = ids[sub.src.to_numpy() == 1]
        keep_all = s1.index[np.isin(pd.util.hash_array(s1.to_numpy()) % 8, a.buckets)].to_numpy()
        cp = Path(a.art) / f"cand_{country}_k5_5.parquet"
        b_best = record_best(a.art, country)
        preds, bands = [], []
        for i in range(0, len(keep_all), a.chunk_s1):
            keep = list(map(int, keep_all[i:i + a.chunk_s1]))
            part = pd.read_parquet(cp, filters=[("a", "in", keep)])
            part = hybrid(part, a.art, country, rows, a.dense_k, keep_a=keep, b_best=b_best)
            X = pair_features(sub, part, workers=20).merge(part[CTX + HYBRID_CTX], on=["a", "b"])
            X["p"] = booster.predict(X[cols].to_numpy(np.float32), num_threads=20)
            preds.append(X[["a", "b", "p"]])
            bands.append(X[(X.p > a.band[0]) & (X.p < a.band[1])])
            del X, part
            gc.collect()
            print(f"  {country} {min(i + a.chunk_s1, len(keep_all))}/{len(keep_all)} S1 {time.time() - T:.0f}s", flush=True)
        P = pd.concat(preds, ignore_index=True)
        ctx = context_features(P, sub[["name_core", "addr_clean"]])            # context over ALL candidates
        B = pd.concat(bands, ignore_index=True).drop(columns=["p"])
        B = ctx.merge(B, on=["a", "b"])                                         # band rows: context + features
        B["s1_id"], B["o_id"], B["country"] = ids.loc[B.a].to_numpy(), ids.loc[B.b].to_numpy(), country
        pos = set(map(tuple, gt[gt.s1.isin(set(B.s1_id))][["s1", "other"]].to_numpy()))
        B["y"] = np.fromiter(((s, o) in pos for s, o in zip(B.s1_id, B.o_id)), bool, len(B))
        B.to_parquet(out)
        B[["a", "b"]].assign(p=B.p1).to_parquet(Path(a.art) / f"extra_band_{a.tag}_{country}_pairs.parquet")
        print(f"{country}: {len(B)} band pairs, pos rate {B.y.mean():.3f}, {time.time() - T:.0f}s", flush=True)
        del sub, P, ctx, B, preds, bands
        gc.collect()


if __name__ == "__main__":
    main()
