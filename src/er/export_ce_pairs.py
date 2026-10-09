"""Export cross-encoder training pairs (text_a, text_b, y) for given S1 hash buckets to a parquet file,
so a cross-encoder can be trained anywhere (local GPU or Kaggle) with no access to the raw artifacts.

  python -m er.export_ce_pairs --art artifacts/train --gt .../train_ground_truth.tsv --buckets 6 7 \
      --max-pairs 1200000 --out kaggle_ce/pairs.parquet
"""
import argparse
from pathlib import Path

import numpy as np

from .cross_encoder import build_train_pairs, load_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--buckets", type=int, nargs="+", required=True)
    ap.add_argument("--max-pairs", type=int, default=1_200_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="raw", choices=["raw", "norm"])
    a = ap.parse_args()
    P = build_train_pairs(a)
    txt = load_texts(a.art, np.concatenate([P.a.to_numpy(), P.b.to_numpy()]), a.view)
    P["text_a"] = [txt[int(x)] for x in P.a]
    P["text_b"] = [txt[int(x)] for x in P.b]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    P[["text_a", "text_b", "y"]].to_parquet(a.out, compression="zstd")
    print(f"wrote {a.out}: {len(P)} pairs, pos rate {P.y.mean():.3f}")


if __name__ == "__main__":
    main()
