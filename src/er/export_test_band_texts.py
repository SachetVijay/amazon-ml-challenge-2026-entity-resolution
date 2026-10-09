"""Export test band pairs (a, b, text_a, text_b, country) so a cross-encoder can score them off-box (Kaggle).
  python -m er.export_test_band_texts --art artifacts/test --out kaggle_testband/band.parquet"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_encoder import load_texts
from .predict import country_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = []
    for c in country_rows(Path(a.art) / "records.parquet"):
        B = pd.read_parquet(Path(a.art) / f"band_{c}.parquet", columns=["a", "b"]).assign(country=c)
        parts.append(B)
    B = pd.concat(parts, ignore_index=True)
    txt = load_texts(a.art, np.concatenate([B.a.to_numpy(), B.b.to_numpy()]))
    B["text_a"] = [txt[int(x)] for x in B.a]
    B["text_b"] = [txt[int(x)] for x in B.b]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    B.to_parquet(a.out, compression="zstd")
    print(f"wrote {a.out}: {len(B)} test band pairs")


if __name__ == "__main__":
    main()
