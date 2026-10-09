"""Export the extra-bucket BAND pairs (the uncertain region, with labels) as (text_a, text_b, y) for training a
hard-example cross-encoder.   python -m er.export_band_pairs --art artifacts/train --tag b45 --out hard_b45.parquet"""
import argparse
import glob

import numpy as np
import pandas as pd

from .cross_encoder import load_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = pd.concat([pd.read_parquet(f, columns=["a", "b", "y"]) for f in sorted(glob.glob(f"{a.art}/extra_band_{a.tag}_*.parquet"))
                   if not f.endswith("_pairs.parquet")], ignore_index=True)
    txt = load_texts(a.art, np.concatenate([E.a.to_numpy(), E.b.to_numpy()]))
    E["text_a"] = [txt[int(x)] for x in E.a]
    E["text_b"] = [txt[int(x)] for x in E.b]
    E[["text_a", "text_b", "y"]].sample(frac=1, random_state=0).to_parquet(a.out)
    print(f"wrote {a.out}: {len(E)} hard pairs, pos rate {E.y.mean():.3f}")


if __name__ == "__main__":
    main()
