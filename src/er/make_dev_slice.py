"""Deterministic 10 % development slice of the training data.

Keeps 10 % of Source-1 entities (with all their true matches) plus 10 % of the unmatched S2/S3
records, so the matched/distractor ratio is preserved.

  python -m er.make_dev_slice --train-dir dataset/train --out data/dev10
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.10)
    a = ap.parse_args()
    D, O = a.train_dir.rstrip("/") + "/", a.out.rstrip("/") + "/"
    os.makedirs(O, exist_ok=True)
    rng = np.random.default_rng(0)
    rd = dict(sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(D + "train_ground_truth.tsv", **rd)
    gts = gt.sample(frac=a.frac, random_state=0)
    gts.to_csv(O + "ground_truth.tsv", sep="\t", index=False)
    keep = {x for lst in gts.matched_entity_ids for x in lst.split(",") if x}
    allm = {x for lst in gt.matched_entity_ids for x in lst.split(",") if x}
    del gt
    s1 = pd.read_csv(D + "train_source1.tsv", **rd)
    s1[s1.entity_id.isin(set(gts.source1_entity_id))].to_csv(O + "source1.tsv", sep="\t", index=False)
    del s1
    for s in ("source2", "source3"):
        parts = []
        for ch in pd.read_csv(D + f"train_{s}.tsv", chunksize=1_000_000, **rd):
            unmatched = ~ch.entity_id.isin(allm)
            parts.append(ch[ch.entity_id.isin(keep) | (unmatched & (rng.random(len(ch)) < a.frac))])
        pd.concat(parts).to_csv(O + f"{s}.tsv", sep="\t", index=False)
    print("dev slice written to", O)


if __name__ == "__main__":
    main()
