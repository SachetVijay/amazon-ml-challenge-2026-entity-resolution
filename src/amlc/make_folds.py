"""Create the single fold assignment every model shares (required for honest stacking).

  python -m amlc.make_folds [--group-col COL]

Regression targets are stratified on quantile bins; --group-col keeps near-duplicate
products (same brand/title/group id) in one fold so CV isn't optimistic.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import ART, cfg, train_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-col", default=None)
    a = ap.parse_args()
    c, df = cfg(), train_df()
    y = df[c["target_col"]]
    strat = pd.qcut(y.rank(method="first"), 20, labels=False) if c["task"] == "regression" else y.astype(str)
    fold = np.full(len(df), -1, dtype=np.int16)
    if a.group_col:
        kf = StratifiedGroupKFold(c["n_folds"], shuffle=True, random_state=c["seed"])
        splits = kf.split(df, strat, groups=df[a.group_col])
    else:
        splits = StratifiedKFold(c["n_folds"], shuffle=True, random_state=c["seed"]).split(df, strat)
    for k, (_, va) in enumerate(splits):
        fold[va] = k
    ART.mkdir(exist_ok=True)
    np.save(ART / "folds.npy", fold)
    print(pd.Series(fold).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
