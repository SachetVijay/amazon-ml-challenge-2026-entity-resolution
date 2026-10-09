"""Add location-stripped name features to cached training feature files (no full rebuild).

  python -m er.add_location --art artifacts/train --tag _hyb5
"""
import argparse
from pathlib import Path

import pandas as pd

from .features import location_feats
from .predict import country_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--tag", default="_hyb5")
    a = ap.parse_args()
    path = Path(a.art) / "records.parquet"
    for country, rows in country_rows(path).items():
        fp = Path(a.art) / f"feats{a.tag}_{country}.parquet"
        if not fp.exists():
            continue
        sub = pd.read_parquet(path, columns=["name_clean", "addr_words"], filters=[("country", "==", country)])
        sub.index = rows
        X = pd.read_parquet(fp)
        A, B = sub.loc[X.a.to_numpy()], sub.loc[X.b.to_numpy()]
        F = location_feats(A.name_clean.astype(str).tolist(), B.name_clean.astype(str).tolist(),
                           A.addr_words.astype(str).tolist(), B.addr_words.astype(str).tolist(), 20, 200_000)
        for k, v in F.items():
            X[k] = v
        X.to_parquet(fp)
        print(country, "updated", X.shape, flush=True)
        del X, A, B, sub


if __name__ == "__main__":
    main()
