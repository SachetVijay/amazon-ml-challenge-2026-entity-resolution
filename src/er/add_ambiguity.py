"""Add name-ambiguity columns to cached training feature files (avoids recomputing all features).

  python -m er.add_ambiguity --art artifacts/train --tag _hyb5
"""
import argparse
from pathlib import Path

import pandas as pd

from .features import name_ambiguity
from .predict import country_rows, read_country


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
        sub = read_country(path, country, rows)[["src", "country", "name_core", "name_skel"]].astype(
            {"country": object, "name_core": object, "name_skel": object})
        s1 = sub[sub.src.to_numpy() == 1]
        freq = s1.groupby(["country", "name_core"]).size()
        skel = s1.groupby(["country", "name_skel"]).size()
        X = pd.read_parquet(fp)
        B = sub.loc[X.b.to_numpy()]
        for k, v in name_ambiguity(B.country, B.name_core, B.name_skel, freq, skel).items():
            X[k] = v
        X.to_parquet(fp)
        print(country, "updated", X.shape, flush=True)


if __name__ == "__main__":
    main()
