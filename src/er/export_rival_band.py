"""Rival band pairs for er.exp_band_rival: rival S1 pairs (er.build_competitors) inside the stage-1 band, to be
scored by the cross-encoders.   python -m er.export_rival_band --art artifacts/train --comp-tag fd"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--comp-tag", default="fd")
    ap.add_argument("--band", type=float, nargs=2, default=[0.005, 0.995])
    a = ap.parse_args()
    files = sorted(Path(a.art).glob(f"comp_{a.comp_tag}_*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["a", "b", "p"]) for f in files], ignore_index=True)
    d = d[(d.p > a.band[0]) & (d.p < a.band[1])].reset_index(drop=True)
    d.to_parquet(Path(a.art) / "rival_band.parquet")
    print(f"rival band pairs {len(d)} from {len(files)} files")


if __name__ == "__main__":
    main()
