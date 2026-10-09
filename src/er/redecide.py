"""Re-run only the decision step on saved stage-2 probabilities, with an odds-scaling calibration factor r:
    logit(p') = logit(p) + log(r)      (r < 1: more conservative, r > 1: more liberal)
Used to diagnose test-set miscalibration (e.g. prior shift from a higher distractor rate) via the leaderboard.
Per-country factors allowed (open label set).

  python -m er.redecide --art artifacts/test --r 0.5 --out output_v6c
  python -m er.redecide --art artifacts/test --r 1.0 --r-country France=0.5 --out output_v6f
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .decide import exclusive, expected_f_fast
from .predict import append_lists, country_rows, read_country


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--r-country", nargs="*", default=[], help="COUNTRY=r overrides")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rc = {k: float(v) for k, v in (x.split("=") for x in a.r_country)}
    path = Path(a.art) / "records.parquet"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fm, fc = open(out / "matching_results.tsv", "w"), open(out / "candidate_pairs.tsv", "w")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    tot_s1 = tot_m = 0
    for country, rows in country_rows(path).items():
        r = rc.get(country, a.r)
        P = pd.read_parquet(Path(a.art) / f"pred_s2_{country}.parquet")
        cand = pd.read_parquet(Path(a.art) / f"pred_hyb_{country}.cand.parquet")
        p = P.p.clip(1e-7, 1 - 1e-7).to_numpy()
        P["p"] = 1 / (1 + (1 - p) / (p * r))                     # odds * r
        ex = exclusive(P)
        sel = expected_f_fast(ex)
        sub = pd.read_parquet(path, columns=["entity_id", "src"], filters=[("country", "==", country)])
        sub.index = rows
        ids = pd.Series(sub.entity_id.to_numpy(dtype=object), index=sub.index)
        s1_rows = np.sort(sub.index[sub.src.to_numpy() == 1].to_numpy())
        append_lists(fm, s1_rows, sel.a.to_numpy(), sel.b.to_numpy(), ids)
        append_lists(fc, s1_rows, cand.a.to_numpy(), cand.b.to_numpy(), ids)
        tot_s1 += len(s1_rows)
        tot_m += len(sel)
        print(f"{country}: r={r} -> {len(sel)} matches ({len(sel) / len(s1_rows):.3f}/S1)", flush=True)
    fm.close()
    fc.close()
    print(f"wrote {out}: {tot_m / tot_s1:.3f} matches/S1")


if __name__ == "__main__":
    main()
