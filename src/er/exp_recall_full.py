"""Full-density blocking recall@k for one country (memory-lean: keeps only rows of sampled S1s).

  python -m er.exp_recall_full --art artifacts/train --gt .../train_ground_truth.tsv --country India --k 40
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .blocking import RETRIEVERS, _topk, texts_for
from .predict import country_rows, read_country
from .prep import load_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--country", default="India")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--sample-mod", type=int, default=8)
    ap.add_argument("--max-df", type=int, default=None)
    a = ap.parse_args()
    T = time.time()
    path = f"{a.art}/records.parquet"
    rows = country_rows(path)[a.country]
    sub = read_country(path, a.country, rows)
    ids = pd.Series(sub.entity_id.to_numpy(dtype=object), index=sub.index)
    src = sub.src.to_numpy()
    a_idx, b_idx = sub.index[src == 1].to_numpy(), sub.index[src != 1].to_numpy()
    samp = (pd.util.hash_array(ids.loc[a_idx].to_numpy()) % a.sample_mod) == 0
    gt = load_gt(a.gt)
    gt = gt[gt.s1.isin(set(ids.loc[a_idx[samp]]))]
    g = set(zip(gt.s1, gt.other))
    print(f"{a.country}: {len(a_idx)} S1, {len(b_idx)} other, sampled S1 {samp.sum()}, GT pairs {len(g)}", flush=True)

    kw = dict(RETRIEVERS["all"])
    if a.max_df:
        kw["max_df"] = a.max_df
    X = TfidfVectorizer(**kw).fit_transform(texts_for(sub, "all"))
    pos = pd.Series(np.arange(len(sub)), index=sub.index)
    XA, XB = X[pos.loc[a_idx].to_numpy()], X[pos.loc[b_idx].to_numpy()]
    print(f"tfidf nnz/row {X.nnz / X.shape[0]:.1f} {time.time() - T:.0f}s", flush=True)

    # b2a: every record retrieves top-k S1; keep only hits on sampled S1
    q, d, s = _topk(XB, XA, a.k, 0.0, 22)
    keep = samp[d]
    b2a = pd.DataFrame({"a": a_idx[d[keep]], "b": b_idx[q[keep]], "s": s[keep]})
    # rank of that S1 within the record's list (lists are sorted by score)
    starts = np.r_[0, np.cumsum(np.bincount(q, minlength=len(b_idx)))[:-1]]
    b2a["r"] = (np.arange(len(q)) - starts[q])[keep] + 1
    # a2b: sampled S1 retrieve top-k records
    q2, d2, s2 = _topk(XA[samp], XB, a.k, 0.0, 22)
    a_s = a_idx[samp]
    starts2 = np.r_[0, np.cumsum(np.bincount(q2, minlength=len(a_s)))[:-1]]
    a2b = pd.DataFrame({"a": a_s[q2], "b": b_idx[d2], "s": s2, "r": np.arange(len(q2)) - starts2[q2] + 1})
    print(f"search done {time.time() - T:.0f}s", flush=True)
    for df in (b2a, a2b):
        df["hit"] = [(x, y) in g for x, y in zip(ids.loc[df.a].to_numpy(), ids.loc[df.b].to_numpy())]
    for k in (1, 3, 5, 10, 20, 40):
        if k > a.k:
            break
        hb = set(map(tuple, b2a.loc[(b2a.r <= k) & b2a.hit, ["a", "b"]].to_numpy()))
        ha = set(map(tuple, a2b.loc[(a2b.r <= k) & a2b.hit, ["a", "b"]].to_numpy()))
        n_pairs = ((b2a.r <= k).sum() + (a2b.r <= k).sum()) / samp.sum()
        print(f"k={k:2d}: b2a {len(hb) / len(g):.4f}  a2b {len(ha) / len(g):.4f}  union {len(hb | ha) / len(g):.4f}"
              f"  (~{n_pairs:.0f} raw pairs/S1)")
    # which way do misses go: true S1's score vs the record's best score at k=5
    miss = b2a[b2a.hit & (b2a.r > 5)]
    print("b2a hits beyond rank 5:", len(miss), "median rank", miss.r.median())


if __name__ == "__main__":
    main()
