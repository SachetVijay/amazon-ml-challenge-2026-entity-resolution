"""Validation (labels): one-token swap pairs (record adds exactly one token and drops exactly one token vs the S1
name, legal forms ignored). Truth rate by filler score of the ADDED and DROPPED token (score = df_records/df_S1,
per country, label-free). Also pairs that only ADD one token / only DROP one token."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
def toks(s): return {t for t in s.split() if t not in LEGAL}
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
full = pd.read_parquet("artifacts/train/oof_band_v12.parquet")
oof = pd.read_parquet("artifacts/train/oof_hyb5.parquet", columns=["a", "country"]).drop_duplicates("a")
full = full.merge(oof, on="a")
rows_all = pd.read_parquet("artifacts/train/records.parquet", columns=["country"]).country.to_numpy()
for c in ("US", "India"):
    r = pd.read_parquet("artifacts/train/records.parquet", columns=["src", "name_clean", "addr_clean"], filters=[("country", "==", c)])
    r.index = np.flatnonzero(rows_all == c)
    s = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
    o = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
    ratio = ((o.reindex(s.index.union(o.index)).fillna(0) + 1e-4) / (s.reindex(s.index.union(o.index)).fillna(0) + 1e-4))
    F = full[(full.country == c) & (full.p > 1e-3)].reset_index(drop=True)
    A = r.name_clean.astype(object).loc[F.a].tolist(); B = r.name_clean.astype(object).loc[F.b].tolist()
    kind, add_r, drop_r = [], [], []
    for x, y in zip(A, B):
        tx, ty = toks(x), toks(y); dx, dy = tx - ty, ty - tx
        if len(dx) == 1 and len(dy) == 1 and tx & ty: kind.append("swap"); add_r.append(ratio.get(next(iter(dy)), 1)); drop_r.append(ratio.get(next(iter(dx)), 1))
        elif not dx and len(dy) == 1 and tx: kind.append("add"); add_r.append(ratio.get(next(iter(dy)), 1)); drop_r.append(np.nan)
        else: kind.append(""); add_r.append(np.nan); drop_r.append(np.nan)
    F["kind"], F["add_r"], F["drop_r"] = kind, add_r, drop_r
    F["y"] = np.fromiter(((s_, o_) in ys for s_, o_ in zip(F.s1_id, F.o_id)), bool, len(F))
    G = F[F.kind != ""].copy()
    G["addr"] = cpdist(r.addr_clean.astype(object).loc[G.a].tolist(), r.addr_clean.astype(object).loc[G.b].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    G = G[G.addr >= 90]
    G["cls"] = pd.cut(G.add_r, [0, 1.5, 2.5, 1e9], labels=["native(<1.5)", "mid", "filler(>=2.5)"])
    for k in ("swap", "add"):
        H = G[G.kind == k]
        t = H.groupby("cls", observed=True).agg(n=("y", "size"), true_rate=("y", "mean"), model_p=("p", "median"))
        print(f"{c} {k:4s} (same address) by filler score of the ADDED token:\n{t.round(3).to_string()}", flush=True)
    H = G[G.kind == "swap"].copy()
    toks_ab = [(next(iter(toks(x) - toks(y))), next(iter(toks(y) - toks(x)))) for x, y in zip(r.name_clean.astype(object).loc[H.a], r.name_clean.astype(object).loc[H.b])]
    H["drop_df"] = [s.get(u, 0) for u, v in toks_ab]; H["add_df"] = [s.get(v, 0) for u, v in toks_ab]
    H["pair"] = [f"{u}->{v}" for u, v in toks_ab]
    for thr in (0.001, 0.003):
        m = (H.add_r < 1.5) & (H.drop_r < 1.5) & (H.add_df >= thr) & (H.drop_df >= thr)
        print(f"{c} swap, both tokens native AND frequent in S1 names (df>={thr}): n={m.sum()} true rate {H.y[m].mean():.3f} "
              f"model p median {H.p[m].median():.3f}", flush=True)
    m = (H.add_r < 1.5) & (H.drop_r < 1.5) & (H.add_df >= 0.001) & (H.drop_df >= 0.001)
    top = H[m].groupby("pair").agg(n=("y", "size"), true=("y", "mean"), p=("p", "median")).sort_values("n", ascending=False).head(12)
    print(top.round(3).to_string() + "\n", flush=True)
