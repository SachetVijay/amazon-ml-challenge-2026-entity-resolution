"""Out-of-fold nearest-neighbour target features from an embedding.

  python -m amlc.knn_feats --emb qwen3 --k 20      -> artifacts/feats/knn_qwen3_*.npy

For each train row, neighbours come only from *other* folds (no leakage); test rows
use all of train. Features: sim-weighted mean / median / std of neighbour targets
(transformed space), top-1 sim + target, mean sim.
"""
import argparse

import faiss
import numpy as np

from .data import folds, fwd, load_matrix, save_matrix, y_raw


def _norm(x):
    x = np.ascontiguousarray(x.astype(np.float32))
    faiss.normalize_L2(x)
    return x


def _feats(sims, idx, z_pool):
    t = z_pool[idx]
    w = np.clip(sims, 1e-6, None) ** 4
    wmean = (w * t).sum(1) / w.sum(1)
    return np.column_stack([wmean, np.median(t, 1), t.std(1), t[:, :5].mean(1), sims[:, 0], t[:, 0], sims.mean(1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    e_tr, e_te = load_matrix("emb", a.emb)
    e_tr, e_te = _norm(e_tr), _norm(e_te)
    z, f = fwd(y_raw()), folds()
    out_tr = np.zeros((len(z), 7), np.float32)
    for k in np.unique(f):
        pool = np.where(f != k)[0]
        ix = faiss.IndexFlatIP(e_tr.shape[1])
        ix.add(e_tr[pool])
        s, i = ix.search(e_tr[f == k], a.k)
        out_tr[f == k] = _feats(s, i, z[pool])
    ix = faiss.IndexFlatIP(e_tr.shape[1])
    ix.add(e_tr)
    s, i = ix.search(e_te, a.k)
    out_te = _feats(s, i, z).astype(np.float32)
    save_matrix("feats", f"knn_{a.emb}", out_tr, out_te)
    print("knn feats", out_tr.shape, "corr(wmean, z)=", np.corrcoef(out_tr[:, 0], z)[0, 1].round(4))


if __name__ == "__main__":
    main()
