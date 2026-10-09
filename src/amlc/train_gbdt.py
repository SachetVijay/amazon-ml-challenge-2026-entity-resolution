"""LightGBM on any combination of feature blocks / embeddings, 5-fold OOF.

  python -m amlc.train_gbdt --feats hand tfidf_svd knn_qwen3 --emb qwen3 siglip --name lgb_v1
  python -m amlc.train_gbdt --feats hand --objective huber --name lgb_huber
"""
import argparse

import lightgbm as lgb
import numpy as np

from .data import cfg, folds, fwd, inv, load_matrix, save_preds, y_raw
from .metrics import score


def build(feats, embs, pca_dim):
    trs, tes = [], []
    for n in feats:
        a, b = load_matrix("feats", n)
        trs.append(a), tes.append(b)
    for n in embs:
        a, b = load_matrix("emb", n)
        a, b = a.astype(np.float32), b.astype(np.float32)
        if pca_dim and a.shape[1] > pca_dim:  # trees struggle with 1-4k dense dims; PCA keeps it tractable
            from sklearn.decomposition import PCA
            p = PCA(pca_dim, random_state=0).fit(np.vstack([a, b]))
            a, b = p.transform(a), p.transform(b)
        trs.append(a), tes.append(b)
    return np.hstack(trs).astype(np.float32), np.hstack(tes).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="*", default=["hand", "tfidf_svd"])
    ap.add_argument("--emb", nargs="*", default=[])
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--objective", default="regression", help="regression|huber|l1|fair ...")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--name", default="lgb")
    a = ap.parse_args()

    X_tr, X_te = build(a.feats, a.emb, a.pca_dim)
    y, z, f = y_raw(), fwd(y_raw()), folds()
    print("X", X_tr.shape)
    params = dict(objective=a.objective, learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=40,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  metric="l1", verbose=-1, seed=cfg()["seed"], num_threads=0)
    if a.gpu:
        params["device_type"] = "cuda"
    oof, test = np.zeros(len(y)), np.zeros(len(X_te))
    ks = np.unique(f)
    for k in ks:
        tr, va = f != k, f == k
        m = lgb.train(params, lgb.Dataset(X_tr[tr], z[tr]), a.rounds, valid_sets=[lgb.Dataset(X_tr[va], z[va])],
                      callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X_tr[va], num_iteration=m.best_iteration)
        test += m.predict(X_te, num_iteration=m.best_iteration) / len(ks)
        print(f"fold {k}: it={m.best_iteration} score={score(y[va], inv(oof[va])):.4f}", flush=True)
    print(f"CV {a.name}: {score(y, inv(oof)):.4f}")
    save_preds(a.name, inv(oof), inv(test))


if __name__ == "__main__":
    main()
