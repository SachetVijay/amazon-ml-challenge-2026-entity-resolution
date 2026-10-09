"""Fast, strong text baseline: Ridge on sparse word+char TF-IDF (+ scaled hand feats).

  python -m amlc.train_ridge --alpha 2.0
"""
import argparse

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import ART, folds, fwd, inv, load_matrix, save_preds, y_raw
from .metrics import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--name", default="ridge_tfidf")
    a = ap.parse_args()
    X_tr = sp.load_npz(ART / "feats" / "tfidf_train.npz")
    X_te = sp.load_npz(ART / "feats" / "tfidf_test.npz")
    h_tr, h_te = load_matrix("feats", "hand")
    h_tr, h_te = np.nan_to_num(np.sign(h_tr) * np.log1p(np.abs(h_tr))), np.nan_to_num(np.sign(h_te) * np.log1p(np.abs(h_te)))
    sc = StandardScaler().fit(h_tr)
    X_tr = sp.hstack([X_tr, sp.csr_matrix(sc.transform(h_tr))]).tocsr()
    X_te = sp.hstack([X_te, sp.csr_matrix(sc.transform(h_te))]).tocsr()

    y, z, f = y_raw(), fwd(y_raw()), folds()
    oof, test = np.zeros(len(y)), np.zeros(X_te.shape[0])
    for k in np.unique(f):
        tr, va = f != k, f == k
        m = Ridge(alpha=a.alpha, solver="sag", max_iter=2000, random_state=0).fit(X_tr[tr], z[tr])
        oof[va] = m.predict(X_tr[va])
        test += m.predict(X_te) / len(np.unique(f))
        print(f"fold {k}: {score(y[va], inv(oof[va])):.4f}")
    print(f"CV {a.name}: {score(y, inv(oof)):.4f}")
    save_preds(a.name, inv(oof), inv(test))


if __name__ == "__main__":
    main()
