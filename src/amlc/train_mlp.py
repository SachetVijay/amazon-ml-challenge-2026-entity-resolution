"""Fusion MLP over embeddings + tabular features (cheap, strong, complements GBDT).

  python -m amlc.train_mlp --emb qwen3e06 siglip2b --feats hand knn_qwen3e06 --name mlp_v1
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from .data import cfg, folds, fwd, inv, load_matrix, save_preds, y_raw
from .metrics import greater_is_better, score


def blocks(embs, feats):
    tr, te = [], []
    for n in embs:
        a, b = load_matrix("emb", n)
        tr.append(a.astype(np.float32)), te.append(b.astype(np.float32))
    for n in feats:
        a, b = load_matrix("feats", n)
        a, b = [np.nan_to_num(np.sign(x) * np.log1p(np.abs(x))).astype(np.float32) for x in (a, b)]
        mu, sd = a.mean(0), a.std(0) + 1e-6
        tr.append((a - mu) / sd), te.append((b - mu) / sd)
    return tr, te


class Net(nn.Module):
    def __init__(self, dims, hid=512, drop=0.2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hid), nn.GELU()) for d in dims])
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hid * len(dims), hid), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(hid, 1))

    def forward(self, xs):
        return self.head(torch.cat([p(x) for p, x in zip(self.proj, xs)], -1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=[])
    ap.add_argument("--feats", nargs="*", default=["hand"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--name", default="mlp")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_b, te_b = blocks(a.emb, a.feats)
    y, z, f = y_raw(), fwd(y_raw()), folds()
    zm, zs = z.mean(), z.std()
    zn = (z - zm) / zs
    T = lambda bl, idx: [torch.from_numpy(b[idx]).to(dev) for b in bl]  # noqa: E731
    te_t = [torch.from_numpy(b).to(dev) for b in te_b]
    gib = greater_is_better()
    oof, test = np.zeros(len(y)), np.zeros(len(te_b[0]))
    ks = np.unique(f)
    for k in ks:
        tri, vai = np.where(f != k)[0], np.where(f == k)[0]
        xv = T(tr_b, vai)
        for s in range(a.seeds):
            torch.manual_seed(cfg()["seed"] + s)
            net = Net([b.shape[1] for b in tr_b]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * (len(tri) // a.bs + 1))
            best, best_v, best_t = None, None, None
            for ep in range(a.epochs):
                net.train()
                for bi in np.array_split(np.random.permutation(tri), len(tri) // a.bs + 1):
                    loss = nn.functional.huber_loss(net(T(tr_b, bi)), torch.from_numpy(zn[bi]).float().to(dev), delta=1.0)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
                net.eval()
                with torch.no_grad():
                    pv = net(xv).cpu().numpy() * zs + zm
                sc = score(y[vai], inv(pv))
                if best is None or (sc > best if gib else sc < best):
                    with torch.no_grad():
                        best, best_v, best_t = sc, pv, net(te_t).cpu().numpy() * zs + zm
            oof[vai] += best_v / a.seeds
            test += best_t / (a.seeds * len(ks))
        print(f"fold {k}: {score(y[vai], inv(oof[vai])):.4f}", flush=True)
    print(f"CV {a.name}: {score(y, inv(oof)):.4f}")
    save_preds(a.name, inv(oof), inv(test))


if __name__ == "__main__":
    main()
