"""Adapt a cross-encoder to a country absent from training, with pseudo-labels only (no labels of that country).

The raw-text cross-encoders were trained on US/India wording; on the unseen country they reject 7-20x more of the
links that the (cross-encoder-free) stage-1 model is certain about. Pseudo-labels from the finished pipeline's own
predictions on the unseen country:
  positives : pairs with stage-1 p1 >= 0.995 and final p >= 0.98 (oversampling those the cross-encoder doubts)
  negatives : the planted look-alike pattern (record = S1 name + a qualifier word, different house number; 3.9 % true
              in US/India) and clear rejects (p1 < 0.005, final p < 0.002)
mixed with 200k labelled US/India hard pairs so the model does not forget. Fine-tune with er/kaggle_train_ce.py, then
predict with  --unseen-ce-override J=<adapted model>.

  python -m er.adapt_unseen_ce --art artifacts/test --country France --pred-s2 <dir with pred_s2_France.parquet> \
      --ce-model models/ce_base2 --us-pairs hard_b45.parquet --out adapt_fr
"""
import argparse
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .cross_encoder import load_texts, score
from .normalize import ADDR_ABBR, LEGAL, to_ascii

STOP = {"no", "number", "n", "deg", "bis", "ter", "de", "du", "des", "la", "le", "les", "d", "l", "unit", "apt", "suite"}
def street(addr):
    for comp in (addr or "").split(","):
        s = to_ascii(comp).lower(); m = re.search(r"\d+", s)
        if m: return m.group().lstrip("0"), " ".join(w for w in (ADDR_ABBR.get(t, t) for t in re.findall(r"[a-z]+", s)) if w not in STOP)
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--pred-s2", required=True, help="dir holding pred_s2_<country>.parquet of the finished pipeline")
    ap.add_argument("--ce-model", required=True, help="cross-encoder to adapt (its doubts pick the hard positives)")
    ap.add_argument("--us-pairs", required=True, help="labelled training-country pairs (text_a, text_b, y)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pathlib import Path
    Path(a.out).mkdir(parents=True, exist_ok=True)
    art, c = a.art, a.country
    rng = np.random.default_rng(0)
    p1 = pd.read_parquet(f"{art}/pred_hyb_{c}.parquet", columns=["a", "b", "p"]).rename(columns={"p": "p1"})
    fin = pd.read_parquet(f"{a.pred_s2}/pred_s2_{c}.parquet")
    D = p1[p1.p1 > 1e-3].merge(fin, on=["a", "b"])
    pos = D[(D.p1 >= 0.995) & (D.p >= 0.98)].sample(260000, random_state=0)
    held, pos = pos.iloc[:20000], pos.iloc[20000:]
    pos[["a", "b"]].to_parquet(f"{a.out}/pos.parquet")
    score(SimpleNamespace(pairs=f"{a.out}/pos.parquet", band=None, art=art, model=a.ce_model, out=f"{a.out}/pos_ce.parquet", max_len=128, view="raw"))
    pos = pos.merge(pd.read_parquet(f"{a.out}/pos_ce.parquet"), on=["a", "b"])
    hard = pos[pos.ce < 3]; easy = pos[pos.ce >= 3].sample(min(60000, (pos.ce >= 3).sum()), random_state=0)
    print(f"positives: ce1 doubts (logit<3) {len(hard)}, easy sample {len(easy)}", flush=True)
    # negatives
    rows = np.flatnonzero(pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["src", "name_clean", "business_address"], filters=[("country", "==", c)]); r.index = rows
    toks = lambda s: {t for t in s.split() if t not in LEGAL and len(t) > 1}
    sdf = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
    odf = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
    ratio = (odf + 1e-4) / (sdf.reindex(odf.index).fillna(0) + 1e-4)
    N = D[(D.p < 0.01)]
    N = N.sample(min(600000, len(N)), random_state=1)
    A, B = r.name_clean.astype(object).loc[N.a].tolist(), r.name_clean.astype(object).loc[N.b].tolist()
    qual = np.array([bool(toks(y) - toks(x)) and not (toks(x) - toks(y)) and max(ratio.get(t, 1) for t in toks(y) - toks(x)) >= 20 for x, y in zip(A, B)])
    N = N[qual]
    na = [street(x)[0] for x in r.business_address.astype(object).loc[N.a]]; nb = [street(x)[0] for x in r.business_address.astype(object).loc[N.b]]
    shadow = N[[x != y and x != "" and y != "" for x, y in zip(na, nb)]].head(40000)
    clear = D[(D.p1 < 0.005) & (D.p < 0.002)].sample(60000, random_state=2)
    print(f"negatives: shadow pattern {len(shadow)}, clear rejects {len(clear)}", flush=True)
    F = pd.concat([hard[["a", "b"]].assign(y=1.0), easy[["a", "b"]].assign(y=1.0), shadow[["a", "b"]].assign(y=0.0), clear[["a", "b"]].assign(y=0.0)])
    txt = load_texts(art, np.concatenate([F.a.to_numpy(), F.b.to_numpy()]))
    F["text_a"] = [txt[int(x)] for x in F.a]; F["text_b"] = [txt[int(x)] for x in F.b]
    U = pd.read_parquet(a.us_pairs).sample(200000, random_state=3)
    M = pd.concat([F[["text_a", "text_b", "y"]], U[["text_a", "text_b", "y"]]]).sample(frac=1, random_state=4)
    M.to_parquet(f"{a.out}/pairs.parquet"); held[["a", "b"]].to_parquet(f"{a.out}/held_pos.parquet")
    print(f"adaptation set {len(M)} pairs (France {len(F)}, US/India {len(U)}), pos rate {M.y.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
