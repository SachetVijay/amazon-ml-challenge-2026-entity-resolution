"""Pseudo-labelled French pairs for adapting ce1 (base2) + labelled US/India hard pairs (no forgetting).
positives : France pairs with stage-1 p1 >= 0.995 and final p >= 0.98 (CE-free confidence); oversample those ce1 doubts.
negatives : France shadow pattern (record = S1 name + qualifier word, different house number; 3.9% true in US) with final p < 0.01,
            + France clear rejects (p1 < 0.005, final p < 0.002).
held-out  : 20k further positives for evaluation."""
import sys
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from er.cross_encoder import load_texts, score
from er.normalize import LEGAL
from street_mix import street
art = "artifacts/test"; c = "France"
rng = np.random.default_rng(0)
p1 = pd.read_parquet(f"{art}/pred_hyb_{c}.parquet", columns=["a", "b", "p"]).rename(columns={"p": "p1"})
fin = pd.read_parquet("backup_pred_s2_v14/pred_s2_France.parquet")
D = p1[p1.p1 > 1e-3].merge(fin, on=["a", "b"])
pos = D[(D.p1 >= 0.995) & (D.p >= 0.98)].sample(260000, random_state=0)
held, pos = pos.iloc[:20000], pos.iloc[20000:]
pos[["a", "b"]].to_parquet("/tmp/fr_pos.parquet")
score(SimpleNamespace(pairs="/tmp/fr_pos.parquet", band=None, art=art, model="models/ce_base2", out="/tmp/fr_pos_ce1.parquet", max_len=128, view="raw"))
pos = pos.merge(pd.read_parquet("/tmp/fr_pos_ce1.parquet"), on=["a", "b"])
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
U = pd.read_parquet("kaggle_hard/hard_b45.parquet").sample(200000, random_state=3)
M = pd.concat([F[["text_a", "text_b", "y"]], U[["text_a", "text_b", "y"]]]).sample(frac=1, random_state=4)
M.to_parquet("kaggle_frad/pairs.parquet"); held[["a", "b"]].to_parquet("kaggle_frad/held_pos.parquet")
print(f"adaptation set {len(M)} pairs (France {len(F)}, US/India {len(U)}), pos rate {M.y.mean():.3f}", flush=True)
