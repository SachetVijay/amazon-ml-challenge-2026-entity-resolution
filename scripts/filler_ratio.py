"""Filler score per token and country = df among S2/S3 names / df among S1 names (legal forms excluded)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
def df(names):
    s = pd.Series([t for n in names for t in set(n.split()) if t not in LEGAL])
    return s.value_counts() / len(names)
for split, countries in (("train", ("US", "India")), ("test", ("US", "India", "France"))):
    for c in countries:
        r = pd.read_parquet(f"artifacts/{split}/records.parquet", columns=["src", "name_clean"], filters=[("country", "==", c)])
        d1, d2 = df(r.name_clean[r.src == 1].astype(object).tolist()), df(r.name_clean[r.src != 1].astype(object).tolist())
        T = pd.DataFrame({"s1": d1, "rec": d2}).fillna(0)
        T = T[(T.s1 >= 0.002) | (T.rec >= 0.002)]
        T["ratio"] = (T.rec + 1e-4) / (T.s1 + 1e-4)
        show = ["services", "service", "center", "centre", "group", "groupe", "enterprises", "solutions", "company", "cie", "compagnie",
                "developpement", "fils", "freres", "societe", "france", "india", "club", "ecole", "amicale", "comite", "amis",
                "groupement", "sportive", "lycee", "ehpad", "association", "pharmacie", "clinique", "partners", "sri", "shri", "the"]
        s = " ".join(f"{t}:{T.ratio[t]:.1f}" for t in show if t in T.index)
        print(f"{split} {c:6s} | {s}", flush=True)
        print(f"   top ratio tokens: " + ", ".join(f"{t}:{v:.1f}" for t, v in T.ratio.sort_values(ascending=False).head(15).items()), flush=True)
