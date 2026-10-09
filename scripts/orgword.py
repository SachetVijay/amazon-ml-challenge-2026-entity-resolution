"""Org-type word ADDED by the record (absent from the S1 name): how predictive of non-match?"""
import re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from anyascii import anyascii
from er.predict import country_rows
ORG = {  # organisation-type words (EN + FR); fillers / legal forms deliberately excluded
 "club", "amicale", "ecole", "college", "lycee", "comite", "association", "amis", "parents", "union", "federation",
 "maison", "sportive", "culturelle", "residence", "cabinet", "boulangerie", "pharmacie", "clinique", "institut",
 "fondation", "restaurant", "cafe", "hotel", "garage", "clinic", "school", "church", "ministries", "academy",
 "university", "hospital", "pharmacy", "bakery", "salon", "barbershop", "studio", "gym", "fitness", "dental",
 "dentistry", "law", "realty", "construction", "trading", "foods", "exports", "textiles", "motors", "farm",
 "foundation", "society", "trust", "institute", "apartments", "consultancy", "technologies", "tavern", "bar",
 "hospitality", "outsourcing", "developers", "builders", "industries", "logistics", "media", "infra", "infrastructure"}
tok = lambda s: set(re.findall(r"[a-z]+", anyascii(str(s)).lower()))
def added_org(a, b):
    ta, tb = tok(a), tok(b)
    return bool((tb - ta) & ORG)
# (a) US/India OOF: label rate of pairs where the record adds an org word, by decision
art = "artifacts/train"
D = pd.read_parquet(f"{art}/oof_stage2_hyb5_v9.parquet", columns=["a", "b", "country", "p2", "y"])
Bd = pd.read_parquet(f"{art}/oof_band_v11l.parquet", columns=["a", "b", "p"])
D = D.merge(Bd, on=["a", "b"], how="left"); D["p"] = D.p.fillna(D.p2)
D = D[D.p > 0.05]
rec = pd.read_parquet(f"{art}/records.parquet", columns=["business_name"])
D["org_added"] = [added_org(x, y) for x, y in zip(rec.business_name.to_numpy()[D.a.to_numpy()], rec.business_name.to_numpy()[D.b.to_numpy()])]
for c, g in D.groupby("country"):
    for nm, m in (("record ADDS org word", g.org_added), ("no added org word", ~g.org_added)):
        h = g[m]
        print(f"OOF {c:5s} {nm:22s}: pairs p>0.05 {len(h):7d} | true-match rate {h.y.mean():.3f} | "
              f"among p>0.5: n={int((h.p > .5).sum()):6d} precision {h.y[h.p > .5].mean():.3f}")
# (b) France test (v9 France decisions = current best): how many predicted links add an org word?
tart = "artifacts/test"
rows = country_rows(f"{tart}/records.parquet")["France"]
r = pd.read_parquet(f"{tart}/records.parquet", columns=["business_name"], filters=[("country", "==", "France")]); r.index = rows
P = pd.read_parquet(f"{tart}/pred_s2_France.parquet")
P = P.loc[P.groupby("b").p.idxmax()]
P = P[P.p > 0.05]
P["org_added"] = [added_org(x, y) for x, y in zip(r.business_name.loc[P.a].to_numpy(), r.business_name.loc[P.b].to_numpy())]
print(f"\nFRANCE test winners p>0.05: {len(P)} | record adds org word: {P.org_added.sum()} "
      f"| of those p>0.5: {int((P.org_added & (P.p > .5)).sum())} | p in 0.5-0.95: {int((P.org_added & (P.p > .5) & (P.p < .95)).sum())}")
print("examples:"); 
for x in P[P.org_added & (P.p > .5)].sample(8, random_state=0).itertuples():
    print(f"  p={x.p:.2f}  {r.business_name.loc[x.a]}  <->  {r.business_name.loc[x.b]}")
