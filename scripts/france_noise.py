"""Token-level noise mining: which tokens does the generator add/remove between S1 and its records?
Train: true pairs. France (test): confident predicted pairs (p>0.99) and uncertain ones (0.3<p<0.9)."""
import collections, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from anyascii import anyascii
from er.predict import country_rows
tok = lambda s: re.findall(r"[a-z0-9]+", anyascii(str(s)).lower())
def mine(pairs, rec, n=40000):
    pairs = pairs.sample(min(n, len(pairs)), random_state=0)
    A, B = rec.loc[pairs.a.to_numpy()], rec.loc[pairs.b.to_numpy()]
    add_n, rem_n, add_a, rem_a = (collections.Counter() for _ in range(4))
    for an, bn, aa, ba in zip(A.business_name, B.business_name, A.business_address, B.business_address):
        TA, TB = set(tok(an)), set(tok(bn)); add_n.update(TB - TA); rem_n.update(TA - TB)
        XA, XB = set(tok(aa)), set(tok(ba)); add_a.update(XB - XA); rem_a.update(XA - XB)
    k = len(pairs) / 100
    f = lambda c: ", ".join(f"{w}:{v / k:.1f}" for w, v in c.most_common(25))
    return f"  NAME +[{f(add_n)}]\n  NAME -[{f(rem_n)}]\n  ADDR +[{f(add_a)}]\n  ADDR -[{f(rem_a)}]"
cols = ["business_name", "business_address"]
# France test
path = "artifacts/test/records.parquet"
rows = country_rows(path)["France"]
rec = pd.read_parquet(path, columns=cols, filters=[("country", "==", "France")]); rec.index = rows
P = pd.read_parquet("artifacts/test/pred_s2_France.parquet")
best = P.loc[P.groupby("b").p.idxmax()]
print("FRANCE confident (p>0.99), per 100 pairs:\n" + mine(best[best.p > .99], rec))
print("\nFRANCE uncertain (0.3<p<0.9), per 100 pairs:\n" + mine(best[(best.p > .3) & (best.p < .9)], rec))
# US test confident for reference of generator behaviour in a known country
rows = country_rows(path)["US"]
rec = pd.read_parquet(path, columns=cols, filters=[("country", "==", "US")]); rec.index = rows
P = pd.read_parquet("artifacts/test/pred_s2_US.parquet")
best = P.loc[P.groupby("b").p.idxmax()]
print("\nUS confident (p>0.99), per 100 pairs:\n" + mine(best[best.p > .99], rec))
print("\nUS uncertain (0.3<p<0.9), per 100 pairs:\n" + mine(best[(best.p > .3) & (best.p < .9)], rec))
