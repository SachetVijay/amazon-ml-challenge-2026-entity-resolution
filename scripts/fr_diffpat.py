"""Pattern composition of France link differences between leaderboard-scored submissions."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
fr = set(c.entity_id[c.country == "France"])
def load(p):
    m = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    m = m[m.source1_entity_id.isin(fr)]
    return {(s, o) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) for o in l.split(",") if o}
V = {k: load(p) for k, p in (("v11", "output_v11/matching_results.tsv"), ("v11fL", "output_v11fL/matching_results.tsv"),
                              ("v11_v9FR", "output_v11_v9FR.tsv"), ("v12", "output_v12/matching_results.tsv"),
                              ("v13", "output_v13/matching_results.tsv"))}
r = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "src", "name_clean", "addr_clean", "addr_nums"], filters=[("country", "==", "France")])
R = r.set_index("entity_id")
s1 = r[r.src == 1]
df1 = pd.Series([t for n in s1.name_clean.astype(object) for t in {x for x in n.split() if x not in LEGAL}]).value_counts() / len(s1)
dfo = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in {x for x in n.split() if x not in LEGAL}]).value_counts() / (r.src != 1).sum()
ratio = (dfo.reindex(df1.index.union(dfo.index)).fillna(0) + 1e-4) / (df1.reindex(df1.index.union(dfo.index)).fillna(0) + 1e-4)
def pattern(s, o):
    A, B = R.loc[s], R.loc[o]
    ta = {x for x in A.name_clean.split() if x not in LEGAL}; tb = {x for x in B.name_clean.split() if x not in LEGAL}
    if B.addr_clean == "": return "address-less record"
    same_num = A.addr_nums.split()[:1] == B.addr_nums.split()[:1] and B.addr_nums != ""
    if not (ta & tb): return "no shared name word (coined/renamed)" + (" | same no." if same_num else " | diff no.")
    dx, dy = ta - tb, tb - ta
    if not dx and not dy: return "same name words" + (" | same no." if same_num else " | diff no.")
    if len(dx) == 1 and len(dy) == 1:
        u, v = next(iter(dx)), next(iter(dy))
        if df1.get(u, 0) >= 0.001 and df1.get(v, 0) >= 0.001:
            return "swap freq word -> " + ("filler(r>=2)" if ratio.get(v, 1) >= 2 else "native word")
        return "swap rare word (typo)"
    if not dx: return "record adds words" + (" filler" if all(ratio.get(v, 1) >= 2 for v in dy) else " native")
    if not dy: return "record drops words"
    return "other multi-word change"
def comp(tag, S):
    pats = pd.Series([pattern(s, o) for s, o in S]).value_counts()
    return (pats / len(S)).rename(f"{tag} (n={len(S)})")
rows = [comp("ALL v13 links", V["v13"]),
        comp("4CE-vs-6CE: added by 4CE (+0.003 FR)", V["v11_v9FR"] - V["v11"]), comp("4CE-vs-6CE: removed by 4CE", V["v11"] - V["v11_v9FR"]),
        comp("x2 probe added (-0.0021 FR)", V["v11fL"] - V["v11"]),
        comp("v12 added (~+0.002 FR)", V["v12"] - V["v11_v9FR"]), comp("v12 removed", V["v11_v9FR"] - V["v12"])]
T = pd.concat(rows, axis=1).fillna(0)
pd.set_option("display.width", 250)
print((T * 100).round(1).to_string())
