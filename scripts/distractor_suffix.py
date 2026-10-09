"""Words with the DISTRACTOR fingerprint (mostly pure appends, ~no same address) among v22 links, per country."""
import sys
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
def prof(na, nb):
    ta = set(na.split()); w = nb.split()
    if not w or w[-1] in ta or w[-1] in LEGAL or len(w[-1]) < 2: return None
    drop = {t for t in ta - set(w[:-1]) if t not in LEGAL and len(t) > 1}
    return w[-1], not drop
rt = pd.read_parquet("artifacts/test/records.parquet", columns=["entity_id", "name_clean", "addr_clean", "country", "src"])
nm, ad = rt.name_clean.astype(object).to_numpy(), rt.addr_clean.astype(object).to_numpy()
idx = pd.Series(rt.index, index=rt.entity_id.astype(object)); cty = rt.country.to_numpy()
m = pd.read_csv("output_v22.tsv", sep="\t", dtype=str, keep_default_na=False)
st = defaultdict(list)
for s, l in zip(m.source1_entity_id, m.matched_entity_ids):
    if not l: continue
    a = idx[s]
    for o in l.split(","):
        b = idx[o]; x = prof(nm[a], nm[b])
        if x: st[(cty[a], x[0])].append((x[1], ad[a] == ad[b] and ad[a] != ""))
rows = [(c, w, len(v), np.mean([p for p, _ in v]), np.mean([q for _, q in v])) for (c, w), v in st.items() if len(v) >= 30]
D = pd.DataFrame(rows, columns=["country", "word", "n", "pure", "same_addr"])
print("v22 links, appended words with pure-append share >= 0.5 (distractor fingerprint), n>=30:")
print(D[D.pure >= 0.5].sort_values("n", ascending=False).head(25).to_string(index=False))
print("totals by country:", D[D.pure >= 0.5].groupby("country").n.sum().to_dict())
