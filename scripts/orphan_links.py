"""How many LINKED records look like orphans?
 (A) address-less linked record r -> X, name N: orphan groups with name N = distinct (N, number) keys of >=2 UNLINKED
     records with address, no S1 with that key, number != X's number.
 (B) linked record with address r -> X whose same-key twin (same name_core + number) is UNLINKED and no S1 has that key
     ("split group").
train: 'linked' = GT matched (true links; baseline rates). test: v14 links."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.prep import load_gt
split = sys.argv[1]; art = f"artifacts/{split}"
if split == "train":
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); link = dict(zip(gt.other, gt.s1))
    countries = ("US", "India")
else:
    m = pd.read_csv("output_v14/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    link = {o: s for s, l in zip(m.source1_entity_id, m.matched_entity_ids) for o in l.split(",") if o}
    countries = ("US", "India", "France")
for c in countries:
    r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "name_core", "addr_clean", "addr_first_num"], filters=[("country", "==", c)])
    r = r.astype({k: object for k in ("entity_id", "name_core", "addr_clean", "addr_first_num")})
    s1 = r[r.src == 1].set_index("entity_id")
    o = r[r.src != 1].copy()
    o["X"] = o.entity_id.map(link)
    o["linked"] = o.X.notna()
    o["key"] = o.name_core + "|" + o.addr_first_num
    s1key = set(s1.name_core + "|" + s1.addr_first_num)
    wa = o[(o.addr_first_num != "")]
    ul = wa[~wa.linked]
    g = ul.groupby("key").size()
    orphan_keys = set(g[g >= 2].index) - s1key
    orph_by_name = pd.Series([k.split("|")[0] for k in orphan_keys]).value_counts()
    kvis = s1.groupby("name_core").size()
    # (A)
    A = o[o.linked & (o.addr_clean == "")].copy()
    A["k_vis"] = A.name_core.map(kvis).fillna(0)
    A["k_orph"] = A.name_core.map(orph_by_name).fillna(0)
    # (B)
    B = wa[wa.linked]
    ulk = set(ul.key)
    split_grp = B.key.isin(ulk) & ~B.key.isin(s1key)
    n_links = o.linked.sum()
    print(f"{split} {c:7s}: links {n_links} | (A) address-less links {len(A)} ({len(A)/n_links:.2%}); with >=1 orphan namesake group "
          f"{np.mean(A.k_orph >= 1):.1%}; k_vis==1 & k_orph>=1: {np.sum((A.k_vis == 1) & (A.k_orph >= 1))} | (B) linked-with-address whose "
          f"same-key twin is unlinked & key not in S1: {split_grp.sum()} ({split_grp.mean():.2%})", flush=True)
