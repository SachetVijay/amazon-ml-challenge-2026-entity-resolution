"""Orphan test: do unlinked records come in groups (copies of an entity missing from S1)?
Key = (name_core, first house number) for records with an address; also name_core only for all records.
train: GT-unmatched records vs matched records.  test: records not linked by v14."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.prep import load_gt
split = sys.argv[1]; art = f"artifacts/{split}"
rows_all = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
if split == "train":
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); matched = set(gt.other)
else:
    m = pd.read_csv("output_v14/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    matched = {o for l in m.matched_entity_ids for o in l.split(",") if o}
for c in (("US", "India") if split == "train" else ("US", "India", "France")):
    r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "name_core", "addr_first_num"], filters=[("country", "==", c)])
    r = r[r.src != 1].astype({"entity_id": object, "name_core": object, "addr_first_num": object})
    r["linked"] = r.entity_id.isin(matched)
    k = r[r.addr_first_num != ""]
    key = k.name_core + "|" + k.addr_first_num
    grp = key.map(key.value_counts())
    # twin = another record with the same key; unlinked twin = another UNLINKED record with same key
    ul = k[~k.linked]; kul = key[~k.linked]
    ul_cnt = kul.map(kul.value_counts())
    s1 = pd.read_parquet(f"{art}/records.parquet", columns=["src", "name_core", "addr_first_num"], filters=[("country", "==", c)])
    s1 = s1[s1.src == 1]; s1key = set(s1.name_core.astype(object) + "|" + s1.addr_first_num.astype(object))
    has_s1 = kul.isin(s1key)
    n1 = (pd.read_parquet(f"{art}/records.parquet", columns=["src"], filters=[("country", "==", c)]).src == 1).sum()
    print(f"{split} {c:7s}: records/S1 {len(r)/n1:.2f}, unlinked {np.mean(~r.linked):.1%} | unlinked w/ address: {len(ul)} ; "
          f"has an unlinked same-key twin {np.mean(ul_cnt >= 2):.1%} ; same key exists in S1 {has_s1.mean():.1%} ; "
          f"twin AND no S1 with key {np.mean((ul_cnt >= 2) & ~has_s1):.1%} | linked records with a same-key twin {np.mean(grp[k.linked] >= 2):.1%}", flush=True)
