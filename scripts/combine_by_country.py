"""matching_results = version A for all countries except those listed, which come from version B.
  python3 scripts/combine_by_country.py A.tsv B.tsv France out.tsv"""
import sys
import pandas as pd
a, b, countries, out = sys.argv[1], sys.argv[2], set(sys.argv[3].split(",")), sys.argv[4]
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
use_b = set(c.entity_id[c.country.isin(countries)])
A = pd.read_csv(a, sep="\t", dtype=str, keep_default_na=False)
B = pd.read_csv(b, sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id").matched_entity_ids
A["matched_entity_ids"] = [B[s] if s in use_b else m for s, m in zip(A.source1_entity_id, A.matched_entity_ids)]
A.to_csv(out, sep="\t", index=False)
print(f"wrote {out}: {len(A)} rows, {len(use_b)} rows taken from {b}")
