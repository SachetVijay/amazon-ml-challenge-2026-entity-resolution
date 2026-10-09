import pandas as pd
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
m = pd.read_csv("output_v4/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
m = m.merge(c, left_on="source1_entity_id", right_on="entity_id")
lst = m.matched_entity_ids.str.split(",")
m["n2"] = lst.apply(lambda l: sum(x.startswith("S2") for x in l if x))
m["n3"] = lst.apply(lambda l: sum(x.startswith("S3") for x in l if x))
g = m.groupby("country").agg(S1=("n2", "size"), S2_per_S1=("n2", "mean"), S3_per_S1=("n3", "mean"),
                             empty=("matched_entity_ids", lambda s: (s == "").mean()))
print(g.round(3))
tot = {}
for s in (2, 3):
    d = pd.read_csv(f"student_resource/dataset/test/test_source{s}.tsv", sep="\t", usecols=["country"], dtype=str)
    tot[f"S{s}_records_per_S1"] = d.country.value_counts() / c.country.value_counts()
print(pd.DataFrame(tot).round(3))
print("train reference: S2/S1 1.674, S3/S1 1.788 matched; records per S1 (train): S2 2.28, S3 2.39; empty 5.6%")
