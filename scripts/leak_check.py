import numpy as np, pandas as pd
D = "student_resource/dataset/train/"
gt = pd.read_csv(D + "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False).sample(200000, random_state=0)
pairs = gt.assign(o=gt.matched_entity_ids.str.split(",")).explode("o")
pairs = pairs[pairs.o.notna() & (pairs.o != "")]
pos = {}
for s in (1, 2, 3):
    ids = pd.read_csv(D + f"train_source{s}.tsv", sep="\t", usecols=[0], dtype=str).entity_id
    pos.update(dict(zip(ids, np.arange(len(ids)) / len(ids))))   # relative row position in its file
pairs["r1"] = pairs.source1_entity_id.map(pos)
pairs["ro"] = pairs.o.map(pos)
num = lambda s: s.str.split("-").str[1].astype(np.int64)
pairs["n1"], pairs["no"] = num(pairs.source1_entity_id), num(pairs.o)
rnd = pairs.sample(frac=1, random_state=1)
print("corr(row pos S1, row pos match)      :", round(np.corrcoef(pairs.r1, pairs.ro)[0, 1], 4))
print("corr(id number S1, id number match)  :", round(np.corrcoef(pairs.n1, pairs.no)[0, 1], 4))
print("median |row pos diff| true vs random :", round((pairs.r1 - pairs.ro).abs().median(), 4), "vs",
      round((pairs.r1.to_numpy() - rnd.ro.to_numpy()).__abs__().mean() * 0 + np.median(np.abs(pairs.r1.to_numpy() - rnd.ro.to_numpy())), 4))
# records of the same entity: are S2 siblings adjacent in the S2 file?
sib = pairs[pairs.o.str.startswith("S2")].groupby("source1_entity_id").ro.agg(["min", "max", "size"])
sib = sib[sib["size"] >= 2]
print("median row-span of an entity's S2 records:", round((sib["max"] - sib["min"]).median(), 4), "(random ≈ 0.33)")
