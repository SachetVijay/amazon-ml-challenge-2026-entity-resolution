import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast, f05
from er.prep import load_gt
from er.predict import country_rows, read_country
O = pd.read_parquet("artifacts/train/oof_stage2_hyb5_ce_v4.parquet", columns=["a", "b", "s1_id", "o_id", "country", "p2"])
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
samp = set(O.s1_id)
truth = gt[gt.s1.isin(samp)].groupby("s1").other.agg(set).to_dict()
ex = exclusive(O.rename(columns={"p2": "p"}))
sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
pred = {s: set(g) for s, g in sel.groupby("s1_id")["o_id"]}
rows = []
path = "artifacts/train/records.parquet"
for c, r in country_rows(path).items():
    sub = read_country(path, c, r)[["entity_id", "src", "name_core", "addr_nums", "addr_empty"]]
    s1 = sub[sub.src.to_numpy() == 1]
    freq = s1.groupby("name_core").size()
    s1 = s1[s1.entity_id.isin(samp)]
    for eid, nc, nums in zip(s1.entity_id, s1.name_core, s1.addr_nums):
        rows.append((c, eid, freq.get(nc, 1), len(str(nc).split()), f05(pred.get(eid, set()), truth.get(eid, set()))))
R = pd.DataFrame(rows, columns=["country", "s1", "name_freq", "n_tok", "f"])
R["freq_bin"] = pd.cut(R.name_freq, [0, 1, 2, 5, 20, 1e9], labels=["1 (unique)", "2", "3-5", "6-20", ">20"])
print("macro F0.5 by S1 name frequency (how many S1 in the country share the core name):")
print(R.groupby(["freq_bin"], observed=True).f.agg(["mean", "size"]).round(4))
print(R.groupby(["country", "freq_bin"], observed=True).f.agg(["mean", "size"]).round(4))
print("\nby number of core-name tokens:")
print(R.groupby(pd.cut(R.n_tok, [0, 1, 2, 3, 99]), observed=True).f.agg(["mean", "size"]).round(4))
