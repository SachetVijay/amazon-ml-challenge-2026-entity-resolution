"""Do shadow entities (unlinked >=2-record groups sharing name_core+number, key not in S1) emit address-less records?
For S1 names that are unique (k_vis=1): mean # address-less S2/S3 records with that exact core name,
names WITH a shadow group vs WITHOUT.  Train analog: unique names vs names held by exactly 2 S1s."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
split = sys.argv[1]; art = f"artifacts/{split}"
if split == "train":
    from er.prep import load_gt
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); linked = set(gt.other); countries = ("US", "India")
else:
    m = pd.read_csv("output_v14/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    linked = {o for l in m.matched_entity_ids for o in l.split(",") if o}; countries = ("US", "India", "France")
for c in countries:
    r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "name_core", "addr_clean", "addr_first_num"], filters=[("country", "==", c)])
    r = r.astype({k: object for k in r.columns if k != "src"})
    s1 = r[r.src == 1]; o = r[r.src != 1].copy(); o["linked"] = o.entity_id.isin(linked)
    kvis = s1.groupby("name_core").size()
    nal = o[o.addr_clean == ""].groupby("name_core").size()
    s1key = set(s1.name_core + "|" + s1.addr_first_num)
    ul = o[~o.linked & (o.addr_first_num != "")]; key = ul.name_core + "|" + ul.addr_first_num
    g = key.value_counts(); sh = pd.Series([k.split("|")[0] for k in set(g[g >= 2].index) - s1key]).value_counts()
    names1 = kvis[kvis == 1].index; names2 = kvis[kvis == 2].index
    with_sh = names1[names1.isin(sh.index)]; without = names1[~names1.isin(sh.index)]
    f = lambda ns: nal.reindex(ns).fillna(0).mean()
    print(f"{split} {c:7s}: address-less records per unique S1 name: WITHOUT shadow {f(without):.4f} ({len(without)} names) | "
          f"WITH shadow {f(with_sh):.4f} ({len(with_sh)} names) | ratio {f(with_sh)/max(f(without),1e-9):.2f} || names held by 2 S1s: {f(names2):.4f} (ratio {f(names2)/max(f(without),1e-9):.2f})", flush=True)
