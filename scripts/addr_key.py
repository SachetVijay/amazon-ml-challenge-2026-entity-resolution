"""Address-key blocking audit: for each S2/S3 record with an address, S1s sharing its exact address key
(house number w/o leading zeros + street/city words) -- are they among the record's candidates?
  python scripts/addr_key.py test   -> rates per country (no labels)
  python scripts/addr_key.py train  -> same on the validation sample + how often such missing S1s are the true owner"""
import re, sys
import numpy as np, pandas as pd
split = sys.argv[1]
art = f"artifacts/{split}"
STREET = {"rue", "avenue", "boulevard", "allee", "impasse", "chemin", "route", "place", "cours", "quai", "cite", "square",
          "passage", "lotissement", "street", "road", "drive", "lane", "court", "place", "way", "avenue", "boulevard", "st",
          "rd", "dr", "ln", "ct", "ave", "blvd", "no", "bis", "ter", "unit", "suite", "apt", "fl", "floor"}
def key(addr):
    if not addr: return None
    toks = addr.split()
    nums = [t for t in toks if t.isdigit()]
    if not nums: return None
    words = sorted({t for t in toks if not t.isdigit() and not t.startswith(("frreg_", "usst_", "inst_")) and t not in STREET and len(t) > 1})
    if len(words) < 2: return None
    return str(int(nums[0])) + "|" + " ".join(words)
rows_all = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
out = {}
for country in pd.unique(rows_all):
    idx = np.flatnonzero(rows_all == country)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src", "addr_clean"], filters=[("country", "==", country)])
    r.index = idx
    r["k"] = [key(a) for a in r.addr_clean.astype(object)]
    s1 = r[(r.src == 1) & r.k.notna()]
    kmap = s1.groupby("k").apply(lambda g: list(g.index)).to_dict()
    oth = r[(r.src != 1) & r.k.notna()]
    oth = oth[oth.k.isin(kmap)]
    if split == "test":
        cand = pd.read_parquet(f"{art}/pred_hyb_{country}.cand.parquet", columns=["a", "b"])
    else:   # validation sample: bucket-0 S1s only (their full candidate lists)
        cand = pd.read_parquet(f"{art}/oof_hyb5.parquet", columns=["a", "b", "country"])
        cand = cand[cand.country == country][["a", "b"]]
        s1ok = set(cand.a.unique())
        kmap = {k: [a for a in v if a in s1ok] for k, v in kmap.items()}
        oth = oth[[len(kmap[k]) > 0 for k in oth.k]]
    cs = set(zip(cand.a.to_numpy(), cand.b.to_numpy()))
    pairs = [(a, b) for b, k in zip(oth.index, oth.k) for a in kmap[k]]
    miss = [(a, b) for a, b in pairs if (a, b) not in cs]
    n_rec = (r.src != 1).sum()
    print(f"{country:7s}: records {n_rec}, with address key {r[(r.src!=1)].k.notna().mean():.1%}; records having an S1 address twin "
          f"{len(oth)} ({len(oth)/n_rec:.2%}); twin pairs {len(pairs)}, NOT in candidates {len(miss)} ({len(miss)/max(len(pairs),1):.2%})", flush=True)
    out[country] = miss
    if split == "train":
        import sys as _s; _s.path.insert(0, "src")
        from er.prep import load_gt
        gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
        ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
        ids = r.entity_id.astype(object)
        t_in = np.mean([(ids[a], ids[b]) in ys for a, b in pairs if (a, b) in cs]) if pairs else float("nan")
        t_miss = np.mean([(ids[a], ids[b]) in ys for a, b in miss]) if miss else float("nan")
        print(f"         true-owner rate: twins in candidates {t_in:.3f}, twins NOT in candidates {t_miss:.3f}", flush=True)
pd.to_pickle(out, f"/tmp/addr_twin_miss_{split}.pkl")
