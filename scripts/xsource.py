"""Cross-source evidence test (training ground truth, US).
Q1: are an entity's records correlated with EACH OTHER beyond both being noisy copies of S1?
    (e.g. P(raw name of S3 record == raw name of S2 record of same entity) vs == S1 raw name)
Q2: for address-less records whose exact core name is shared by k>=2 S1s, does the record's best raw-name match
    among each candidate S1's OTHER records single out the true owner (vs 1/k chance)?"""
import sys, time
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, "src")
from er.prep import load_gt
T = time.time()
cols = ["entity_id", "src", "country", "business_name", "business_address", "name_core", "addr_clean"]
rec = pd.read_parquet("artifacts/train/records.parquet", columns=cols, filters=[("country", "==", "US")])
rec = rec.astype({c: object for c in cols if c != "src"})
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
R = rec.set_index("entity_id")
gt = gt[gt.s1.isin(R.index)]
owner = dict(zip(gt.other, gt.s1))
recs_of = gt.groupby("s1").other.agg(list).to_dict()
print(f"US records {len(rec)}, true pairs {len(gt)}  ({time.time()-T:.0f}s)", flush=True)
# ---- Q1: pairwise raw-name equality within an entity
rng = np.random.default_rng(0)
s1s = rng.choice(np.array(list(recs_of)), 60000, replace=False)
eq_s1, eq_sib, eq_rand, n1, n2 = 0, 0, 0, 0, 0
rat_s1, rat_sib = [], []
allrec = rec[rec.src != 1].entity_id.to_numpy()
for s in s1s:
    rs = recs_of[s]
    s2 = [r for r in rs if r.startswith("S2")]; s3 = [r for r in rs if r.startswith("S3")]
    if not s2 or not s3: continue
    a, b = R.at[s2[0], "business_name"], R.at[s3[0], "business_name"]
    n1 += 1
    eq_sib += a == b
    eq_s1 += (b == R.at[s, "business_name"])
    rat_sib.append(fuzz.ratio(a, b)); rat_s1.append(fuzz.ratio(b, R.at[s, "business_name"]))
print(f"Q1 ({n1} entities with S2+S3): P(S3 raw name == S2 raw name of same entity) {eq_sib/n1:.3f} | "
      f"P(S3 raw name == S1 raw name) {eq_s1/n1:.3f} | mean ratio S3~S2 {np.mean(rat_sib):.1f} vs S3~S1 {np.mean(rat_s1):.1f}", flush=True)
# ---- Q2: address-less records with a shared exact core name
s1rec = rec[rec.src == 1]
grp = s1rec.groupby("name_core").entity_id.agg(list)
grp = grp[grp.str.len() >= 2]
noaddr = rec[(rec.src != 1) & (rec.addr_clean == "") & rec.name_core.isin(grp.index)]
noaddr = noaddr[noaddr.entity_id.map(owner).notna()]
noaddr = noaddr[[owner[e] in set(grp[c]) for e, c in zip(noaddr.entity_id, noaddr.name_core)]]
noaddr = noaddr.sample(min(20000, len(noaddr)), random_state=0)
print(f"Q2: {len(noaddr)} address-less true records whose owner shares its exact core name with >=1 other S1", flush=True)
hits = {"exact_raw_sib": [0, 0], "best_ratio_sib": [0, 0], "best_ratio_s1": [0, 0]}
chance, tie_any = [], 0
for e, nm, c in zip(noaddr.entity_id, noaddr.business_name, noaddr.name_core):
    group = grp[c]; own = owner[e]
    chance.append(1 / len(group))
    sc_exact, sc_ratio, sc_s1 = {}, {}, {}
    for z in group:
        sibs = [r for r in recs_of.get(z, []) if r != e]
        names = [R.at[r, "business_name"] for r in sibs]
        sc_exact[z] = max([nm == x for x in names], default=False)
        sc_ratio[z] = max([fuzz.ratio(nm, x) for x in names], default=-1)
        sc_s1[z] = fuzz.ratio(nm, R.at[z, "business_name"])
    for key, sc in (("exact_raw_sib", sc_exact), ("best_ratio_sib", sc_ratio), ("best_ratio_s1", sc_s1)):
        best = max(sc.values()); winners = [z for z, v in sc.items() if v == best]
        if own in winners:
            hits[key][0] += 1 / len(winners)       # expected accuracy with random tie-break
        hits[key][1] += 1
print(f"  random guess accuracy (1/k): {np.mean(chance):.3f}")
for k, (h, n) in hits.items():
    print(f"  pick owner by {k:15s}: accuracy {h/n:.3f}")
print(f"done {time.time()-T:.0f}s")
