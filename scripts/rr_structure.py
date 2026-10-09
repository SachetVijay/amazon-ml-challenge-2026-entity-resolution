"""Do unmatched (distractor) records form their own clusters? Nearest-neighbour structure among S2/S3 records."""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
from er.biencoder import load_emb, gpu_topk
from er.predict import country_rows
from er.prep import load_gt
art = "artifacts/train"
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
ent = dict(zip(gt.other, gt.s1))                         # record -> its S1 entity (matched records only)
for country in ("US", "India"):
    rows = country_rows(f"{art}/records.parquet")[country]
    rec = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "src"], filters=[("country", "==", country)])
    E = load_emb(art, country)
    ids = rec.entity_id.to_numpy(dtype=object)
    other = np.flatnonzero(rec.src.to_numpy() != 1)       # positions of S2/S3 records
    matched = np.array([i in ent for i in ids[other]])
    rng = np.random.default_rng(0)
    qm = rng.choice(other[matched], 8000, replace=False)
    qu = rng.choice(other[~matched], 8000, replace=False)
    for name, q in (("matched", qm), ("UNMATCHED", qu)):
        s, i = gpu_topk(E, q, other, 4)
        nn = other[i[:, 1:]]                            # drop self (rank 0)
        s = s[:, 1:]
        qe = np.array([ent.get(x) for x in ids[q]], dtype=object)
        ne = np.vectorize(lambda x: ent.get(x), otypes=[object])(ids[nn])
        same = (ne[:, 0] == qe) & (qe != None)
        nn_unm = np.array([x is None for x in ne[:, 0]])
        hi = s[:, 0] > 0.95
        print(f"{country} {name:9s}: NN1 cos median {np.median(s[:, 0]):.3f} | NN1 same entity {same.mean():.1%} | "
              f"NN1 unmatched {nn_unm.mean():.1%} | NN1 cos>0.95 {hi.mean():.1%}, of which unmatched {nn_unm[hi].mean():.1%}")
