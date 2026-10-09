"""Missing address twins (S1 at the record's exact address key, not a candidate): does the twin also carry the
record's NAME (crowding-out of the true owner) or a different name (co-located business)?
Compared across countries on test; on train, true-owner rate by name similarity bucket."""
import sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
split = sys.argv[1]; art = f"artifacts/{split}"
miss = pd.read_pickle(f"/tmp/addr_twin_miss_{split}.pkl")
rows_all = pd.read_parquet(f"{art}/records.parquet", columns=["country"]).country.to_numpy()
if split == "train":
    sys.path.insert(0, "src"); from er.prep import load_gt
    gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv"); ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
for country, pairs in miss.items():
    if not pairs: continue
    idx = np.flatnonzero(rows_all == country)
    r = pd.read_parquet(f"{art}/records.parquet", columns=["entity_id", "name_core", "name_clean"], filters=[("country", "==", country)])
    r.index = idx
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    sim = cpdist(r.name_core.loc[a].astype(object).tolist(), r.name_core.loc[b].astype(object).tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    msg = f"{country:7s} missing twins {len(pairs):7d}: name sim >=90 {np.mean(sim >= 90):.1%} | 70-90 {np.mean((sim >= 70) & (sim < 90)):.1%} | <70 {np.mean(sim < 70):.1%}"
    if split == "train":
        ids = r.entity_id.astype(object)
        yt = np.array([(ids[x], ids[y]) in ys for x, y in zip(a, b)])
        msg += f" | true-owner rate: sim>=90 {yt[sim >= 90].mean() if (sim >= 90).any() else float('nan'):.3f}, <70 {yt[sim < 70].mean():.3f}"
    print(msg, flush=True)
