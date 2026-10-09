"""(1) Validation: links added/removed by the test-like-context model vs the current model under test-like
context, and their precision. (2) Test France: overlap of v12's added links with the liberal probe's (which hurt)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.decide import exclusive, expected_f_fast
from er.prep import load_gt
gt = load_gt("student_resource/dataset/train/train_ground_truth.tsv")
ys = set(map(tuple, gt[["s1", "other"]].to_numpy()))
def links(path):
    P = pd.read_parquet(path)
    ex = exclusive(P[["a", "b", "s1_id", "o_id", "p"]])
    sel = expected_f_fast(ex).merge(ex[["a", "b", "s1_id", "o_id"]], on=["a", "b"])
    return set(zip(sel.s1_id, sel.o_id))
old, new = links(sys.argv[1]), links(sys.argv[2])
add, rem = new - old, old - new
prec = lambda s: np.mean([x in ys for x in s]) if s else float("nan")
print(f"validation (old->old vs new->new): kept {len(old & new)}, added {len(add)} (precision {prec(add):.3f}), "
      f"removed {len(rem)} (precision {prec(rem):.3f}); base precision {prec(old):.4f}")
def load(p):
    m = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    return {(s, o) for s, l in zip(m.source1_entity_id, m.matched_entity_ids) for o in l.split(",") if o}
c = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
fr = set(c.entity_id[c.country == "France"])
F = lambda s: {x for x in s if x[0] in fr}
best, v12 = F(load(sys.argv[3])), F(load(sys.argv[4]))
v11, lib = F(load("output_v11/matching_results.tsv")), F(load("output_v11fL/matching_results.tsv"))
a12, alib = v12 - best, lib - v11
print(f"test France: v12 adds {len(a12)}, liberal probe added {len(alib)}, overlap {len(a12 & alib)} "
      f"({len(a12 & alib) / len(a12):.1%} of v12's additions); v12 removes {len(best - v12)}")
