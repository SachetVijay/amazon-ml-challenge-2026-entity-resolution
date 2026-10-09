"""Mechanism check: do French pairs whose locality is written differently (region vs department) get lower
neural scores, even when the canonical addresses agree?"""
import re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import FR_PHRASES, FR_REGIONS, to_ascii
from er.predict import country_rows
art = "artifacts/test"
rows = country_rows(f"{art}/records.parquet")["France"]
r = pd.read_parquet(f"{art}/records.parquet", columns=["business_address", "addr_clean", "addr_nums", "name_core"],
                    filters=[("country", "==", "France")]); r.index = rows
regions = set(FR_REGIONS)
def form(addr):   # which locality form does the raw address use?
    s = " " + re.sub(r"[^a-z0-9]+", " ", to_ascii(addr)) + " "
    has_reg = any(f" {x} " in s for x in regions)
    has_dep = any(f" {p} " in s for p, reg in FR_PHRASES if p not in regions and len(p) > 4)
    return "reg" if has_reg and not has_dep else ("dep" if has_dep and not has_reg else ("both" if has_reg else "none"))
P = pd.read_parquet(f"{art}/pred_s2_France.parquet")
best = P.loc[P.groupby("b").p.idxmax()]
ce = pd.read_parquet(f"{art}/ce0_France.parquet")                          # e5-base CE logits (band pairs)
best = best.merge(ce, on=["a", "b"], how="left")
A, B = r.loc[best.a], r.loc[best.b]
best["fa"] = [form(x) for x in A.business_address]
best["fb"] = [form(x) for x in B.business_address]
# "canonically consistent" = same canonical region tokens + same house numbers + similar core name (likely true)
reg_tok = lambda s: " ".join(sorted(t for t in s.split() if t.startswith("frreg_")))
best["same_canon_region"] = [reg_tok(x) == reg_tok(y) and reg_tok(x) != "" for x, y in zip(A.addr_clean, B.addr_clean)]
best["same_nums"] = (A.addr_nums.to_numpy() == B.addr_nums.to_numpy()) & (A.addr_nums.to_numpy() != "")
best["same_name"] = A.name_core.to_numpy() == B.name_core.to_numpy()
easy = best.same_canon_region & best.same_nums & best.same_name        # canonical evidence says: clearly the same
for nm, m in (("SAME raw form (reg-reg / dep-dep)", easy & (best.fa == best.fb)),
              ("DIFFERENT raw form (reg <-> dep)", easy & (((best.fa == "reg") & (best.fb == "dep")) | ((best.fa == "dep") & (best.fb == "reg"))))):
    g = best[m]
    print(f"{nm:34s} n={len(g):7d} | p median {g.p.median():.4f} | p<0.95 {(g.p < .95).mean():.1%} | "
          f"CE logit median {g.ce.median():.2f} (scored {g.ce.notna().mean():.0%})")
