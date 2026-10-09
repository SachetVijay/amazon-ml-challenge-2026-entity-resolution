"""Link structure mix per 1k S1: v22 test links per country vs train OOF (p>0.5) with true rate per cell."""
import sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from er.normalize import LEGAL
from er.prep import load_gt
NOISE = {"center", "services", "service", "partners", "fils", "associes", "cie", "developpement", "groupe", "france"}
def core(n): return {t for t in n.split() if t not in LEGAL and len(t) > 1}
def ncat(na, nb):
    ta, tb = core(na), core(nb); D, A = ta - tb, tb - ta
    if not ta & tb: return "disjoint"
    if not D and not A: return "exact"
    if not D and len(A) == 1: return "app1-noise" if A <= NOISE else "app1-other"
    if len(D) == 1 and not A: return "drop1"
    if len(D) == 1 and len(A) == 1: return "swap1-noise" if A <= NOISE else "swap1-other"
    return "multi"
def acat(x, y):
    if not x or not y: return "noaddr"
    if x == y: return "same"
    sx, sy = x.split(), y.split()
    nx, ny = [t for t in sx if any(c.isdigit() for c in t)], [t for t in sy if any(c.isdigit() for c in t)]
    tx, ty = set(sx) - set(nx), set(sy) - set(ny)
    if tx == ty: return "samestreet-num" if nx != ny else "same~"
    if len(tx & ty) >= max(1, min(len(tx), len(ty)) - 1): return "near"
    return "diff"
