"""Post-decision rule for a country absent from training: demote same-address one-word CATEGORY swaps.

A record whose name equals an S1 name with one frequent, native name word replaced by another (e.g. `X Club` -> `X Ecole`,
same street and house number) is the generator's planted-distractor recipe. On the unseen country the leaderboard
showed that accepting more of these (v16 -> v17c) cost -0.0014 overall. Abbreviations (subsequence with the same
initial), near-identical variants and swaps INTO filler-like words (`Cie`, `(France)`, `Groupe`: the generator's
noise) are kept. Demoted pairs get p = 0.01 and the entity decisions are recomputed.

  python -m er.demote_swaps --art artifacts/test --country France --pred-s2 artifacts/test \
      --base-tsv output/matching_results.tsv --out output/matching_results_demoted.tsv
"""
import argparse

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .decide import exclusive, expected_f_fast
from .normalize import ADDR_ABBR, LEGAL, to_ascii
import re

STOP = {"no", "number", "n", "deg", "bis", "ter", "de", "du", "des", "la", "le", "les", "d", "l", "unit", "apt", "suite"}
def street(addr):
    for comp in (addr or "").split(","):
        s = to_ascii(comp).lower(); m = re.search(r"\d+", s)
        if m: return m.group().lstrip("0"), " ".join(w for w in (ADDR_ABBR.get(t, t) for t in re.findall(r"[a-z]+", s)) if w not in STOP)
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--pred-s2", required=True)
    ap.add_argument("--base-tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inplace-words", nargs="*", default=[],
                    help="filler-like words whose swaps are still demoted when the word takes the replaced word's place "
                         "(same position, or both last) instead of being appended after the legal form")
    ap.add_argument("--dfmin", type=float, default=1e-3, help="minimum S1-name frequency of a swapped word")
    ap.add_argument("--broad", action="store_true",
                    help="also: category swap alongside other name noise, address-less records, same street/other number")
    ap.add_argument("--keep-words", nargs="+", default=["cie", "france", "compagnie", "societe", "fils", "freres", "centre", "services", "groupe"],
                    help="filler-like words: swaps INTO these are treated as the generator's noise and kept")
    args = ap.parse_args()
    c = args.country
    FILL_LIKE = set(args.keep_words)
    def toks(s): return {t for t in s.split() if t not in LEGAL and len(t) > 1}
    def abbrev(a, b):
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        if s[0] != l[0]: return False
        i = 0
        for ch in l:
            if i < len(s) and ch == s[i]: i += 1
        return i == len(s)
    rows = np.flatnonzero(pd.read_parquet(f"{args.art}/records.parquet", columns=["country"]).country.to_numpy() == c)
    r = pd.read_parquet(f"{args.art}/records.parquet", columns=["entity_id", "src", "name_clean", "business_name", "business_address", "addr_clean"], filters=[("country", "==", c)]); r.index = rows
    sdf = pd.Series([t for n in r.name_clean[r.src == 1].astype(object) for t in toks(n)]).value_counts() / (r.src == 1).sum()
    odf = pd.Series([t for n in r.name_clean[r.src != 1].astype(object) for t in toks(n)]).value_counts() / (r.src != 1).sum()
    ratio = (odf + 1e-4) / (sdf.reindex(odf.index).fillna(0) + 1e-4)
    P = pd.read_parquet(f"{args.pred_s2}/pred_s2_{c}.parquet")
    C = P[P.p > 0.05].copy()
    A, B = r.name_clean.astype(object).loc[C.a].tolist(), r.name_clean.astype(object).loc[C.b].tolist()
    INPLACE = set(args.inplace_words)
    def words(raw):
        return re.findall(r"[a-z0-9]+", to_ascii(raw).lower())
    def inplace(ra, rb, u, v):
        wa, wb = words(ra), words(rb)
        if u not in wa or v not in wb: return False
        i, j = wa.index(u), wb.index(v)
        if any(w in LEGAL for w in wb[:j]) or (j == 0 and i != 0):
            return False                       # appended after the legal form / moved to the front: generator noise
        return i == j or (i == len(wa) - 1 and j == len(wb) - 1)
    RAW = list(zip(r.business_name.astype(object).loc[C.a].tolist(), r.business_name.astype(object).loc[C.b].tolist()))
    cat = np.zeros(len(C), bool)
    for i, (x, y) in enumerate(zip(A, B)):
        ta, tb = toks(x), toks(y); dx, dy = ta - tb, tb - ta
        CONNECT = {"and", "et", "associes", "service", "developpement"}
        if args.broad and ta & tb and not (len(dx) == 1 and len(dy) == 1):
            hit = [(u, v) for u in dx for v in dy if sdf.get(u, 0) >= args.dfmin and sdf.get(v, 0) >= args.dfmin
                   and ratio.get(v, 1) < 2 and v not in FILL_LIKE and v not in CONNECT and not abbrev(u, v) and fuzz.ratio(u, v) < 70]
            if hit: cat[i] = True
        if len(dx) == 1 and len(dy) == 1 and ta & tb:
            u, v = next(iter(dx)), next(iter(dy))
            if sdf.get(u, 0) >= args.dfmin and sdf.get(v, 0) >= args.dfmin and ratio.get(v, 1) < 2 and not abbrev(u, v) and fuzz.ratio(u, v) < 70:
                if v not in FILL_LIKE:             # record swaps in a CATEGORY word (the planted-distractor recipe)
                    cat[i] = True
                elif v in INPLACE and inplace(RAW[i][0], RAW[i][1], u, v):
                    cat[i] = True
    C = C[cat]
    sa = [street(x) for x in r.business_address.astype(object).loc[C.a]]; sb = [street(x) for x in r.business_address.astype(object).loc[C.b]]
    st = cpdist([x[1] for x in sa], [x[1] for x in sb], scorer=fuzz.ratio, workers=-1)
    same = np.array([(x[0] == y[0] and x[0] != "") for x, y in zip(sa, sb)]) & (st >= 85)
    if args.broad:
        same = np.ones(len(C), bool)     # category change is the distractor signature wherever the record points
    C = C[same]
    ex = exclusive(P); sel = expected_f_fast(ex)
    selk = set(zip(sel.a, sel.b))
    insel = np.array([(a, b) in selk for a, b in zip(C.a, C.b)])
    print(f"category-swap same-address pairs (p>0.05): {len(C)}; currently LINKED: {insel.sum()} ({insel.sum()/len(sel):.2%} of French links); p quantiles of linked {C.p[insel].quantile([.1,.5,.9]).round(3).tolist()}")
    pairs = pd.Series([f"{sorted(toks(r.name_clean.loc[a]) - toks(r.name_clean.loc[b]))[0]}->{sorted(toks(r.name_clean.loc[b]) - toks(r.name_clean.loc[a]))[0]}" for a, b in zip(C.a[insel], C.b[insel])]).value_counts()
    print("top swaps among linked:", pairs.head(15).to_dict())
    if True:
        key = set(zip(C.a, C.b))
        m = np.array([(a, b) in key for a, b in zip(P.a, P.b)])
        P2 = P.assign(p=np.where(m, np.minimum(P.p, 0.01), P.p))
        ex2 = exclusive(P2); sel2 = expected_f_fast(ex2)
        ids = r.entity_id.astype(object)
        fr = {}
        for a, b in zip(sel2.a, sel2.b): fr.setdefault(ids[a], []).append(ids[b])
        s1_fr = set(ids[r.index[r.src == 1]])
        base = pd.read_csv(args.base_tsv, sep="\t", dtype=str, keep_default_na=False)
        new = [",".join(fr.get(s, [])) if s in s1_fr else l for s, l in zip(base.source1_entity_id, base.matched_entity_ids)]
        base.assign(matched_entity_ids=new).to_csv(args.out, sep="\t", index=False)
        print(f"wrote {args.out}: France links {len(sel)} -> {len(sel2)}")


if __name__ == "__main__":
    main()
