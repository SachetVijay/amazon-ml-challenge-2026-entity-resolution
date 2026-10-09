"""Pair features for (S1 record a, S2/S3 record b). Deliberately no country feature (France is unseen).

  feats = pair_features(rec, cand)   # cand has columns a, b (+ retrieval score/rank columns)
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

from .normalize import LEGAL

FUZZ = {
    "ratio": fuzz.ratio, "pratio": fuzz.partial_ratio, "tsort": fuzz.token_sort_ratio,
    "tset": fuzz.token_set_ratio, "jw": distance.JaroWinkler.normalized_similarity,
}


def _pairwise(x: list, y: list, scorer) -> np.ndarray:
    return cpdist(x, y, scorer=scorer, workers=-1, dtype=np.float32)


def _set_feats(args):
    """Python-level set features on a chunk (run in a pool)."""
    an, bn, ask, bsk, aw, bw, aas, bas, anum, bnum, afn, bfn, al, bl = args
    out = np.zeros((len(an), 12), np.float32)
    for i in range(len(an)):
        A, B = set(an[i].split()), set(bn[i].split())
        out[i, 0] = len(A & B) / max(len(A | B), 1)
        out[i, 1] = len(A & B) / max(min(len(A), len(B)), 1)            # containment (extra/missing words)
        SA, SB = set(ask[i].split()), set(bsk[i].split())
        out[i, 2] = len(SA & SB) / max(len(SA | SB), 1)
        WA, WB = set(aw[i].split()), set(bw[i].split())
        out[i, 3] = len(WA & WB) / max(len(WA | WB), 1)
        out[i, 4] = len(WA & WB) / max(min(len(WA), len(WB)), 1)
        XA, XB = set(aas[i].split()), set(bas[i].split())
        out[i, 5] = len(XA & XB) / max(len(XA | XB), 1)
        NA, NB = set(anum[i].split()), set(bnum[i].split())
        out[i, 6] = len(NA & NB) / max(len(NA | NB), 1) if (NA or NB) else -1
        out[i, 7] = (afn[i] == bfn[i]) if (afn[i] and bfn[i]) else -1
        # truncated / partial house numbers: "030" vs "2030", "1915-1919" vs "1915"
        out[i, 8] = (bool(afn[i]) and bool(bfn[i]) and (afn[i].endswith(bfn[i]) or bfn[i].endswith(afn[i])
                     or afn[i].startswith(bfn[i]) or bfn[i].startswith(afn[i])))
        out[i, 9] = len(NA - NB) if (NA and NB) else -1                  # conflicting numbers
        LA, LB = set(al[i].split()), set(bl[i].split())
        out[i, 10] = -1 if not (LA and LB) else len(LA & LB) / len(LA | LB)
        out[i, 11] = (an[i].replace(" ", "") == bn[i].replace(" ", ""))
    return out


def _loc_strip(args):
    """Names minus location tokens (tokens present in either record's address): city/locality words in a
    business name describe *where*, not *who* (Nantes Centre SARL @ Nantes; Bangalore North Outsourcing)."""
    an, bn, aa, ba = args
    resA, resB = [], []
    out = np.zeros((len(an), 4), np.float32)
    for i in range(len(an)):
        loc = set(aa[i].split()) | set(ba[i].split())
        A = [t for t in an[i].split() if t not in loc and t not in LEGAL]
        B = [t for t in bn[i].split() if t not in loc and t not in LEGAL]
        SA, SB = set(A), set(B)
        out[i, 0] = len(an[i].split()) - len(A)                            # location tokens removed from A
        out[i, 1] = len(SA & SB) / max(len(SA | SB), 1) if (SA or SB) else -1
        out[i, 2] = len(SA & SB) / max(min(len(SA), len(SB)), 1) if (SA and SB) else -1
        out[i, 3] = float(bool(SA) and SA == SB)
        resA.append(" ".join(A))
        resB.append(" ".join(B))
    return out, resA, resB


LOC_COLS = ["loc_removedA", "loc_name_jac", "loc_name_cont", "loc_name_eq"]


def location_feats(a_name, b_name, a_addr, b_addr, workers, chunk) -> dict:
    arrays = [a_name, b_name, a_addr, b_addr]
    jobs = [tuple(arr[i:i + chunk] for arr in arrays) for i in range(0, len(a_name), chunk)]
    with Pool(workers) as p:
        res = p.map(_loc_strip, jobs)
    M = np.vstack([r[0] for r in res])
    ra = [x for r in res for x in r[1]]
    rb = [x for r in res for x in r[2]]
    F = {c: M[:, j] for j, c in enumerate(LOC_COLS)}
    F["loc_name_ratio"] = _pairwise(ra, rb, fuzz.ratio)
    F["loc_name_tsort"] = _pairwise(ra, rb, fuzz.token_sort_ratio)
    return F


SET_COLS = ["name_jac", "name_cont", "skel_jac", "addr_jac", "addr_cont", "addrskel_jac", "num_jac",
            "firstnum_eq", "num_partial", "num_conflict", "legal_jac", "name_nospace_eq"]


NEEDED = ["name_clean", "name_core", "name_sorted", "name_skel", "legal", "addr_clean", "addr_words",
          "addr_skel", "addr_nums", "addr_first_num", "addr_empty", "src", "country", "business_name"]


def pair_features(rec: pd.DataFrame, cand: pd.DataFrame, workers: int = 16, chunk: int = 200_000,
                  block: int = 500_000) -> pd.DataFrame:
    """Chunked over pairs so peak memory stays ~O(block) regardless of candidate-set size."""
    s1 = rec[rec.src == 1]
    freq = s1.groupby(["country", "name_core"]).size()
    skel_freq = s1.groupby(["country", "name_skel"]).size()
    parts = []
    for i in range(0, len(cand), block):
        parts.append(_pair_block(rec, cand.iloc[i:i + block], freq, workers, chunk, skel_freq))
    return pd.concat(parts, ignore_index=True)


def _initials(name: str) -> str:
    return "".join(t[0] for t in name.split() if t)


def acronym_feats(a_core: list, b_core: list) -> np.ndarray:
    """1 if one name is the other's initials (Twin Union -> TU, Indraya Education -> IE), else 0.
    Only meaningful for multi-word names; -1 when not applicable."""
    out = np.full(len(a_core), -1, np.float32)
    for i, (x, y) in enumerate(zip(a_core, b_core)):
        xs, ys = x.replace(" ", ""), y.replace(" ", "")
        ix, iy = _initials(x), _initials(y)
        if len(ix) >= 2 and 2 <= len(ys) <= 6:
            out[i] = float(ys == ix)
        elif len(iy) >= 2 and 2 <= len(xs) <= 6:
            out[i] = float(xs == iy)
    return out


def name_ambiguity(country, name_core, name_skel, freq, skel_freq) -> dict:
    """How many S1 entities (same country) share this record's core name / name skeleton.
    1 => an address-less record can only belong to one entity; large => chain, needs the address."""
    idx = pd.MultiIndex.from_arrays([country, name_core])
    sidx = pd.MultiIndex.from_arrays([country, name_skel])
    return {"nameB_s1freq": np.log1p(freq.reindex(idx).fillna(0).to_numpy(np.float32)),
            "skelB_s1freq": np.log1p(skel_freq.reindex(sidx).fillna(0).to_numpy(np.float32))}


def _pair_block(rec, cand, freq, workers, chunk, skel_freq):
    A = rec.loc[cand.a.values, NEEDED].reset_index(drop=True)
    B = rec.loc[cand.b.values, NEEDED].reset_index(drop=True)
    F = {}
    for col, short in [("name_clean", "nc"), ("name_core", "nk"), ("name_sorted", "ns")]:
        for fn_name, fn in FUZZ.items():
            if short != "nc" and fn_name in ("tsort", "tset"):
                continue
            F[f"{short}_{fn_name}"] = _pairwise(A[col].tolist(), B[col].tolist(), fn)
    nsA = A.name_core.str.replace(" ", "", regex=False).tolist()
    nsB = B.name_core.str.replace(" ", "", regex=False).tolist()
    F["nsp_ratio"] = _pairwise(nsA, nsB, fuzz.ratio)
    F["nsp_pratio"] = _pairwise(nsA, nsB, fuzz.partial_ratio)
    F["skel_ratio"] = _pairwise(A.name_skel.tolist(), B.name_skel.tolist(), fuzz.ratio)
    for fn_name in ("ratio", "tset", "pratio"):
        F[f"addr_{fn_name}"] = _pairwise(A.addr_clean.tolist(), B.addr_clean.tolist(), FUZZ[fn_name])
    F["addrw_tsort"] = _pairwise(A.addr_words.tolist(), B.addr_words.tolist(), fuzz.token_sort_ratio)
    # cross-field: business name appears inside the other's address or vice versa (rare but cheap)
    F["nameB_in_addrA"] = _pairwise(B.name_core.tolist(), A.addr_clean.tolist(), fuzz.partial_ratio)
    F["acronym"] = acronym_feats(A.name_clean.tolist(), B.name_clean.tolist())
    F.update(location_feats(A.name_clean.tolist(), B.name_clean.tolist(), A.addr_words.tolist(),
                            B.addr_words.tolist(), workers, chunk))

    cols = ["name_core", "name_core", "name_skel", "name_skel", "addr_words", "addr_words", "addr_skel",
            "addr_skel", "addr_nums", "addr_nums", "addr_first_num", "addr_first_num", "legal", "legal"]
    srcs = [A, B] * 7
    arrays = [s[c].tolist() for s, c in zip(srcs, cols)]
    jobs = [tuple(arr[i:i + chunk] for arr in arrays) for i in range(0, len(cand), chunk)]
    with Pool(workers) as p:
        S = np.vstack(p.map(_set_feats, jobs))
    for j, c in enumerate(SET_COLS):
        F[c] = S[:, j]

    F["lenA"] = A.name_core.str.len().to_numpy(np.float32)
    F["lenB"] = B.name_core.str.len().to_numpy(np.float32)
    F["len_diff"] = np.abs(F["lenA"] - F["lenB"])
    F["addrA_empty"] = A.addr_empty.to_numpy(np.float32)
    F["addrB_empty"] = B.addr_empty.to_numpy(np.float32)
    F["b_is_s3"] = (B.src.to_numpy() == 3).astype(np.float32)
    F["b_nonlatin"] = B.business_name.str.contains(r"[^\x00-\x7FÀ-ɏ]", regex=True).to_numpy(np.float32)
    # name frequency among S1 (chains / common names need the address to disambiguate)
    F["nameA_freq"] = np.log1p(freq.reindex(pd.MultiIndex.from_arrays([A.country, A.name_core])).fillna(0).to_numpy(np.float32))
    F.update(name_ambiguity(B.country, B.name_core, B.name_skel, freq, skel_freq))
    feats = pd.DataFrame(F)
    keep = [c for c in cand.columns if c.startswith(("score_", "rank_"))]
    return pd.concat([cand[["a", "b"]].reset_index(drop=True), cand[keep].reset_index(drop=True).astype(np.float32), feats], axis=1)
