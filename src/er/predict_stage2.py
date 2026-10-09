"""Test-time stage 2: stage-1 pair probabilities + cross-encoder on the uncertain band + cluster context
-> stage-2 LightGBM -> exclusive assignment -> expected-F0.5 -> both TSVs.

Requires er.predict to have written artifacts/test/pred_hyb_<country>.parquet (+ .cand.parquet).

  python -m er.predict_stage2 --art artifacts/test --ce-model models/ce \
      --stage2 artifacts/train/stage2_hyb5_ce.txt --out output_v4
"""
import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decide import exclusive, expected_f_fast
from .predict import append_lists, country_rows, read_country
from .stage2 import context_features


def band_frame(ctx, ces, a, country):
    """Band pairs: cluster context (+p1) + cached stage-1 test features + cross-encoder logits."""
    inb = ctx[(ctx.p1 > a.band[0]) & (ctx.p1 < a.band[1])].drop(columns=[c for c in ("ce",) if c in ctx])
    keys = inb[["a", "b"]]
    feats = []
    import pyarrow.parquet as pq
    for fp in sorted((Path(a.art) / "feats_cache").glob(f"{country}_*.parquet")):
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=1_000_000):
            feats.append(batch.to_pandas().merge(keys, on=["a", "b"]))
    F = pd.concat(feats, ignore_index=True)
    D = inb.merge(F.drop(columns=[c for c in F.columns if c in inb.columns and c not in ("a", "b")]), on=["a", "b"])
    for c in ces:
        D = D.merge(c, on=["a", "b"], how="left")
    return inb, D


def apply_band2(P2, ctx, ces, q_booster, band2_booster, a, country, src_of=None):
    """Second pass (er.exp_band_rival): rival scorer q on every band pair, record-side context recomputed from q
    (p1 outside the band), optional per-source slot features, then the second-pass specialist.
    src_of: callable global row -> source (needed when the model uses slot features)."""
    from .exp_band_rival import QCTX, SLOT, q_context, slot_features
    inb, D = band_frame(ctx, ces, a, country)
    qn = q_booster.feature_name()
    D["q"] = q_booster.predict(D[qn].to_numpy(np.float32), num_threads=22)
    A = ctx[["a", "b", "p1"]].merge(D[["a", "b", "q"]], on=["a", "b"], how="left")
    A["v"] = A.q.fillna(A.p1)
    Q = q_context(A[A.b.isin(set(D.b))])
    D = D.drop(columns=["q"]).merge(Q[["a", "b"] + QCTX], on=["a", "b"], how="left")
    names = band2_booster.feature_name()
    if any(n in SLOT for n in names):
        A = A[A.a.isin(set(D.a))]
        D = D.merge(slot_features(A.assign(src=src_of(A.b.to_numpy()))), on=["a", "b"], how="left")
    missing = [n for n in names if n not in D.columns]
    assert not missing, f"second-pass features missing at test time: {missing}"
    pb = band2_booster.predict(D[names].to_numpy(np.float32), num_threads=22)
    upd = pd.Series(pb, index=pd.MultiIndex.from_arrays([D.a.to_numpy(), D.b.to_numpy()]))
    newp = upd.reindex(pd.MultiIndex.from_arrays([P2.a.to_numpy(), P2.b.to_numpy()])).to_numpy()
    print(f"  {country}: second-pass specialist rescored {len(D)} of {len(inb)} band pairs "
          f"(rival scorer: {len(qn)} features)", flush=True)
    return P2.assign(p=np.where(np.isnan(newp), P2.p.to_numpy(), newp))


def apply_band(P2, ctx, ces, band_booster, a, country):
    """Replace stage-2 probabilities inside the stage-1 band with the band specialist's.
    Specialist inputs = cluster context (+p1) + cached stage-1 test features + cross-encoder logits."""
    names = band_booster.feature_name()
    inb = ctx[(ctx.p1 > a.band[0]) & (ctx.p1 < a.band[1])].drop(columns=[c for c in ("ce",) if c in ctx])
    keys = inb[["a", "b"]]
    feats = []
    import pyarrow.parquet as pq
    for fp in sorted((Path(a.art) / "feats_cache").glob(f"{country}_*.parquet")):
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=1_000_000):
            feats.append(batch.to_pandas().merge(keys, on=["a", "b"]))
    F = pd.concat(feats, ignore_index=True)
    D = inb.merge(F.drop(columns=[c for c in F.columns if c in inb.columns and c not in ("a", "b")]), on=["a", "b"])
    for c in ces:
        D = D.merge(c, on=["a", "b"], how="left")
    missing = [n for n in names if n not in D.columns]
    assert not missing, f"band model features missing at test time: {missing}"
    pb = band_booster.predict(D[names].to_numpy(np.float32), num_threads=22)
    upd = pd.Series(pb, index=pd.MultiIndex.from_arrays([D.a.to_numpy(), D.b.to_numpy()]))
    idx = pd.MultiIndex.from_arrays([P2.a.to_numpy(), P2.b.to_numpy()])
    newp = upd.reindex(idx).to_numpy()
    print(f"  {country}: band specialist rescored {len(D)} pairs "
          f"(of {len(inb)} in band; {len(inb) - len(D)} without cached features keep stage 2)", flush=True)
    return P2.assign(p=np.where(np.isnan(newp), P2.p.to_numpy(), newp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--ce-model", nargs="+", required=True, help="cross-encoder dirs; first also feeds stage 2")
    ap.add_argument("--band-model", default=None, help="band specialist LightGBM (from er.exp_band)")
    ap.add_argument("--band-model-unseen", default=None, help="band specialist for countries absent from training")
    ap.add_argument("--unseen-n-ce", type=int, default=4, help="that model uses the first N cross-encoders")
    ap.add_argument("--unseen-ce-override", nargs="*", default=[],
                    help="J=MODEL_DIR: for countries absent from training, score cross-encoder index J with MODEL_DIR "
                         "(e.g. a copy adapted to the unseen country with pseudo-labels)")
    ap.add_argument("--unseen-use-seen", action="store_true",
                    help="unseen countries use the seen-country (all cross-encoder) second pass; meant for use with "
                         "--unseen-ce-override once the cross-encoders are adapted to the new country")
    ap.add_argument("--unseen-ce-idx", type=int, nargs="*", default=None,
                    help="explicit cross-encoder indices for unseen countries (overrides --unseen-n-ce)")
    ap.add_argument("--unseen-r", type=float, default=1.0, help="odds calibration factor for unseen countries")
    ap.add_argument("--train-records", default=None, help="training records.parquet (defines the seen countries)")
    ap.add_argument("--q-model", default=None, help="rival scorer (er.exp_band_rival); enables the second pass")
    ap.add_argument("--band-model2", default=None, help="second-pass band specialist for seen countries")
    ap.add_argument("--q-model-unseen", default=None)
    ap.add_argument("--band-model2-unseen", default=None)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=[0.005, 0.995])
    ap.add_argument("--min-p", type=float, default=1e-3)
    ap.add_argument("--raw-candidates", action="store_true", help="write the hybrid blocking set as candidate_pairs.tsv")
    ap.add_argument("--consensus", action="store_true")
    ap.add_argument("--rule", default="expected_f", choices=["expected_f", "thr"])
    ap.add_argument("--thr", type=float, default=0.75)
    a = ap.parse_args()
    T = time.time()
    booster = lgb.Booster(model_file=a.stage2)
    band_booster = lgb.Booster(model_file=a.band_model) if a.band_model else None
    band_unseen = lgb.Booster(model_file=a.band_model_unseen) if a.band_model_unseen else None
    ld = lambda f: lgb.Booster(model_file=f) if f else None
    q_seen, b2_seen, q_unseen, b2_unseen = map(ld, (a.q_model, a.band_model2, a.q_model_unseen, a.band_model2_unseen))
    seen_countries = None
    if a.train_records:   # open label set: "seen" = whatever countries the training split contains
        seen_countries = set(pd.read_parquet(a.train_records, columns=["country"]).country.astype(str).unique())
        print("countries seen in training:", sorted(seen_countries), flush=True)
    feats = booster.feature_name()
    path = Path(a.art) / "records.parquet"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fm, fc = open(out / "matching_results.tsv", "w"), open(out / "candidate_pairs.tsv", "w")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    n_s1 = n_match = 0
    for country, rows in country_rows(path).items():
        pp = Path(a.art) / f"pred_hyb_{country}.parquet"
        P = pd.read_parquet(pp)
        n_all = len(P)
        # pairs with p <= min_p can never be selected; dropping them leaves ranks, top-2 siblings and
        # group sums (essentially) unchanged and keeps stage-2 memory bounded (~30M -> ~2M pairs)
        P = P[P.p > a.min_p].reset_index(drop=True)
        print(f"{country}: {n_all} stage-1 pairs, {len(P)} with p > {a.min_p}", flush=True)
        cand = pd.read_parquet(pp.with_suffix(".cand.parquet"))
        # 1. cross-encoder(s) on the uncertain band (cached per country and model)
        band = P[(P.p > a.band[0]) & (P.p < a.band[1])][["a", "b", "p"]]
        band_path = Path(a.art) / f"band_{country}.parquet"
        band.to_parquet(band_path)
        ces = []
        unseen_c = seen_countries is not None and country not in seen_countries
        override = {int(k): v for k, v in (x.split("=", 1) for x in a.unseen_ce_override)} if unseen_c else {}
        for j, cm in enumerate(a.ce_model):
            view = "norm" if cm.rstrip("/").endswith("_norm") else "raw"   # *_norm models read normalised text
            ce_path = Path(a.art) / f"ce{j}_{country}.parquet"
            if j in override:
                cm = override[j]
                ce_path = Path(a.art) / f"ce{j}u_{Path(cm.rstrip('/')).name}_{country}.parquet"
            if not ce_path.exists():
                from types import SimpleNamespace

                from .cross_encoder import score
                score(SimpleNamespace(pairs=str(band_path), band=None, art=a.art, model=cm,
                                      out=str(ce_path), max_len=128, view=view))
            ces.append(pd.read_parquet(ce_path).rename(columns={"ce": f"ce{j}"}))
        ce = ces[0].rename(columns={"ce0": "ce"})
        # 2. cluster context + stage-2 model
        sub = read_country(path, country, rows)
        ctx = context_features(P, sub[["name_core", "addr_clean"]])
        if a.consensus:
            from .biencoder import load_emb
            from .stage2 import consensus_features
            ctx = consensus_features(ctx, load_emb(a.art, country), rows, col="p1")
        ctx = ctx.merge(ce, on=["a", "b"], how="left")
        if "ce" not in ctx:
            ctx["ce"] = np.nan
        cols = feats
        if feats[0].startswith("Column_"):  # model trained on a bare array: same column order as exp_stage2
            cols = [c for c in ctx.columns if c not in ("a", "b")]
            assert len(cols) == booster.num_feature(), (cols, booster.num_feature())
        P2 = ctx[["a", "b"]].assign(p=booster.predict(ctx[cols].to_numpy(np.float32), num_threads=22))
        unseen = seen_countries is not None and country not in seen_countries
        src_arr = sub.src.to_numpy()
        src_of = lambda rows_b: src_arr[np.searchsorted(rows, rows_b)]
        ces_unseen = [ces[j] for j in a.unseen_ce_idx] if a.unseen_ce_idx is not None else ces[:a.unseen_n_ce]
        if unseen and a.unseen_use_seen and b2_seen is not None:
            P2 = apply_band2(P2, ctx, ces, q_seen, b2_seen, a, country, src_of)
            print(f"  {country}: unseen in training -> seen-country second pass with adapted cross-encoders "
                  f"{sorted(override)}", flush=True)
        elif unseen and b2_unseen is not None:
            P2 = apply_band2(P2, ctx, ces_unseen, q_unseen, b2_unseen, a, country, src_of)
            print(f"  {country}: unseen in training -> second pass with {a.unseen_n_ce} CEs", flush=True)
        elif not unseen and b2_seen is not None:
            P2 = apply_band2(P2, ctx, ces, q_seen, b2_seen, a, country, src_of)
        elif unseen and band_unseen is not None:
            # open-set rule: a country absent from training uses the band specialist WITHOUT the cross-encoders
            # trained on hard examples of the training countries (they transfer worst to an unseen country)
            P2 = apply_band(P2, ctx, ces[:a.unseen_n_ce], band_unseen, a, country)
            if a.unseen_r != 1.0:
                q = P2.p.clip(1e-7, 1 - 1e-7).to_numpy()
                P2 = P2.assign(p=1 / (1 + (1 - q) / (q * a.unseen_r)))
            print(f"  {country}: unseen in training -> robust band model ({a.unseen_n_ce} CEs), r={a.unseen_r}", flush=True)
        elif band_booster is not None:
            P2 = apply_band(P2, ctx, ces, band_booster, a, country)
        P2.to_parquet(Path(a.art) / f"pred_s2_{country}.parquet")
        # 3. decide + write
        ex = exclusive(P2)
        sel = expected_f_fast(ex) if a.rule == "expected_f" else ex[ex.p >= a.thr]
        ids = pd.Series(sub.entity_id.to_numpy(dtype=object), index=sub.index)
        s1_rows = np.sort(sub.index[sub.src.to_numpy() == 1].to_numpy())
        append_lists(fm, s1_rows, sel.a.to_numpy(), sel.b.to_numpy(), ids)
        # candidate_pairs.tsv = the last filtering stage: the pairs the matching models actually score (stage-1
        # learned filter p1 > min_p on top of the hybrid blocking set); --raw-candidates writes the hybrid set instead
        cp = cand if a.raw_candidates else P
        append_lists(fc, s1_rows, cp.a.to_numpy(), cp.b.to_numpy(), ids)
        n_s1 += len(s1_rows)
        n_match += len(sel)
        print(f"{country}: {len(sel)} matches for {len(s1_rows)} S1 ({len(ce)} CE-scored pairs) "
              f"{time.time() - T:.0f}s", flush=True)
        del P, P2, ctx, sub, cand, ex, sel, ids
        gc.collect()
    fm.close()
    fc.close()
    print(f"wrote {out}: {n_match} matches, mean {n_match / n_s1:.2f}/S1 {time.time() - T:.0f}s")


if __name__ == "__main__":
    main()
