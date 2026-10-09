#!/usr/bin/env bash
# Night chain v2: extra-bucket band data + France probes + v10 (validation-picked).
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
set -a; . ./.env; set +a
K=.venv/bin/kaggle
# A. extra-bucket band data (heavy CPU) once v9's heavy CPU phases are over
while ! grep -qE "V9_TRAIN_DONE|Traceback" v9.log 2>/dev/null; do sleep 60; done
nice -n 5 $PY -m er.build_extra_band --art $T --gt $GT --model final/v9_models/model.txt --buckets 4 5 --tag b45 > extra.log 2>&1 || { echo "extra band build failed"; }
$PY -c "import pandas as pd,glob; pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('$T/extra_band_b45_*_pairs.parquet'))]).to_parquet('$T/extra_b45_pairs.parquet')"
# B. v9 done -> France probes
while ! grep -qE "V9_DONE|Traceback|failed" v9.log 2>/dev/null; do sleep 60; done
grep -q V9_DONE v9.log || { echo "v9 failed"; exit 1; }
for v in "fL 2.0" "fC 0.5"; do set -- $v
  $PY -m er.redecide --art $S --r 1.0 --r-country France=$2 --out output_v9$1 | tail -1
  D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v9$1; mkdir -p $D; cp output_v9$1/matching_results.tsv $D/
  python3 student_resource/utils/validate_submission.py --matching $D/matching_results.tsv --test-dir student_resource/dataset/test --check-ids 2>&1 | grep -E "^PASS|^FAIL"
done
echo PROBES_DONE
# C. GPU: score extra band pairs with the leak-free CEs (base, mDeBERTa, normalised)
$PY -m er.cross_encoder score --art $T --pairs $T/extra_b45_pairs.parquet --model models/ce_base --out $T/extra_ce_base.parquet
$PY -m er.cross_encoder score --art $T --pairs $T/extra_b45_pairs.parquet --model models/ce_mdeberta --out $T/extra_ce_md.parquet
$PY -m er.cross_encoder score --art $T --pairs $T/extra_b45_pairs.parquet --model models/ce_norm --view norm --out $T/extra_ce_norm.parquet
# D. v10a: 4 leak-free CEs, bucket 0 + extra buckets 4-5
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v9 --ce $T/ce0_v9_oof.parquet $T/ce2_v9_oof.parquet $T/ce3_v9_oof.parquet \
    --extra-tag b45 --extra-ce $T/extra_ce_base.parquet $T/extra_ce_md.parquet $T/extra_ce_norm.parquet --save-tag _v10a > v10a.log 2>&1
grep -E "extra training|band AUC|F0.5 using" v10a.log
BEST=v9; CES_BEST="models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm"
pick() { $PY -c "import json; a=json.load(open('$T/band_metrics_$1.json'))['band']; b=json.load(open('$T/band_metrics_$2.json'))['band']; print('yes' if b > a + 1e-5 else 'no')"; }
[ -f $T/band_metrics_v10a.json ] && [ "$(pick v9 v10a)" = "yes" ] && { BEST=v10a; CES_BEST="models/ce_base models/ce_mdeberta models/ce_norm"; }
# E. v10b: + e5-large if Kaggle delivered by now (don't wait past 06:00 local time)
if $K kernels status sachetv/amlc-ce-e5large 2>&1 | grep -q COMPLETE || { until $K kernels status sachetv/amlc-ce-e5large 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL" || [ $(date +%H) -ge 6 -a $(date +%H) -lt 12 ]; do sleep 120; done; $K kernels status sachetv/amlc-ce-e5large 2>&1 | grep -q COMPLETE; }; then
  rm -rf models/ce_large_raw && mkdir -p models/ce_large_raw && $K kernels output sachetv/amlc-ce-e5large -p models/ce_large_raw > /dev/null 2>&1
  M=$(dirname $(find models/ce_large_raw -name config.json | head -1)); rm -rf models/ce_large && cp -r $M models/ce_large
  $PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_large --out $T/ce4_v9_oof.parquet
  $PY -m er.cross_encoder score --art $T --pairs $T/extra_b45_pairs.parquet --model models/ce_large --out $T/extra_ce_large.parquet
  $PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v9 --ce $T/ce0_v9_oof.parquet $T/ce2_v9_oof.parquet $T/ce3_v9_oof.parquet $T/ce4_v9_oof.parquet \
      --extra-tag b45 --extra-ce $T/extra_ce_base.parquet $T/extra_ce_md.parquet $T/extra_ce_norm.parquet $T/extra_ce_large.parquet --save-tag _v10b > v10b.log 2>&1
  grep -E "extra training|band AUC|F0.5 using" v10b.log
  [ -f $T/band_metrics_v10b.json ] && [ "$(pick $BEST v10b)" = "yes" ] && { BEST=v10b; CES_BEST="models/ce_base models/ce_mdeberta models/ce_norm models/ce_large"; }
fi
echo "v10 choice: $BEST ($CES_BEST)"
[ "$BEST" = "v9" ] && { echo "V10_NOT_BETTER"; exit 0; }
# F. test inference for the chosen v10 (cached CE test scores are per index -> use a fresh dir of CE caches)
mkdir -p final/v10_models && cp final/v9_models/model.txt final/v9_models/stage2.txt final/v10_models/ && cp $T/band_model_$BEST.txt final/v10_models/band.txt
# reuse v9's cached test-band CE scores (same stage 1 -> same band pairs); v9 order: 0 base, 1 base2, 2 mDeBERTa, 3 norm
mkdir -p $S/v9_ce && mv $S/ce*_*.parquet $S/v9_ce/ 2>/dev/null
for c in US France India; do
  cp $S/v9_ce/ce0_$c.parquet $S/ce0_$c.parquet; cp $S/v9_ce/ce2_$c.parquet $S/ce1_$c.parquet; cp $S/v9_ce/ce3_$c.parquet $S/ce2_$c.parquet
done   # ce3 (e5-large, v10b only) is scored fresh by predict_stage2
$PY -m er.predict_stage2 --art $S --ce-model $CES_BEST --stage2 final/v10_models/stage2.txt --band-model final/v10_models/band.txt --out output_v10 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v10/matching_results.tsv \
    --candidate output_v10/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v10; mkdir -p $D && cp output_v10/matching_results.tsv $D/
echo V10_DONE
