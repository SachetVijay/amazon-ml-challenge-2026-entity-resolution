#!/usr/bin/env bash
# Night chain: (1) after v9 -> France-only calibration probes; (2) after Kaggle e5-large -> v10 (5 cross-encoders).
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
set -a; . ./.env; set +a
K=.venv/bin/kaggle
while ! grep -qE "V9_DONE|Traceback|failed" v9.log 2>/dev/null; do sleep 60; done
grep -q V9_DONE v9.log || { echo "v9 failed; stopping night chain"; exit 1; }
# (1) France-only calibration probes (light: decision step only)
for v in "fL 2.0" "fC 0.5"; do set -- $v
  $PY -m er.redecide --art $S --r 1.0 --r-country France=$2 --out output_v9$1 | tail -2
  D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v9$1; mkdir -p $D; cp output_v9$1/matching_results.tsv $D/
  python3 student_resource/utils/validate_submission.py --matching $D/matching_results.tsv --test-dir student_resource/dataset/test --check-ids 2>&1 | grep -E "^PASS|^FAIL"
done
echo PROBES_DONE
# (2) v10 when the large cross-encoder is ready
until $K kernels status sachetv/amlc-ce-e5large 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 120; done
$K kernels status sachetv/amlc-ce-e5large 2>&1 | grep -q COMPLETE || { echo "e5-large failed on Kaggle"; exit 1; }
rm -rf models/ce_large_raw && mkdir -p models/ce_large_raw && $K kernels output sachetv/amlc-ce-e5large -p models/ce_large_raw > /dev/null 2>&1
M=$(dirname $(find models/ce_large_raw -name config.json | head -1)); rm -rf models/ce_large && cp -r $M models/ce_large
$PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_large --out $T/ce4_v9_oof.parquet || exit 1
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v9 --ce $T/ce0_v9_oof.parquet $T/ce1_v9_oof.parquet $T/ce2_v9_oof.parquet $T/ce3_v9_oof.parquet $T/ce4_v9_oof.parquet --save-tag _v10 > v10_band.log 2>&1 || exit 1
grep -E "band AUC|F0.5 using" v10_band.log
OK=$($PY -c "import json; a=json.load(open('$T/band_metrics_v9.json'))['band']; b=json.load(open('$T/band_metrics_v10.json'))['band']; print(f'v9 {a:.5f} v10 {b:.5f}', flush=True); print('yes' if b >= a - 2e-4 else 'no')" | tail -1)
[ "$OK" = "yes" ] || { echo "V10_NOT_BETTER"; exit 0; }
mkdir -p final/v10_models && cp final/v9_models/model.txt final/v9_models/stage2.txt final/v10_models/ && cp $T/band_model_v10.txt final/v10_models/band.txt
$PY -m er.predict_stage2 --art $S --ce-model models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm models/ce_large \
    --stage2 final/v10_models/stage2.txt --band-model final/v10_models/band.txt --out output_v10 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v10/matching_results.tsv \
    --candidate output_v10/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v10; mkdir -p $D && cp output_v10/matching_results.tsv $D/
echo V10_DONE
