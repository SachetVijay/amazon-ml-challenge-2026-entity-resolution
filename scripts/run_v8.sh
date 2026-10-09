#!/usr/bin/env bash
# v8 = v7 + normalised-text cross-encoder (Kaggle) as a 4th band-specialist input. No stage-1 change:
# reuses v7 stage-1 predictions, cached test features and the 3 cached cross-encoder test scores.
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
set -a; . ./.env; set +a
K=.venv/bin/kaggle
while ! grep -q V7_DONE v7safe.log 2>/dev/null; do grep -q Traceback v7safe.log && { echo "v7 failed"; exit 1; }; sleep 60; done
until $K kernels status sachetv/amlc-ce-norm 2>&1 | grep -qE "COMPLETE|ERROR"; do sleep 60; done
$K kernels status sachetv/amlc-ce-norm 2>&1 | grep -q COMPLETE || { echo "kaggle norm CE failed"; exit 1; }
rm -rf models/ce_norm_raw && mkdir -p models/ce_norm_raw
$K kernels output sachetv/amlc-ce-norm -p models/ce_norm_raw > /dev/null 2>&1
M=$(dirname $(find models/ce_norm_raw -name config.json | head -1)); rm -rf models/ce_norm && cp -r $M models/ce_norm
python3 -c "import json,glob; L=json.load(open(glob.glob('models/ce_norm_raw/*.log')[0])); o=[e['data'].strip() for e in L if e['stream_name']=='stdout']; print(o[0]); print('\n'.join(o[-3:]))"
$PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_norm --view norm --out $T/ce4_v7_oof.parquet || exit 1
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet $T/ce2_v7_oof.parquet $T/ce3_v7_oof.parquet $T/ce4_v7_oof.parquet --save-tag _4ce > band4.log 2>&1 || exit 1
grep -E "band AUC|F0.5 using" band4.log
BETTER=$($PY -c "import json; a=json.load(open('$T/band_metrics_3ce.json'))['band']; b=json.load(open('$T/band_metrics_4ce.json'))['band']; print(f'3ce {a:.5f} 4ce {b:.5f}'); print('yes' if b >= a - 2e-4 else 'no')" | tee /dev/stderr | tail -1)
# gate = "not measurably worse on US/India": this model targets France, which validation cannot see
[ "$BETTER" = "yes" ] || { echo "V8_NOT_BETTER (normalised CE hurts US/India validation by > 0.0002)"; exit 0; }
mkdir -p final/v8_models && cp final/v7_models/model.txt final/v7_models/stage2.txt final/v8_models/ && cp $T/band_model_4ce.txt final/v8_models/band.txt
$PY -m er.predict_stage2 --art $S --ce-model models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm \
    --stage2 final/v8_models/stage2.txt --band-model final/v8_models/band.txt --out output_v8 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v8/matching_results.tsv \
    --candidate output_v8/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v8; mkdir -p $D && cp output_v8/matching_results.tsv $D/
echo V8_DONE
