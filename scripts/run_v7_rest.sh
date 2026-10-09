#!/usr/bin/env bash
# v7 after stage-1 retrain: OOF CE scoring (2 CEs) -> stage 2 -> band specialist (1 vs 2 CEs, auto-pick)
# -> test inference (stage 1 in parallel) -> stage 2 + band -> validate -> deliver to Downloads\AMLC_Submissions\v7
set -e
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
while ! grep -q V7_STAGE1_DONE v7t.log 2>/dev/null; do grep -q "Traceback" v7t.log 2>/dev/null && exit 1; sleep 60; done
# test stage 1 in the background (CPU) while the band models train
mkdir -p $S/v6_backup; for f in $S/pred_hyb_* $S/ce*_*.parquet $S/band_*.parquet $S/pred_s2_*; do [ -f $f ] && mv $f $S/v6_backup/; done
rm -rf $S/feats_cache
( $PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art $S --model final/v7_models/model.txt \
    --hybrid --dense-k 5 --rule expected_f --out output_v7_s1 --chunk-s1 100000 --resume > v7_s1.log 2>&1; echo S1_DONE >> v7_s1.log ) &
$PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_base --out $T/ce_v7_oof.parquet
$PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_base2 --out $T/ce2_v7_oof.parquet
$PY -m er.exp_stage2 --art $T --gt $GT --tag _hyb5 --ce $T/ce_v7_oof.parquet --save-tag _v7
cp $T/stage2_hyb5_v7.txt final/v7_models/stage2.txt
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet --save-tag _1ce
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet $T/ce2_v7_oof.parquet --save-tag _2ce
CHOICE=$($PY -c "import json; a=json.load(open('$T/band_metrics_1ce.json')); b=json.load(open('$T/band_metrics_2ce.json')); print('2ce' if b['band'] > a['band'] + 1e-5 else '1ce')")
echo "band choice: $CHOICE"; cp $T/band_model_$CHOICE.txt final/v7_models/band.txt
if [ "$CHOICE" = "2ce" ]; then CES="models/ce_base models/ce_base2"; else CES="models/ce_base"; fi
echo V7_TRAIN_DONE
while ! grep -q S1_DONE v7_s1.log; do sleep 60; done
grep -q "Traceback" v7_s1.log && { echo "stage 1 failed"; exit 1; }
$PY -m er.predict_stage2 --art $S --ce-model $CES --stage2 final/v7_models/stage2.txt --band-model final/v7_models/band.txt --out output_v7
python3 student_resource/utils/validate_submission.py --matching output_v7/matching_results.tsv \
    --candidate output_v7/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v7; mkdir -p $D && cp output_v7/matching_results.tsv $D/
echo V7_DONE
