#!/usr/bin/env bash
# v9 = full retrain with native-script Indian states + mojibake cleanup (+ all cross-encoders that help).
# Memory-safe: one heavy CPU job at a time. Starts after v8 finishes (either outcome).
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
while ! grep -qE "V8_DONE|V8_NOT_BETTER|failed|Traceback" v8.log 2>/dev/null; do sleep 60; done
if grep -q V8_DONE v8.log; then CES="models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm"; NCE=4
else CES="models/ce_base models/ce_base2 models/ce_mdeberta"; NCE=3; fi
echo "v9 uses $NCE cross-encoders"
# 1. re-normalise (row order verified), back up v7/v8 artifacts
for A in $T $S; do
  mkdir -p $A/v8_backup; mv $A/records.parquet $A/v8_backup/records.parquet
  $PY -m er.prep --split-dir student_resource/dataset/$(basename $A) --prefix $(basename $A) --out $A || exit 1
  $PY -c "import pandas as pd; a=pd.read_parquet('$A/v8_backup/records.parquet',columns=['entity_id']); b=pd.read_parquet('$A/records.parquet',columns=['entity_id']); assert a.entity_id.equals(b.entity_id); print('$A row order OK')" || exit 1
done
for f in $T/feats_hyb5_*.parquet $T/oof_hyb5.parquet $T/model_hyb5.txt; do [ -f $f ] && mv $f $T/v8_backup/; done
for f in $S/pred_hyb_* $S/ce*_*.parquet $S/band_*.parquet $S/pred_s2_*; do [ -f $f ] && mv $f $S/v8_backup/; done
mv $S/feats_cache $S/v8_backup/feats_cache 2>/dev/null
# 2. stage 1 retrain (heavy)
$PY -m er.train_full --split-dir student_resource/dataset/train --prefix train --art $T --gt $GT --hybrid --dense-k 5 > v9_train.log 2>&1 || exit 1
mkdir -p final/v9_models && cp $T/model_hyb5.txt final/v9_models/model.txt
grep -E "OOF exclusive thr 0.75|US: thr|India: thr" v9_train.log
# 3. test stage 1 (heavy) || OOF cross-encoder scoring (GPU, light)
( i=0; for m in $CES; do v=raw; case $m in *_norm) v=norm;; esac
    $PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model $m --view $v --out $T/ce${i}_v9_oof.parquet; i=$((i+1)); done
  echo OOF_CE_DONE > v9_oofce.flag ) > v9_oofce.log 2>&1 &
$PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art $S --model final/v9_models/model.txt \
    --hybrid --dense-k 5 --rule expected_f --out output_v9_s1 --chunk-s1 100000 --resume > v9_s1.log 2>&1 || exit 1
wait
# 4. stage 2 + band specialist (heavy, sequential)
$PY -m er.exp_stage2 --art $T --gt $GT --tag _hyb5 --ce $T/ce0_v9_oof.parquet --save-tag _v9 > v9_s2.log 2>&1 || exit 1
cp $T/stage2_hyb5_v9.txt final/v9_models/stage2.txt
OOFS=$(for i in $(seq 0 $((NCE-1))); do printf "$T/ce${i}_v9_oof.parquet "; done)
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v9 --ce $OOFS --save-tag _v9 > v9_band.log 2>&1 || exit 1
grep -E "band AUC|F0.5 using" v9_band.log
cp $T/band_model_v9.txt final/v9_models/band.txt
echo V9_TRAIN_DONE
# 5. test: cross-encoders on the band + stage 2 + band specialist -> deliver
$PY -m er.predict_stage2 --art $S --ce-model $CES --stage2 final/v9_models/stage2.txt --band-model final/v9_models/band.txt --out output_v9 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v9/matching_results.tsv \
    --candidate output_v9/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v9; mkdir -p $D && cp output_v9/matching_results.tsv $D/
echo V9_DONE
