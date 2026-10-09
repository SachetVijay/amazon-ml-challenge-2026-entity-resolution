#!/usr/bin/env bash
# v5 = French-robust normalisation + acronym feature (stage-1 retrain) + best cross-encoder + stage 2.
#   bash scripts/run_v5.sh models/ce_base      (cross-encoder dir to use)
set -e
CE=${1:?cross-encoder dir}
export PYTHONPATH=src
PY=.venv/bin/python
GT=student_resource/dataset/train/train_ground_truth.tsv
T=artifacts/train; S=artifacts/test
# 1. re-normalise records (row order must be unchanged: candidate caches use global row ids)
for A in $T $S; do
  [ -f $A/records_v4.parquet ] || mv $A/records.parquet $A/records_v4.parquet
  [ -f $A/records.parquet ] || $PY -m er.prep --split-dir student_resource/dataset/$(basename $A) --prefix $(basename $A) --out $A
  $PY -c "import pandas as pd,sys; a=pd.read_parquet('$A/records_v4.parquet',columns=['entity_id']); b=pd.read_parquet('$A/records.parquet',columns=['entity_id']); assert a.entity_id.equals(b.entity_id), 'row order changed'; print('$A row order OK', len(b))"
done
# 2. stage-1 features + model (old v4 feature files kept as backup)
mkdir -p $T/v4_backup
for f in $T/feats_hyb5_*.parquet $T/oof_hyb5.parquet $T/model_hyb5.txt; do [ -f $f ] && mv $f $T/v4_backup/; done
$PY -m er.train_full --split-dir student_resource/dataset/train --prefix train --art $T --gt $GT --hybrid --dense-k 5
# 3. cross-encoder on the new OOF uncertain band + stage 2
$PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model $CE --out $T/ce_v5_oof.parquet
$PY -m er.exp_stage2 --art $T --gt $GT --tag _hyb5 --ce $T/ce_v5_oof.parquet
echo V5_TRAIN_DONE
# 4. test: stage 1 -> cross-encoder band -> stage 2 -> TSVs
mkdir -p $S/v4_backup
for f in $S/pred_hyb_* $S/ce_*.parquet $S/band_*.parquet $S/pred_s2_*; do [ -f $f ] && mv $f $S/v4_backup/; done
$PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art $S --model $T/model_hyb5.txt \
    --hybrid --dense-k 5 --rule expected_f --out output_v5_s1 --chunk-s1 100000 --resume
$PY -m er.predict_stage2 --art $S --ce-model $CE --stage2 $T/stage2_hyb5_ce.txt --out output_v5
python3 student_resource/utils/validate_submission.py --matching output_v5/matching_results.tsv \
    --candidate output_v5/candidate_pairs.tsv --test-dir student_resource/dataset/test
echo V5_DONE
