#!/usr/bin/env bash
# v7 training: French region/department canonicalisation + street/saint + French abbreviations (normalisation
# change -> re-prep + full feature recompute), then stage-1 retrain. v6 artifacts backed up first.
set -e
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
for A in $T $S; do
  mkdir -p $A/v6_backup
  mv $A/records.parquet $A/v6_backup/records.parquet
  $PY -m er.prep --split-dir student_resource/dataset/$(basename $A) --prefix $(basename $A) --out $A
  $PY -c "import pandas as pd; a=pd.read_parquet('$A/v6_backup/records.parquet',columns=['entity_id']); b=pd.read_parquet('$A/records.parquet',columns=['entity_id']); assert a.entity_id.equals(b.entity_id); print('$A row order OK')"
done
for f in $T/feats_hyb5_*.parquet $T/oof_hyb5.parquet $T/model_hyb5.txt $T/stage2_hyb5_*.txt $T/oof_stage2_hyb5_*.parquet $T/band_model.txt $T/oof_band.parquet; do [ -f $f ] && mv $f $T/v6_backup/; done
$PY -m er.train_full --split-dir student_resource/dataset/train --prefix train --art $T --gt $GT --hybrid --dense-k 5
mkdir -p final/v7_models && cp $T/model_hyb5.txt final/v7_models/model.txt
echo V7_STAGE1_DONE
