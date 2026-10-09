#!/bin/bash
# v16 = v14 + France (unseen country) scored with ce1 adapted to French pseudo-labels.
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
until grep -q FE2_DONE fe2.log 2>/dev/null; do sleep 30; done
mkdir -p backup_pred_s2_v16pre && cp artifacts/test/pred_s2_*.parquet backup_pred_s2_v16pre/
M=final/v14_models
nice -n 5 $PY -m er.predict_stage2 --art artifacts/test --ce-model models/ce_base models/ce_base2 models/ce_mdeberta \
    models/ce_norm models/ce_hard models/ce_large --stage2 $M/stage2.txt --band-model $M/band.txt \
    --band-model-unseen $M/band_unseen.txt --unseen-n-ce 4 --train-records artifacts/train/records.parquet \
    --q-model $M/q.txt --band-model2 $M/band2.txt --q-model-unseen $M/q_unseen.txt --band-model2-unseen $M/band2_unseen.txt \
    --unseen-ce-override 1=models/ce_base2_fr --out output_v16 > v16_pred.log 2>&1
mkdir -p backup_pred_s2_v16 && cp artifacts/test/pred_s2_*.parquet backup_pred_s2_v16/
cp backup_pred_s2_v14/pred_s2_*.parquet artifacts/test/
echo V16_DONE >> v16_pred.log
