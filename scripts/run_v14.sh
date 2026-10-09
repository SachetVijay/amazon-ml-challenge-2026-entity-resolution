#!/bin/bash
# v14 = v13 + per-source slot features in the second-pass specialists (seen + unseen countries).
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train
GT=student_resource/dataset/train/train_ground_truth.tsv
BASE="$T/ce0_v9_oof.parquet $T/ce1_v9_oof.parquet $T/ce2_v9_oof.parquet $T/ce3_v9_oof.parquet"
nice -n 5 $PY -m er.exp_band_rival --art $T --gt $GT --s2-tag _v9 --ce $BASE \
    --rival-ce $T/rce0.parquet $T/rce1.parquet $T/rce2.parquet $T/rce3.parquet --save-tag _v14u --pick-best > v14u.log 2>&1 || exit 1
mkdir -p final/v14_models backup_pred_s2_v13b
cp -n artifacts/test/pred_s2_*.parquet backup_pred_s2_v13b/
cp final/v13_models/model.txt final/v13_models/stage2.txt final/v13_models/band.txt final/v13_models/band_unseen.txt final/v14_models/
cp $T/q_model_v14.txt final/v14_models/q.txt; cp $T/band_model_v14.txt final/v14_models/band2.txt
cp $T/q_model_v14u.txt final/v14_models/q_unseen.txt; cp $T/band_model_v14u.txt final/v14_models/band2_unseen.txt
M=final/v14_models
nice -n 5 $PY -m er.predict_stage2 --art artifacts/test --ce-model models/ce_base models/ce_base2 models/ce_mdeberta \
    models/ce_norm models/ce_hard models/ce_large --stage2 $M/stage2.txt --band-model $M/band.txt \
    --band-model-unseen $M/band_unseen.txt --unseen-n-ce 4 --unseen-r 1.0 --train-records artifacts/train/records.parquet \
    --q-model $M/q.txt --band-model2 $M/band2.txt --q-model-unseen $M/q_unseen.txt --band-model2-unseen $M/band2_unseen.txt \
    --out output_v14 > v14_pred.log 2>&1
echo V14_DONE >> v14_pred.log
