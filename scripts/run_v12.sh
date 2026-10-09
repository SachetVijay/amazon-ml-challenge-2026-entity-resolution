#!/bin/bash
# v12: band specialists retrained on test-like (full-density) competition context, then test re-prediction.
cd ~/amlc
PY=.venv/bin/python
T=artifacts/train
GT=student_resource/dataset/train/train_ground_truth.tsv
BASE="$T/ce0_v9_oof.parquet $T/ce1_v9_oof.parquet $T/ce2_v9_oof.parquet $T/ce3_v9_oof.parquet"
until grep -q COMP_DONE comp.log; do sleep 30; done
grep -q Traceback comp.log && { echo "COMP FAILED" > v12.log; exit 1; }
nice -n 5 $PY -m er.exp_band_fd --art $T --gt $GT --s2-tag _v9 --ce $BASE $T/ce5_v9_oof.parquet $T/ce4_v9_oof.parquet \
    --comp-tag fd --save-tag _v12 > v12.log 2>&1 || exit 1
nice -n 5 $PY -m er.exp_band_fd --art $T --gt $GT --s2-tag _v9 --ce $BASE --comp-tag fd --save-tag _v12u > v12u.log 2>&1 || exit 1
mkdir -p final/v12_models backup_pred_s2_final_r1
cp -n artifacts/test/pred_s2_*.parquet backup_pred_s2_final_r1/
cp final/final_models/model.txt final/final_models/stage2.txt final/v12_models/
cp $T/band_model_v12.txt final/v12_models/band.txt
cp $T/band_model_v12u.txt final/v12_models/band_unseen.txt
nice -n 5 $PY -m er.predict_stage2 --art artifacts/test --ce-model models/ce_base models/ce_base2 models/ce_mdeberta \
    models/ce_norm models/ce_hard models/ce_large --stage2 final/v12_models/stage2.txt --band-model final/v12_models/band.txt \
    --band-model-unseen final/v12_models/band_unseen.txt --unseen-n-ce 4 --unseen-r 1.0 \
    --train-records artifacts/train/records.parquet --out output_v12 > v12_pred.log 2>&1
echo V12_DONE >> v12_pred.log
