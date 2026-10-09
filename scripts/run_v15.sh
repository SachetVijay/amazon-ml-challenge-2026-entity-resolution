#!/bin/bash
# v15 candidates: France (unseen-country) second-pass specialist built on French-robust cross-encoders only.
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train
GT=student_resource/dataset/train/train_ground_truth.tsv
for v in "b 3" "a 2 3"; do set -- $v; tag=$1; shift; idx="$@"
  ces=""; rces=""; for j in $idx; do ces="$ces $T/ce${j}_v9_oof.parquet"; rces="$rces $T/rce$j.parquet"; done
  nice -n 5 $PY -m er.exp_band_rival --art $T --gt $GT --s2-tag _v9 --ce $ces --ce-index $idx --rival-ce $rces \
      --save-tag _v15u$tag --pick-best > v15u$tag.log 2>&1 || exit 1
  M=final/v15$tag"_models"; mkdir -p $M; cp final/v14_models/*.txt $M/
  cp $T/q_model_v15u$tag.txt $M/q_unseen.txt; cp $T/band_model_v15u$tag.txt $M/band2_unseen.txt
  nice -n 5 $PY -m er.predict_stage2 --art artifacts/test --ce-model models/ce_base models/ce_base2 models/ce_mdeberta \
      models/ce_norm models/ce_hard models/ce_large --stage2 $M/stage2.txt --band-model $M/band.txt \
      --band-model-unseen $M/band_unseen.txt --unseen-ce-idx $idx --train-records artifacts/train/records.parquet \
      --q-model $M/q.txt --band-model2 $M/band2.txt --q-model-unseen $M/q_unseen.txt --band-model2-unseen $M/band2_unseen.txt \
      --out output_v15$tag > v15$tag"_pred.log" 2>&1
  mkdir -p backup_pred_s2_v15$tag && cp artifacts/test/pred_s2_France.parquet backup_pred_s2_v15$tag/
done
cp backup_pred_s2_v14/pred_s2_*.parquet artifacts/test/
echo V15_DONE > v15.done
