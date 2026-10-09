#!/bin/bash
# v13: second-pass band specialist with cross-encoder-measured rival strength (er.exp_band_rival).
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train
GT=student_resource/dataset/train/train_ground_truth.tsv
set -a; . ./.env; set +a
K=.venv/bin/kaggle
fetch() {  # $1 kernel, $2 feature index j
  until $K kernels status sachetv/$1 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 60; done
  if $K kernels status sachetv/$1 2>&1 | grep -q COMPLETE && [ ! -f $T/rce$2.parquet ]; then
    rm -rf kg_$1 && mkdir kg_$1 && $K kernels output sachetv/$1 -p kg_$1 > /dev/null 2>&1
    $PY -c "import pandas as pd; d = pd.read_parquet('kg_$1/ce_rival.parquet'); d[['a','b','ce']].to_parquet('$T/rce$2.parquet.tmp'); print('$1', len(d))" \
      && mv $T/rce$2.parquet.tmp $T/rce$2.parquet
  fi
}
fetch amlc-score-rival-large 5 &
fetch amlc-score-rival-mdeberta 2 &
wait
until [ $(ls $T/rce[0-5].parquet 2>/dev/null | wc -l) -eq 6 ]; do grep -q Traceback rce.log && { echo "RCE FAILED"; exit 1; }; sleep 60; done
echo "all rival CE scores present"
BASE="$T/ce0_v9_oof.parquet $T/ce1_v9_oof.parquet $T/ce2_v9_oof.parquet $T/ce3_v9_oof.parquet"
nice -n 5 $PY -m er.exp_band_rival --art $T --gt $GT --s2-tag _v9 --ce $BASE $T/ce5_v9_oof.parquet $T/ce4_v9_oof.parquet \
    --rival-ce $T/rce0.parquet $T/rce1.parquet $T/rce2.parquet $T/rce3.parquet $T/rce4.parquet $T/rce5.parquet \
    --save-tag _v13 > v13.log 2>&1 || exit 1
nice -n 5 $PY -m er.exp_band_rival --art $T --gt $GT --s2-tag _v9 --ce $BASE \
    --rival-ce $T/rce0.parquet $T/rce1.parquet $T/rce2.parquet $T/rce3.parquet --save-tag _v13u > v13u.log 2>&1 || exit 1
mkdir -p final/v13_models backup_pred_s2_v12
cp -n artifacts/test/pred_s2_*.parquet backup_pred_s2_v12/
cp final/v12_models/*.txt final/v13_models/
cp $T/q_model_v13.txt final/v13_models/q.txt; cp $T/band_model_v13.txt final/v13_models/band2.txt
cp $T/q_model_v13u.txt final/v13_models/q_unseen.txt; cp $T/band_model_v13u.txt final/v13_models/band2_unseen.txt
M=final/v13_models
nice -n 5 $PY -m er.predict_stage2 --art artifacts/test --ce-model models/ce_base models/ce_base2 models/ce_mdeberta \
    models/ce_norm models/ce_hard models/ce_large --stage2 $M/stage2.txt --band-model $M/band.txt \
    --band-model-unseen $M/band_unseen.txt --unseen-n-ce 4 --unseen-r 1.0 --train-records artifacts/train/records.parquet \
    --q-model $M/q.txt --band-model2 $M/band2.txt --q-model-unseen $M/q_unseen.txt --band-model2-unseen $M/band2_unseen.txt \
    --out output_v13 > v13_pred.log 2>&1
echo V13_DONE >> v13_pred.log
