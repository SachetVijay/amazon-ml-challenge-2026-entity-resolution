#!/bin/bash
# Serialized predictions (they share artifacts/test/pred_s2_*): v17a -> v17c -> v17b (if e5-large arrives).
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
CES="models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm models/ce_hard models/ce_large"
run() {  # $1 out  $2 model dir  rest: extra args
  out=$1; M=$2; shift 2
  nice -n 5 $PY -m er.predict_stage2 --art artifacts/test --ce-model $CES --stage2 $M/stage2.txt --band-model $M/band.txt \
      --band-model-unseen $M/band_unseen.txt --unseen-n-ce 4 --train-records artifacts/train/records.parquet \
      --q-model $M/q.txt --band-model2 $M/band2.txt --q-model-unseen $M/q_unseen.txt --band-model2-unseen $M/band2_unseen.txt \
      "$@" --out $out > $out.log 2>&1
  mkdir -p backup_pred_s2_$out && cp artifacts/test/pred_s2_France.parquet backup_pred_s2_$out/
  echo "$out done" >> v17.done
}
until grep -q "scored models/ce_mdeberta_fr" gpu_chain.log 2>/dev/null; do sleep 60; done
run output_v17a final/v14_models --unseen-ce-override 0=models/ce_base_fr 1=models/ce_base2_fr 2=models/ce_mdeberta_fr
until grep -q V17C5_DONE v17c5.log 2>/dev/null && grep -q "scored models/ce_hard_fr" gpu_chain.log; do sleep 60; done
mkdir -p final/v17c_models && cp final/v14_models/*.txt final/v17c_models/
cp artifacts/train/q_model_v17c5.txt final/v17c_models/q_unseen.txt; cp artifacts/train/band_model_v17c5.txt final/v17c_models/band2_unseen.txt
run output_v17c final/v17c_models --unseen-ce-idx 0 1 2 3 4 \
    --unseen-ce-override 0=models/ce_base_fr 1=models/ce_base2_fr 2=models/ce_mdeberta_fr 4=models/ce_hard_fr
until grep -q GPU_CHAIN_DONE gpu_chain.log 2>/dev/null; do sleep 60; done
run output_v17b final/v14_models --unseen-use-seen \
    --unseen-ce-override 0=models/ce_base_fr 1=models/ce_base2_fr 2=models/ce_mdeberta_fr 4=models/ce_hard_fr 5=models/ce_large_fr
cp backup_pred_s2_v14/pred_s2_*.parquet artifacts/test/
