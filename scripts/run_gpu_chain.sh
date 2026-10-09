#!/bin/bash
# Serialized GPU work for v17 (8 GB card: never train and score at once).
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
sc() {  # $1 j  $2 model dir -> cache file predict_stage2 uses for --unseen-ce-override
  out=artifacts/test/ce$1u_$(basename $2)_France.parquet
  [ -f $out ] || $PY -c "
from types import SimpleNamespace; from er.cross_encoder import score
score(SimpleNamespace(pairs='artifacts/test/band_France.parquet', band=None, art='artifacts/test', model='$2', out='$out', max_len=128, view='raw'))" >> gpu_chain.log 2>&1
  echo "scored $2 -> $out" >> gpu_chain.log
}
until grep -q FT0_DONE adapt2.done 2>/dev/null; do sleep 30; done
sc 1 models/ce_base2_fr
sc 0 models/ce_base_fr
nice -n 5 $PY src/er/kaggle_train_ce.py --pairs kaggle_frad/pairs.parquet --base models/ce_hard --out models/ce_hard_fr \
    --bs 32 --lr 1e-5 --epochs 1 --max-hours 1.1 > ft4.log 2>&1
echo FT4_DONE >> adapt2.done
sc 4 models/ce_hard_fr
until [ -f models/ce_mdeberta_fr/config.json ]; do sleep 60; done; sc 2 models/ce_mdeberta_fr
until [ -f models/ce_large_fr/config.json ]; do sleep 60; done; sc 5 models/ce_large_fr
echo GPU_CHAIN_DONE >> gpu_chain.log
