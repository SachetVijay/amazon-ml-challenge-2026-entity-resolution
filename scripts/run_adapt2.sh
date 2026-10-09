#!/bin/bash
# v17 prep: French adaptation datasets for ce0 (e5-base) and ce2 (mDeBERTa), then fine-tune ce0 on the local GPU.
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
for jm in "ce_base adapt_fr_ce0" "ce_mdeberta adapt_fr_ce2"; do set -- $jm
  [ -f $2/pairs.parquet ] || nice -n 5 $PY -m er.adapt_unseen_ce --art artifacts/test --country France --pred-s2 backup_pred_s2_v16 \
      --ce-model models/$1 --us-pairs kaggle_hard/hard_b45.parquet --out $2 > $2.log 2>&1 || exit 1
done
echo DATA_DONE > adapt2.done
nice -n 5 $PY src/er/kaggle_train_ce.py --pairs adapt_fr_ce0/pairs.parquet --base models/ce_base --out models/ce_base_fr \
    --bs 32 --lr 1e-5 --epochs 1 --max-hours 1.2 > ft0.log 2>&1
echo FT0_DONE >> adapt2.done
