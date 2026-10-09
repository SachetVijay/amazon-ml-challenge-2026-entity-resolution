#!/bin/bash
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
until grep -q FT0_DONE adapt2.done 2>/dev/null; do sleep 30; done
nice -n 5 $PY src/er/kaggle_train_ce.py --pairs kaggle_frad/pairs.parquet --base models/ce_hard --out models/ce_hard_fr \
    --bs 32 --lr 1e-5 --epochs 1 --max-hours 1.1 > ft4.log 2>&1
echo FT4_DONE >> adapt2.done
