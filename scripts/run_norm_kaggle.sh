#!/usr/bin/env bash
# Normalised-text cross-encoder: export pairs (buckets 2-3, norm view) during v7 phase 3, upload, train on Kaggle.
set -e
cd ~/amlc
set -a; . ./.env; set +a
K=.venv/bin/kaggle
while ! grep -q PHASE2_DONE v7safe.log 2>/dev/null; do grep -q Traceback v7safe.log 2>/dev/null && exit 1; sleep 60; done
mkdir -p kaggle_norm
PYTHONPATH=src .venv/bin/python -m er.export_ce_pairs --art artifacts/train \
    --gt student_resource/dataset/train/train_ground_truth.tsv --buckets 2 3 --max-pairs 1200000 --view norm \
    --out kaggle_norm/pairs.parquet
python3 -c "import pandas as pd; d=pd.read_parquet('kaggle_norm/pairs.parquet'); print(d.head(3).to_string())"
cat > kaggle_norm/dataset-metadata.json <<EOM
{"title": "amlc-ce-pairs-norm", "id": "sachetv/amlc-ce-pairs-norm", "licenses": [{"name": "CC0-1.0"}]}
EOM
$K datasets create -p kaggle_norm 2>&1 | tail -1
for i in $(seq 1 60); do $K datasets status sachetv/amlc-ce-pairs-norm 2>&1 | grep -q ready && break; sleep 20; done
$K kernels push -p kaggle/ce_norm --accelerator NvidiaTeslaT4 2>&1 | tail -1
echo NORM_LAUNCHED
