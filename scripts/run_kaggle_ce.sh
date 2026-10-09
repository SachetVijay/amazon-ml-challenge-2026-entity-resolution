#!/usr/bin/env bash
# Build bucket 6-7 cross-encoder pairs, upload as a PRIVATE Kaggle dataset, launch mDeBERTa training on Kaggle GPU.
set -e
cd ~/amlc
set -a; . ./.env; set +a          # KAGGLE_API_TOKEN (never printed)
K=.venv/bin/kaggle
while ! grep -q "V6_DONE" v6.log 2>/dev/null; do sleep 60; done   # wait for RAM (all of v6)
PYTHONPATH=src .venv/bin/python -m er.export_ce_pairs --art artifacts/train \
    --gt student_resource/dataset/train/train_ground_truth.tsv --buckets 6 7 --max-pairs 1200000 --out kaggle_ce/pairs.parquet
cat > kaggle_ce/dataset-metadata.json <<EOM
{"title": "amlc-ce-pairs-b67", "id": "sachetv/amlc-ce-pairs-b67", "licenses": [{"name": "CC0-1.0"}]}
EOM
$K datasets create -p kaggle_ce 2>&1 | tail -2                       # private by default
for i in $(seq 1 60); do $K datasets status sachetv/amlc-ce-pairs-b67 2>&1 | grep -q ready && break; sleep 20; done
$K datasets status sachetv/amlc-ce-pairs-b67
$K kernels push -p kaggle/ce_train --accelerator NvidiaTeslaT4 2>&1 | tail -1
echo KAGGLE_LAUNCHED
