#!/usr/bin/env bash
# Runs on the compute machine from ~/amlc after bi-encoder training finishes.
export PYTHONPATH=src
PY=.venv/bin/python
while tmux has-session -t enc 2>/dev/null; do sleep 30; done
if [[ $1 == cpu ]]; then
  $PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art artifacts/test \
      --model artifacts/train/model.txt --out output_v2 --chunk-s1 40000 > v2.log 2>&1
  python3 student_resource/utils/validate_submission.py --matching output_v2/matching_results.tsv \
      --candidate output_v2/candidate_pairs.tsv --test-dir student_resource/dataset/test >> v2.log 2>&1
  mkdir -p runs && cp output_v2/matching_results.tsv runs/v2_matching_results.tsv
else
  test -d models/e5er || { echo "no model"; exit 1; }
  for art in artifacts/test artifacts/train; do
    $PY -m er.biencoder encode --art $art --model models/e5er || exit 1
    $PY -m er.biencoder search --art $art --k-other 10 --k-s1 10 || exit 1
  done
  $PY -m er.biencoder recall --art artifacts/train --gt student_resource/dataset/train/train_ground_truth.tsv
fi
echo CHAIN_DONE
