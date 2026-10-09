#!/usr/bin/env bash
# v3: hybrid (sparse top-5 ∪ dense top-5) full-density training -> test prediction -> validation
export PYTHONPATH=src
PY=.venv/bin/python
$PY -m er.train_full --split-dir student_resource/dataset/train --prefix train --art artifacts/train \
    --gt student_resource/dataset/train/train_ground_truth.tsv --hybrid --dense-k 5 > v3_train.log 2>&1 || exit 1
$PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art artifacts/test \
    --model artifacts/train/model_hyb5.txt --hybrid --dense-k 5 --rule expected_f --out output_v3 \
    --chunk-s1 40000 > v3_pred.log 2>&1 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v3/matching_results.tsv \
    --candidate output_v3/candidate_pairs.tsv --test-dir student_resource/dataset/test >> v3_pred.log 2>&1
mkdir -p runs && cp output_v3/matching_results.tsv runs/v3_matching_results.tsv
echo V3_DONE >> v3_pred.log
