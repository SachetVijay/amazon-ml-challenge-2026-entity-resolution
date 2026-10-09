#!/usr/bin/env bash
export PYTHONPATH=src
PY=.venv/bin/python
$PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art artifacts/test \
    --model artifacts/train/model_hyb5.txt --hybrid --dense-k 5 --rule expected_f --out output_v3 \
    --chunk-s1 100000 --resume > v3_pred.log 2>&1 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v3/matching_results.tsv \
    --candidate output_v3/candidate_pairs.tsv --test-dir student_resource/dataset/test >> v3_pred.log 2>&1
mkdir -p runs && cp output_v3/matching_results.tsv runs/v3_matching_results.tsv
echo V3_DONE >> v3_pred.log
