#!/usr/bin/env bash
# v4: stage-1 (hybrid + name-ambiguity model) -> cross-encoder on uncertain band -> stage-2 -> TSVs
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p artifacts/test/v3_backup
for f in artifacts/test/pred_hyb_*; do mv "$f" artifacts/test/v3_backup/ 2>/dev/null; done
$PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art artifacts/test \
    --model artifacts/train/model_hyb5.txt --hybrid --dense-k 5 --rule expected_f --out output_v4_s1 \
    --chunk-s1 100000 --resume > v4.log 2>&1 || exit 1
$PY -m er.predict_stage2 --art artifacts/test --ce-model models/ce --stage2 artifacts/train/stage2_hyb5_ce.txt \
    --out output_v4 >> v4.log 2>&1 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v4/matching_results.tsv \
    --candidate output_v4/candidate_pairs.tsv --test-dir student_resource/dataset/test >> v4.log 2>&1
mkdir -p runs && cp output_v4/matching_results.tsv runs/v4_matching_results.tsv
echo V4_DONE >> v4.log
