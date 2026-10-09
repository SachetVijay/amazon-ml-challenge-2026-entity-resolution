#!/usr/bin/env bash
# v6 = v5 + location-stripped name features (stage 1 retrained by locexp) + stage 2 retrained on a CE band
# re-scored from the new stage-1 OOF. Then test inference -> output_v6.
set -e
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
while tmux has-session -t cons 2>/dev/null; do sleep 30; done
mkdir -p final/v6_models && cp $T/model_hyb5.txt final/v6_models/model.txt
$PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_base --out $T/ce_v6_oof.parquet
$PY -m er.exp_stage2 --art $T --gt $GT --tag _hyb5 --ce $T/ce_v6_oof.parquet --save-tag _v6
cp $T/stage2_hyb5_v6.txt final/v6_models/stage2.txt
echo V6_TRAIN_DONE
mkdir -p $S/v5_backup
for f in $S/pred_hyb_* $S/ce_*.parquet $S/band_*.parquet $S/pred_s2_*; do [ -f $f ] && mv $f $S/v5_backup/; done
$PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art $S --model final/v6_models/model.txt \
    --hybrid --dense-k 5 --rule expected_f --out output_v6_s1 --chunk-s1 100000 --resume
$PY -m er.predict_stage2 --art $S --ce-model models/ce_base --stage2 final/v6_models/stage2.txt --out output_v6
python3 student_resource/utils/validate_submission.py --matching output_v6/matching_results.tsv \
    --candidate output_v6/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v6; mkdir -p $D && cp output_v6/matching_results.tsv $D/
echo V6_DONE
