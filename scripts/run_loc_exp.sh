#!/usr/bin/env bash
# Location-feature experiment (after v5 inference + ensemble experiment finish). Backs up v5 train artifacts.
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; GT=student_resource/dataset/train/train_ground_truth.tsv
while tmux has-session -t v5 2>/dev/null || tmux has-session -t ens 2>/dev/null; do sleep 60; done
mkdir -p $T/v5_backup && cp $T/feats_hyb5_*.parquet $T/oof_hyb5.parquet $T/model_hyb5.txt $T/stage2_hyb5_ce.txt $T/v5_backup/
$PY -m er.add_location --art $T --tag _hyb5 || exit 1
$PY -m er.train_full --split-dir student_resource/dataset/train --prefix train --art $T --gt $GT --hybrid --dense-k 5 || exit 1
$PY -m er.exp_stage2 --art $T --gt $GT --tag _hyb5 --ce $T/ce_v5_oof.parquet --save-tag _ce_loc
echo LOC_DONE
