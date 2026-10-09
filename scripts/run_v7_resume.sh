#!/usr/bin/env bash
# Resume v7 after the WSL restart. Parallel: test stage 1 (CPU) | band specialists (CPU) | mDeBERTa (GPU).
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
tmux new -d -s s1 "$PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art $S --model final/v7_models/model.txt --hybrid --dense-k 5 --rule expected_f --out output_v7_s1 --chunk-s1 100000 --resume > v7_s1.log 2>&1; echo S1_DONE >> v7_s1.log"
tmux new -d -s bands "$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet --save-tag _1ce > band1.log 2>&1; $PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet $T/ce2_v7_oof.parquet --save-tag _2ce > band2.log 2>&1; echo BANDS_DONE >> band2.log"
tmux new -d -s md "bash scripts/run_mdeberta.sh > md.log 2>&1"
