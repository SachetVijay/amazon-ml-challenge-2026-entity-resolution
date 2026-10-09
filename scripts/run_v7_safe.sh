#!/usr/bin/env bash
# Memory-safe v7 completion: ONE heavy CPU job at a time (+ one light GPU/network job).
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
set -a; . ./.env; set +a
# phase 1: test stage 1 (heavy) || mDeBERTa download (network)
( rm -rf models/ce_mdeberta_raw; mkdir -p models/ce_mdeberta_raw
  .venv/bin/kaggle kernels output sachetv/amlc-ce-mdeberta -p models/ce_mdeberta_raw > md_dl.log 2>&1
  M=$(dirname $(find models/ce_mdeberta_raw -name config.json | head -1)); rm -rf models/ce_mdeberta; cp -r $M models/ce_mdeberta
  echo "MD_DOWNLOADED $M" >> md_dl.log ) &
$PY -m er.predict --split-dir student_resource/dataset/test --prefix test --art $S --model final/v7_models/model.txt \
    --hybrid --dense-k 5 --rule expected_f --out output_v7_s1 --chunk-s1 100000 --resume > v7_s1.log 2>&1 || exit 1
wait
echo PHASE1_DONE
# phase 2: band specialists (CPU ~5 GB) || mDeBERTa OOF scoring (GPU, ~2 GB)
$PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_mdeberta --out $T/ce3_v7_oof.parquet > md_score.log 2>&1 &
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet --save-tag _1ce > band1.log 2>&1
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet $T/ce2_v7_oof.parquet --save-tag _2ce > band2.log 2>&1
wait
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet $T/ce2_v7_oof.parquet $T/ce3_v7_oof.parquet --save-tag _3ce > band3.log 2>&1
echo PHASE2_DONE
# phase 3: pick the best band model on OOF, score the test band with its cross-encoders, apply, deliver
CHOICE=$($PY -c "
import json
m = {k: json.load(open(f'$T/band_metrics_{k}.json'))['band'] for k in ('1ce','2ce','3ce')}
print(max(m, key=m.get)); import sys; print(m, file=sys.stderr)")
echo "band choice: $CHOICE"
cp $T/band_model_$CHOICE.txt final/v7_models/band.txt
case $CHOICE in 1ce) CES="models/ce_base";; 2ce) CES="models/ce_base models/ce_base2";; 3ce) CES="models/ce_base models/ce_base2 models/ce_mdeberta";; esac
$PY -m er.predict_stage2 --art $S --ce-model $CES --stage2 final/v7_models/stage2.txt --band-model final/v7_models/band.txt --out output_v7 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v7/matching_results.tsv \
    --candidate output_v7/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v7; mkdir -p $D && cp output_v7/matching_results.tsv $D/
echo V7_DONE
