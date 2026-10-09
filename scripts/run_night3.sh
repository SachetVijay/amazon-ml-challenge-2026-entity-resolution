#!/usr/bin/env bash
# Morning build of v11: best band specialist on validation among {v9, v10a, +hard CE, +hard+e5-large}.
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; S=artifacts/test; GT=student_resource/dataset/train/train_ground_truth.tsv
set -a; . ./.env; set +a
K=.venv/bin/kaggle
BASE_OOF="$T/ce0_v9_oof.parquet $T/ce1_v9_oof.parquet $T/ce2_v9_oof.parquet $T/ce3_v9_oof.parquet"
# 1. hard CE -> 5-CE band (bucket 0 only: hard CE trained on buckets 4-5)
while ! grep -qE "HARD_DONE|Traceback" hard.log 2>/dev/null; do sleep 60; done
grep -q HARD_DONE hard.log && $PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v9 --ce $BASE_OOF $T/ce5_v9_oof.parquet --save-tag _v11h > v11h.log 2>&1
grep -E "band AUC|F0.5 using" v11h.log
# 2. e5-large (wait until done, give up at 04:00 local = 09:30 IST)
until $K kernels status sachetv/amlc-ce-e5large 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL" || [ $(date +%H) -ge 4 -a $(date +%H) -lt 12 ]; do sleep 120; done
LARGE=no
if $K kernels status sachetv/amlc-ce-e5large 2>&1 | grep -q COMPLETE; then
  $K kernels push -p kaggle/score_large --accelerator NvidiaTeslaT4 2>&1 | tail -1          # test band on Kaggle GPUs
  rm -rf models/ce_large_raw && mkdir -p models/ce_large_raw && $K kernels output sachetv/amlc-ce-e5large -p models/ce_large_raw > /dev/null 2>&1
  M=$(dirname $(find models/ce_large_raw -name config.json | head -1)); rm -rf models/ce_large && cp -r $M models/ce_large
  $PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_large --out $T/ce4_v9_oof.parquet \
    && $PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v9 --ce $BASE_OOF $T/ce5_v9_oof.parquet $T/ce4_v9_oof.parquet --save-tag _v11l > v11l.log 2>&1 \
    && LARGE=yes
  grep -E "band AUC|F0.5 using" v11l.log
fi
# 3. pick on validation
BEST=$($PY -c "
import json, os
c = {k: json.load(open(f'$T/band_metrics_{k}.json'))['band'] for k in ('v9', 'v10a', 'v11h', 'v11l') if os.path.exists(f'$T/band_metrics_{k}.json')}
import sys; print(c, file=sys.stderr); print(max(c, key=c.get))")
echo "v11 choice: $BEST"
case $BEST in
  v9)   CES="models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm"; MAP="0 1 2 3";;
  v10a) CES="models/ce_base models/ce_mdeberta models/ce_norm"; MAP="0 2 3";;
  v11h) CES="models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm models/ce_hard"; MAP="0 1 2 3";;
  v11l) CES="models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm models/ce_hard models/ce_large"; MAP="0 1 2 3";;
esac
# 4. assemble cached test CE scores (v9 order 0 base,1 base2,2 mdeberta,3 norm) in the chosen order
mkdir -p $S/v9_ce && cp -n $S/ce[0-3]_*.parquet $S/v9_ce/ 2>/dev/null; rm -f $S/ce*_*.parquet
for c in US France India; do j=0; for m in $MAP; do cp $S/v9_ce/ce${m}_$c.parquet $S/ce${j}_$c.parquet; j=$((j+1)); done; done
if [ "$BEST" = "v11l" ]; then     # e5-large test scores from Kaggle (index 5)
  until $K kernels status sachetv/amlc-score-large 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 60; done
  rm -rf kg_large && mkdir kg_large && $K kernels output sachetv/amlc-score-large -p kg_large > /dev/null 2>&1
  $PY -c "
import pandas as pd; d = pd.read_parquet('kg_large/ce_large_test.parquet')
for c, g in d.groupby('country'): g[['a','b','ce']].to_parquet(f'$S/ce5_{c}.parquet')
print('large test scores', len(d))" || { echo "kaggle large scoring failed -> falling back to v11h"; BEST=v11h; CES="models/ce_base models/ce_base2 models/ce_mdeberta models/ce_norm models/ce_hard"; }
fi
mkdir -p final/v11_models && cp final/v9_models/model.txt final/v9_models/stage2.txt final/v11_models/ && cp $T/band_model_$BEST.txt final/v11_models/band.txt
$PY -m er.predict_stage2 --art $S --ce-model $CES --stage2 final/v11_models/stage2.txt --band-model final/v11_models/band.txt --out output_v11 || exit 1
python3 student_resource/utils/validate_submission.py --matching output_v11/matching_results.tsv --candidate output_v11/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v11; mkdir -p $D && cp output_v11/matching_results.tsv $D/
echo V11_DONE
# 5. France-only probes on v11
for v in "fL 2.0" "fC 0.5"; do set -- $v
  $PY -m er.redecide --art $S --r 1.0 --r-country France=$2 --out output_v11$1 | tail -1
  D=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/v11$1; mkdir -p $D; cp output_v11$1/matching_results.tsv $D/
  python3 student_resource/utils/validate_submission.py --matching $D/matching_results.tsv --test-dir student_resource/dataset/test --check-ids 2>&1 | grep -E "^PASS|^FAIL"
done
echo PROBES_V11_DONE
