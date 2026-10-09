#!/usr/bin/env bash
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train; GT=student_resource/dataset/train/train_ground_truth.tsv
set -a; . ./.env; set +a
rm -rf models/ce_mdeberta_raw; mkdir -p models/ce_mdeberta_raw
.venv/bin/kaggle kernels output sachetv/amlc-ce-mdeberta -p models/ce_mdeberta_raw 2>&1 | tail -1
python3 -c "import json,glob; L=json.load(open(glob.glob('models/ce_mdeberta_raw/*.log')[0])); print('\n'.join(e['data'].strip() for e in L if e['stream_name']=='stdout')[-600:])"
M=$(dirname $(find models/ce_mdeberta_raw -name config.json | head -1)); echo "model dir: $M"
rm -rf models/ce_mdeberta && cp -r $M models/ce_mdeberta
$PY -m er.cross_encoder score --art $T --pairs $T/oof_hyb5.parquet --band 0.005 0.995 --model models/ce_mdeberta --out $T/ce3_v7_oof.parquet || exit 1
while ! grep -q BANDS_DONE band2.log 2>/dev/null; do sleep 60; done
$PY -m er.exp_band --art $T --gt $GT --tag _hyb5 --s2-tag _v7 --ce $T/ce_v7_oof.parquet $T/ce2_v7_oof.parquet $T/ce3_v7_oof.parquet --save-tag _3ce
echo MD_DONE
