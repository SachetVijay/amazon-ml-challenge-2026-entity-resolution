#!/bin/bash
# v13 prep: cross-encoder logits for the rival band pairs (rival S1s outside bucket 0, 0.005 < p1 < 0.995).
cd ~/amlc
export PYTHONPATH=src
PY=.venv/bin/python
T=artifacts/train
$PY -c "
import pandas as pd
d = pd.concat([pd.read_parquet(f'$T/comp_fd_{c}.parquet') for c in ('US', 'India')], ignore_index=True)
d = d[(d.p > 0.005) & (d.p < 0.995)][['a', 'b', 'p']].reset_index(drop=True)
d.to_parquet('$T/rival_band.parquet'); print('rival band pairs', len(d))"
# feature index j follows the test order: 0 base, 1 base2, 2 mdeberta, 3 norm, 4 hard, 5 large
for jm in "4 ce_hard raw" "1 ce_base2 raw" "0 ce_base raw" "3 ce_norm norm" "2 ce_mdeberta raw" "5 ce_large raw"; do set -- $jm
  [ -f $T/rce$1.parquet ] && continue
  $PY -m er.cross_encoder score --art $T --pairs $T/rival_band.parquet --model models/$2 --view $3 --out $T/rce$1.parquet || exit 1
done
echo RCE_DONE
