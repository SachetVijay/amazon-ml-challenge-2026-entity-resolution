#!/bin/bash
# download adapted models from Kaggle when the kernels finish
cd ~/amlc
set -a; . ./.env; set +a
K=.venv/bin/kaggle
get() {  # $1 kernel  $2 target dir
  until $K kernels status sachetv/$1 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 60; done
  $K kernels status sachetv/$1 2>&1 | tail -1
  rm -rf kg_$1 && mkdir kg_$1 && $K kernels output sachetv/$1 -p kg_$1 > /dev/null 2>&1
  M=$(dirname $(find kg_$1 -name config.json | head -1)); [ -n "$M" ] && rm -rf $2 && cp -r $M $2 && echo "fetched $2"
}
get amlc-adapt-mdeberta-fr models/ce_mdeberta_fr &
get amlc-adapt-large-fr models/ce_large_fr &
wait
