#!/usr/bin/env bash
# Waits for the overnight chain to finish (or stall past 10:30 local time) and writes a plain-text summary
# to the Windows Downloads folder so the result is readable without the Mac/Claude session.
cd ~/amlc
OUT=${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/STATUS.txt
while ! grep -qE "PROBES_V11_DONE|Traceback|failed" night3.log 2>/dev/null && ! { [ $(date +%H) -ge 5 ] && [ $(date +%H) -lt 12 ] && ! tmux has-session -t night3 2>/dev/null; }; do sleep 120; done
{
  echo "AMLC overnight status — written $(date '+%Y-%m-%d %H:%M') local time (IST = local + 5:30)"
  echo
  echo "== Band-specialist validation (macro F0.5, US+India, held-out entities) =="
  for t in v9 v10a v11h v11l; do [ -f artifacts/train/band_metrics_$t.json ] && python3 -c "import json; m=json.load(open('artifacts/train/band_metrics_$t.json')); print(f'  $t: {m[\"band\"]:.5f}')"; done
  echo
  echo "== Chosen for v11 =="; grep -E "v11 choice|falling back" night3.log
  echo
  echo "== Submission files (validator) =="; grep -E "rows|PASS|FAIL" night3.log | grep -v candidate
  echo
  echo "== Files ready =="; ls -1 ${WIN_SUBMIT_DIR:-/mnt/c/Users/YOURUSER/Downloads/AMLC_Submissions}/ | grep -E "v11"
  echo
  if grep -q PROBES_V11_DONE night3.log; then
    echo "RESULT: OK. Submit v11 first (AMLC_Submissions\\v11\\matching_results.tsv)."
    echo "Then one France probe (v11fL or v11fC) — Claude will recommend which."
  else
    echo "RESULT: the overnight chain did NOT finish. Last log lines:"; tail -15 night3.log
    echo "Fallback: v9 (already submitted, 0.986000) remains the best available."
  fi
} > "$OUT" 2>&1
echo "status written"
