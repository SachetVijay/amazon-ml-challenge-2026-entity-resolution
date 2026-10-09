#!/usr/bin/env bash
# Push code + config (+ data unless --no-data) to a remote.  Pull back predictions with --pull.
#   scripts/sync.sh compute            # push to compute machine (WSL ~/amlc)
#   scripts/sync.sh aws              # push to the 'amlc-aws' ssh host (~/amlc)
#   scripts/sync.sh aws --pull       # fetch artifacts/oof, artifacts/test, submissions back
set -euo pipefail
TARGET=$1; shift || true
cd "$(dirname "$0")/.."
if [[ $TARGET == compute ]]; then
  SSH=(-e "ssh -o RemoteCommand=none -o RequestTTY=no")
  RP=(--rsync-path="wsl -d ${COMPUTE_DISTRO:-Ubuntu-24.04} -- rsync")
  DEST="${COMPUTE_HOST:-compute-host}:~/amlc"
else
  SSH=(); RP=(); DEST="${AWS_HOST:-amlc-aws}:amlc"
fi
if [[ ${1:-} == --pull ]]; then
  for d in artifacts/oof artifacts/test artifacts/parts submissions; do
    mkdir -p "$d"; rsync -az "${SSH[@]}" "${RP[@]}" "$DEST/$d/" "$d/" || true
  done
  exit 0
fi
EXCL=(--exclude .venv --exclude .git --exclude artifacts --exclude images --exclude __pycache__
      --exclude .env --exclude "*.pem" --exclude kaggle.json --exclude submissions --exclude models)  # never ship secrets
[[ ${1:-} == --no-data ]] && EXCL+=(--exclude data)
rsync -az "${SSH[@]}" "${RP[@]}" "${EXCL[@]}" ./ "$DEST/"
echo "synced -> $DEST"
