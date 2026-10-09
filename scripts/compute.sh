#!/usr/bin/env bash
# Run a command inside the compute machine's WSL Ubuntu, from ~/amlc.
#   scripts/compute.sh 'nvidia-smi'
#   scripts/compute.sh 'tmux new -d -s dl "uv run python -m amlc.download_images --split train"'
# The ssh alias forces `wsl ... --cd ~` as RemoteCommand, so we override it and feed bash via stdin.
set -euo pipefail
HOST=${COMPUTE_HOST:-compute-host}
DISTRO=${COMPUTE_DISTRO:-Ubuntu-24.04}
{ echo 'export PATH="$HOME/.local/bin:$PATH"; mkdir -p ~/amlc; cd ~/amlc'; echo "$*"; } |
  ssh -o RemoteCommand=none -o RequestTTY=no "$HOST" "wsl -d $DISTRO -- bash -s"
