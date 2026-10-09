#!/usr/bin/env bash
# Exit non-zero if someone else is using the GPU (use before launching on the shared compute machine).
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ $n -gt 0 || $used -gt 1500 ]]; then echo "GPU busy ($n procs, ${used}MiB) - not launching"; exit 1; fi
echo "GPU free"
