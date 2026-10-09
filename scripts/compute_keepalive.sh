#!/usr/bin/env bash
# WSL on the compute machine shuts down when the last ssh session closes, killing tmux jobs.
# This holds one idle session open (and keeps the Mac awake) so jobs survive. Ctrl-C to stop.
# Nothing is changed on the compute machine. If the Mac sleeps / network drops, compute machine jobs die -> use AWS for overnight runs.
while true; do
  caffeinate -i ssh -o RemoteCommand=none -o RequestTTY=no -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
    "${COMPUTE_HOST:-compute-host}" "wsl -d ${COMPUTE_DISTRO:-Ubuntu-24.04} -- sleep infinity"
  echo "$(date) keepalive dropped, reconnecting in 10s"; sleep 10
done
