#!/usr/bin/env bash
# Progress of scripts/ec2_run_all.sh: the status file, whether the worker is alive, and the tail of the newest log.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
cat logs/warmstart_v3_STATUS.txt 2>/dev/null || echo "(no status file yet)"
echo
if pgrep -f "ec2_run_all.sh --worker" >/dev/null 2>&1; then echo "worker: RUNNING"; else echo "worker: not running"; fi
LOG=$(ls -t logs/warmstart_v3_2*.log 2>/dev/null | head -1)
[ -n "$LOG" ] && { echo "--- tail of $LOG"; tail -n "${1:-25}" "$LOG"; }
