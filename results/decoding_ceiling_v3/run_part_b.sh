#!/usr/bin/env bash
# Part B chain: waits for the 48 evaluation decoder outputs and the unrestricted subset solves, then runs the search
# inside DANIEL's rule and the restricted subset solves (docs/decoding_ceiling_protocol_v3.md, amendment).
cd "$(dirname "$0")/../.." || exit 2
until [ "$(ls results/decoding_ceiling_v3/evaluation/*.npz 2>/dev/null | wc -l)" -ge 48 ] && \
      [ "$(ls results/decoding_ceiling_v3/nondelay_cpsat/subset/unrestricted_*.json 2>/dev/null | wc -l)" -ge 20 ] && \
      ! pgrep -f "nondelay_cpsat_v3.py bounds --which unrestricted" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] inputs ready"
.venv/bin/python tools/nondelay_cpsat_v3.py search || exit 1
.venv/bin/python tools/nondelay_cpsat_v3.py bounds --which restricted --limit 60 --workers 6 || exit 1
echo "[$(date +%H:%M:%S)] PART B DONE"
