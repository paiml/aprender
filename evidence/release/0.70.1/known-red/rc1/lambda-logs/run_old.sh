#!/usr/bin/env bash
# Re-run each 0.70.0 lambda FAIL/REFUSE row with the v0.69.5-rc.1 binary, using the 0.70.0
# ladder script + contracts (same check). --only, no --cells (rows are printed before cells).
set -uo pipefail
D=/mnt/nvme-raid0/tmp/screen-0701
export DOGFOOD_ALLOW_UNPINNED=1 APR=/mnt/nvme-raid0/targets/screen-0695/release/apr
cd /mnt/nvme-raid0/agent-wt/publish-0700 || exit 2
for id in "$@"; do
  safe=${id//[^A-Za-z0-9._-]/_}
  echo "ROW $id START $(date -u +%FT%TZ)" >> $D/old-runs.log
  timeout 1200 choom -n 1000 -- bash scripts/model_ladder.sh --host lambda --only "$id" --out "$D/receipts-0695" > "$D/old-$safe.log" 2>&1; rc=$?
  echo "ROW $id END $(date -u +%FT%TZ) rc=$rc" >> $D/old-runs.log
done
echo ALLDONE >> $D/old-runs.log
