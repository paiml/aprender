#!/usr/bin/env bash
# Tail the gx10 Phase 3 smoke run + emit F-DISTILL-SMOKE-001 verdict on completion.
#
# Usage:
#   ./scripts/watch-distill-phase-3-gx10.sh <run-name>

set -euo pipefail

GX10_HOST="${GX10_HOST:-gx10}"
GX10_USER="${GX10_USER:-noah}"
RUN_NAME="${1:?Usage: watch-distill-phase-3-gx10.sh <run-name>}"
RUN_DIR_REMOTE="/mnt/nvme-raid0/runs/${RUN_NAME}"
LOG_REMOTE="${RUN_DIR_REMOTE}/launch.log"

echo "=== tailing ${LOG_REMOTE} on ${GX10_HOST} ==="
echo "    Ctrl-C to detach; the remote process keeps running."
echo

# Stream + extract metrics. The training loop emits initial_loss and
# final_loss as final-line markers (per pipeline.rs metrics export).
ssh "${GX10_USER}@${GX10_HOST}" "tail -F '${LOG_REMOTE}'" \
    | grep --line-buffered -E "^step|initial_loss|final_loss|panic|error|F-DISTILL" \
    || true
