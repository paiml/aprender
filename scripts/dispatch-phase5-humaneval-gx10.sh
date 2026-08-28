#!/usr/bin/env bash
# SPEC-DISTILL-001 Phase 5 — HumanEval pass@1 discharge on gx10 Blackwell GB10.
#
# Falsifier: F-DISTILL-HUMANEVAL-001 — pass@1 ≥ threshold (default 30%)
# on the trained Phase 4 student.
#
# Prerequisites
# =============
# 1. Stage D (Phase 4) completed: student-trained.apr at MODEL_PATH below.
# 2. HumanEval JSONL pulled or pre-staged on gx10.
# 3. gx10 reachable; apr built with --features cuda,inference.
#
# Usage
# =====
#   ./scripts/dispatch-phase5-humaneval-gx10.sh                # default
#   MODEL_PATH=/path/to/student.apr ./scripts/dispatch-phase5-humaneval-gx10.sh
#   DRY_RUN=1 ./scripts/dispatch-phase5-humaneval-gx10.sh      # plan only

set -euo pipefail

# --------------------------------------------------------------------------
# Config (override via env)
# --------------------------------------------------------------------------
GX10_HOST="${GX10_HOST:-gx10}"
GX10_USER="${GX10_USER:-noah}"
GX10_REPO_PATH="${GX10_REPO_PATH:-/home/noah/src/aprender}"
# Default model path = Stage D output. Override for any other trained student.
MODEL_PATH="${MODEL_PATH:-/home/noah/runs/distill-smoke-20260520-124239/student-trained.apr}"
# HumanEval JSONL. If unset, the script does `apr pull dataset openai/humaneval`
# and finds the file in the cache.
HUMANEVAL_JSONL="${HUMANEVAL_JSONL:-}"
SAMPLES="${SAMPLES:-1}"      # pass@k value (1 = greedy decoding)
TEMPERATURE="${TEMPERATURE:-0.0}"  # 0 = greedy; 0.8 for sampled pass@k>1
THRESHOLD_PCT="${THRESHOLD_PCT:-30}"  # F-DISTILL-HUMANEVAL-001 minimum pass@1
RUN_NAME="phase5-humaneval-$(date +%Y%m%d-%H%M%S)"
EVIDENCE_DIR="${EVIDENCE_DIR:-evidence/phase5-${RUN_NAME}}"
DRY_RUN="${DRY_RUN:-0}"

# --------------------------------------------------------------------------
# Pre-flight (local)
# --------------------------------------------------------------------------
echo "=== Phase 5 HumanEval eval ==="
echo "  target:        ${GX10_USER}@${GX10_HOST}"
echo "  model:         ${MODEL_PATH}"
echo "  humaneval:     ${HUMANEVAL_JSONL:-(pull from openai/humaneval)}"
echo "  samples:       ${SAMPLES}"
echo "  temperature:   ${TEMPERATURE}"
echo "  threshold:     ${THRESHOLD_PCT}% pass@1 (F-DISTILL-HUMANEVAL-001)"
echo "  run name:      ${RUN_NAME}"
echo "  evidence:      ${EVIDENCE_DIR}"
echo

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY-RUN] would dispatch; exiting before remote work."
    exit 0
fi

# --------------------------------------------------------------------------
# Remote preflight + pull HumanEval if needed
# --------------------------------------------------------------------------
echo "=== remote preflight ==="
ssh "${GX10_USER}@${GX10_HOST}" "
    set -e
    cd '${GX10_REPO_PATH}'
    if [ ! -f '${MODEL_PATH}' ] && [ ! -f '${MODEL_PATH}/model.safetensors' ]; then
        echo 'Model not found at ${MODEL_PATH}' >&2
        exit 1
    fi
    echo 'model OK:' '${MODEL_PATH}'
"

# Resolve HumanEval JSONL on gx10 — pull if not set.
if [ -z "${HUMANEVAL_JSONL}" ]; then
    echo "=== pulling HumanEval dataset on gx10 ==="
    HUMANEVAL_JSONL=$(ssh "${GX10_USER}@${GX10_HOST}" "
        set -e
        cd '${GX10_REPO_PATH}'
        ./target/release/apr pull dataset openai/humaneval \
            -o '\$HOME/data/humaneval/' 2>&1 | tail -3 || true
        find '\$HOME/data/humaneval/' -name '*.jsonl' -o -name '*.json' | head -1
    " | grep -v '^=== ' | tail -1)
    if [ -z "${HUMANEVAL_JSONL}" ]; then
        echo "Failed to resolve HumanEval JSONL on gx10" >&2
        exit 1
    fi
    echo "  HumanEval resolved: ${HUMANEVAL_JSONL}"
fi

# --------------------------------------------------------------------------
# Dispatch HumanEval eval
# --------------------------------------------------------------------------
echo "=== dispatching HumanEval eval on gx10 ==="
RUN_DIR_REMOTE="${GX10_RUNS_DIR:-/home/${GX10_USER}/runs}/${RUN_NAME}"
LOG_REMOTE="${RUN_DIR_REMOTE}/eval.log"

ssh "${GX10_USER}@${GX10_HOST}" "
    set -e
    mkdir -p '${RUN_DIR_REMOTE}'
    cd '${GX10_REPO_PATH}'
    nohup ./target/release/apr eval '${MODEL_PATH}' \\
        --dataset humaneval \\
        --data '${HUMANEVAL_JSONL}' \\
        --device cuda \\
        --samples ${SAMPLES} \\
        --temperature ${TEMPERATURE} \\
        --json \\
        > '${LOG_REMOTE}' 2>&1 &
    DISPATCH_PID=\$!
    echo \"dispatched PID \${DISPATCH_PID}\"
    sleep 5
    if ! kill -0 \${DISPATCH_PID} 2>/dev/null; then
        echo 'EARLY EXIT - capturing tail of log:' >&2
        tail -40 '${LOG_REMOTE}' >&2 || true
        exit 1
    fi
    echo \"PID alive after 5s - eval underway\"
"

# --------------------------------------------------------------------------
# Pull evidence + dispatch manifest
# --------------------------------------------------------------------------
mkdir -p "${EVIDENCE_DIR}"
cat > "${EVIDENCE_DIR}/dispatch.json" <<JSON
{
  "ticket": "PMAT-PHASE5-HUMANEVAL",
  "phase": "SPEC-DISTILL-001 Phase 5 — HumanEval pass@1 discharge",
  "falsifier": "F-DISTILL-HUMANEVAL-001",
  "run_name": "${RUN_NAME}",
  "host": "${GX10_HOST}",
  "model_path": "${MODEL_PATH}",
  "humaneval_jsonl": "${HUMANEVAL_JSONL}",
  "samples": ${SAMPLES},
  "temperature": "${TEMPERATURE}",
  "threshold_pct": ${THRESHOLD_PCT},
  "remote_run_dir": "${RUN_DIR_REMOTE}",
  "remote_log": "${LOG_REMOTE}",
  "dispatched_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo
echo "=== dispatch complete ==="
echo "  evidence/manifest: ${EVIDENCE_DIR}/dispatch.json"
echo "  remote log:        ssh ${GX10_HOST} 'tail -f ${LOG_REMOTE}'"
echo
echo "HumanEval-164 with ${SAMPLES} sample(s) × ~5sec/problem on Blackwell ≈"
echo "$((164 * 5 / 60))min wall time. pass@${SAMPLES} verdict in ${LOG_REMOTE}'s"
echo "metrics block once eval completes."
