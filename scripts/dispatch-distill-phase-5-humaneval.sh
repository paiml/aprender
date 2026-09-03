#!/usr/bin/env bash
# SPEC-DISTILL-001 Phase 5 — HumanEval discharge.
#
# Runs `apr eval` --task humaneval on a Stage D output checkpoint. With
# PMAT-702 (#1874) in main, `eval` no longer falls back to structural
# validation with a fake pass@1=1.0 false positive on broken models —
# inference failure now returns exit code 8 with mode=inference_failed.
#
# Target: pass@1 >= 25% per SPEC-DISTILL-001 AC-DISTILL-004 (loose ship
# threshold; competitive 0.5B sits at 30-40%, upstream 7B teacher at 91%).
#
# Usage:
#   CHECKPOINT=/path/to/student-trained.apr ./scripts/dispatch-distill-phase-5-humaneval.sh
#   SAMPLES=16 ./scripts/dispatch-distill-phase-5-humaneval.sh   # pass@1@16
#   DRY_RUN=1 ./scripts/dispatch-distill-phase-5-humaneval.sh    # plan only

set -euo pipefail

GX10_HOST="${GX10_HOST:-gx10}"
GX10_USER="${GX10_USER:-noah}"
GX10_REPO_PATH="${GX10_REPO_PATH:-/home/noah/src/aprender}"
HUMANEVAL_JSONL="${HUMANEVAL_JSONL:-/home/noah/data/benchmarks/humaneval.jsonl}"
DEVICE="${DEVICE:-cuda}"
SAMPLES="${SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"

if [ -z "${CHECKPOINT:-}" ]; then
    echo "ERROR: set CHECKPOINT=/path/to/student-trained.apr" >&2
    echo "Typical value after Stage D:" >&2
    echo "  CHECKPOINT=\$HOME/runs/distill-stage-d-<run>/student-trained.apr/model.apr" >&2
    exit 2
fi

# Intentional: timestamp for result tracking -- RUN_NAME must be unique per
# dispatch so concurrent evidence directories never collide; it is a run
# identifier, not a reproducible build artifact.
RUN_NAME="distill-phase-5-humaneval-$(date +%Y%m%d-%H%M%S)"
if [ -z "${EVIDENCE_DIR:-}" ]; then
    EVIDENCE_DIR="evidence/${RUN_NAME}"
fi
DRY_RUN="${DRY_RUN:-0}"

echo "=== Phase 5 HumanEval discharge ==="
echo "  target:      ${GX10_USER}@${GX10_HOST}"
echo "  checkpoint:  ${CHECKPOINT}"
echo "  dataset:     ${HUMANEVAL_JSONL}"
echo "  device:      ${DEVICE}"
echo "  samples:     ${SAMPLES} (pass@1@${SAMPLES})"
echo "  temp/top_p:  ${TEMPERATURE} / ${TOP_P}"
echo "  evidence:    ${EVIDENCE_DIR}"
echo

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY-RUN] would dispatch; exiting before remote work."
    exit 0
fi

echo "=== remote preflight ==="
ssh "${GX10_USER}@${GX10_HOST}" bash <<REMOTE_PREFLIGHT
    set -e
    cd '${GX10_REPO_PATH}'
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "ERROR: checkpoint not found at ${CHECKPOINT}" >&2
        exit 1
    fi
    if [ ! -f "${HUMANEVAL_JSONL}" ]; then
        echo "ERROR: HumanEval JSONL not found at ${HUMANEVAL_JSONL}" >&2
        exit 1
    fi
    if [ ! -x ./target/release/apr ]; then
        echo "ERROR: ./target/release/apr not built" >&2
        exit 1
    fi
    echo "apr version: \$(./target/release/apr --version)"
    echo "preflight OK"
REMOTE_PREFLIGHT

echo
echo "=== dispatching Phase 5 evaluation on gx10 ==="
RUN_DIR_REMOTE="${GX10_RUNS_DIR:-/home/${GX10_USER}/runs}/${RUN_NAME}"
LOG_REMOTE="${RUN_DIR_REMOTE}/launch.log"
JSON_REMOTE="${RUN_DIR_REMOTE}/results.json"

ssh "${GX10_USER}@${GX10_HOST}" bash <<REMOTE_DISPATCH
    set -e
    cd '${GX10_REPO_PATH}'
    mkdir -p '${RUN_DIR_REMOTE}'
    nohup ./target/release/apr "eval" '${CHECKPOINT}' --task humaneval --data '${HUMANEVAL_JSONL}' --device '${DEVICE}' --samples '${SAMPLES}' --temperature '${TEMPERATURE}' --json > '${JSON_REMOTE}' 2> '${LOG_REMOTE}' &
    DISPATCH_PID=\$!
    disown
    echo "dispatched PID \${DISPATCH_PID}"
    sleep 5
    if ! kill -0 \${DISPATCH_PID} 2>/dev/null; then
        echo "EARLY EXIT -- log + json tail:" >&2
        tail -20 '${LOG_REMOTE}' >&2 || true
        tail -20 '${JSON_REMOTE}' >&2 || true
        exit 1
    fi
    echo "PID alive after 5 s -- evaluation running"
REMOTE_DISPATCH

mkdir -p "${EVIDENCE_DIR}"
# Telemetry: record when this dispatch actually ran, for the evidence trail --
# not a reproducible build artifact, the wall-clock time IS the datum.
DISPATCHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "${EVIDENCE_DIR}/dispatch.json" <<JSON
{
  "ticket": "SPEC-DISTILL-001 Phase 5 HumanEval discharge (PMAT-684)",
  "run_name": "${RUN_NAME}",
  "host": "${GX10_HOST}",
  "checkpoint": "${CHECKPOINT}",
  "humaneval_jsonl": "${HUMANEVAL_JSONL}",
  "device": "${DEVICE}",
  "samples": ${SAMPLES},
  "temperature": "${TEMPERATURE}",
  "top_p": "${TOP_P}",
  "remote_run_dir": "${RUN_DIR_REMOTE}",
  "remote_log": "${LOG_REMOTE}",
  "remote_results_json": "${JSON_REMOTE}",
  "dispatched_at": "${DISPATCHED_AT}",
  "pmat_702_fix_active": "if results.mode == inference_failed or pass_at_k.rate == 0.0 with non-zero exit, that is a real signal; pre-PMAT-702 the broken-model case showed pass at 1 = 1.0 false-positive"
}
JSON

echo
echo "=== dispatch complete ==="
echo "  evidence manifest:  ${EVIDENCE_DIR}/dispatch.json"
echo "  remote results:     ssh ${GX10_HOST} 'cat ${JSON_REMOTE} | jq .'"
echo "  remote log:         ssh ${GX10_HOST} 'tail -f ${LOG_REMOTE}'"
echo
echo "Estimated wall time: 164 problems × ${SAMPLES} samples on ${DEVICE}: ~5-8 h on GB10."
echo "Target: pass@1 >= 25% (AC-DISTILL-004 loose ship threshold)."
echo "Falsifier: mode=inference_failed or pass@1=0.0 with non-zero exit means re-train (Stage D failed to produce a usable model)."
