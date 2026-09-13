#!/usr/bin/env bash
# SPEC-DISTILL-001 Phase 4 Stage D -- full production distillation dispatch.
#
# Lessons baked in (PMAT-701 family cascade post-mortem, 2026-05-22):
#   - cuBLAS backend default (PMAT-704 / #1879). Realizar Q4K path is opt-in only.
#   - PMAT-705 ProgressCallback wired by default; per-step loss visible via
#     APR_DISTILL_LOG_EVERY (default 50 for long runs).
#   - PMAT-699 P0 checkpointing (every 5000 steps) -- survives kill/crash.
#   - PMAT-703 vocab alignment auto-applies when teacher.vocab > student.vocab.
#   - Disk preflight: require >= 15 GB free (Stage D writes ~12 GB of checkpoints).
#   - Teacher / student inputs from APR-stamped checkpoints (apr-leaderboard cache
#     for known-good metadata) -- not the dispatch-script-imported GGUF that lacks
#     stamped metadata (PMAT-704 cascade incident).
#
# Usage:
#   ./scripts/dispatch-distill-stage-d.sh
#   STEPS=10000 ./scripts/dispatch-distill-stage-d.sh   # shorter run for re-validation
#   APR_DISTILL_MAX_STEPS=10 ./scripts/dispatch-distill-stage-d.sh  # ~60s smoke (PMAT-706, no output written)
#   DRY_RUN=1   ./scripts/dispatch-distill-stage-d.sh   # plan only

set -euo pipefail

# ============================================================================
# Config (override via env)
# ============================================================================
GX10_HOST="${GX10_HOST:-gx10}"
GX10_USER="${GX10_USER:-noah}"
GX10_REPO_PATH="${GX10_REPO_PATH:-/home/noah/src/aprender}"
GX10_RUN_PREFIX="${GX10_RUN_PREFIX:-/home/noah/runs}"

# Teacher: MODEL-1 7B (paiml/qwen2.5-coder-7b-apache-q4k-v1 lineage).
# The apr-leaderboard checkpoint has the FULL stamped metadata that the cuda
# distill backend's TransformerConfig::from_apr_metadata requires; the
# dispatch script's `apr import --preserve-q4k` path produces a checkpoint
# WITHOUT stamped metadata, which fails validation. Operators with a stamped
# 7B teacher elsewhere can override TEACHER_APR.
TEACHER_APR="${TEACHER_APR:-/home/noah/src/apr-leaderboard/checkpoints/qwen2.5-coder-7b-instruct-q4k.apr}"
TEACHER_TOKENIZER="${TEACHER_TOKENIZER:-/home/noah/src/apr-leaderboard/checkpoints/qwen2.5-coder-7b-instruct-q4k.tokenizer.json}"

# Student: 0.5B Qwen2.5-Coder-Instruct (existing staging from prior 10K run).
# Operators can stage a fresh student dir if needed.
STUDENT_DIR_SRC="${STUDENT_DIR_SRC:-/home/noah/runs/distill-smoke-20260521-233519/student}"

# Hyperparameters per SPEC-DISTILL-001 Phase 4 §4 (no §86/§87 overrides):
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"      # pipeline default; 1000/32 ~ 31 steps/epoch
LR="${LR:-1.5e-5}"
T="${T:-4.0}"                       # KD temperature
ALPHA="${ALPHA:-0.3}"               # CE weight (per Hinton 2015 -- KD signal dominates)
DATASET_DIR="${DATASET_DIR:-}"      # set to a real-corpus shard dir to disable synthetic

# Observability (PMAT-705):
APR_DISTILL_LOG_EVERY="${APR_DISTILL_LOG_EVERY:-50}"   # log loss every N steps; 0 to silence

# Checkpoint cadence (PMAT-699):
APR_DISTILL_CHECKPOINT_EVERY="${APR_DISTILL_CHECKPOINT_EVERY:-5000}"

# Backend selection (PMAT-704):
APR_DISTILL_TEACHER_BACKEND="${APR_DISTILL_TEACHER_BACKEND:-auto}"  # auto, cudatrainer, realizar-q4k

# Smoke-validation early-break (PMAT-706): run at most N steps, print a
# [SMOKE] loss/throughput/projection summary, and skip the final export.
# Empty = full run (no change). Forwarded into the remote `env` invocation so
# `APR_DISTILL_MAX_STEPS=10 ./scripts/dispatch-distill-stage-d.sh` validates the
# whole cascade in ~60s before committing to a 30-50h Stage D run.
APR_DISTILL_MAX_STEPS="${APR_DISTILL_MAX_STEPS:-}"

# Disk preflight threshold (GB). Stage D 50K writes ~12 GB checkpoints + 2 GB final.
DISK_FREE_REQUIRED_GB="${DISK_FREE_REQUIRED_GB:-15}"

RUN_NAME="distill-stage-d-$(date +%Y%m%d-%H%M%S)"
if [ -z "${EVIDENCE_DIR:-}" ]; then
    EVIDENCE_DIR="evidence/distill-stage-d-${RUN_NAME}"
fi
DRY_RUN="${DRY_RUN:-0}"

# ============================================================================
# Pre-flight (local)
# ============================================================================
echo "=== Stage D dispatch ==="
echo "  target:        ${GX10_USER}@${GX10_HOST}"
echo "  teacher:       ${TEACHER_APR}"
echo "  student src:   ${STUDENT_DIR_SRC}"
echo "  steps:         ${STEPS}"
echo "  batch_size:    ${BATCH_SIZE}"
echo "  LR:            ${LR}"
echo "  T (KD temp):   ${T}"
echo "  alpha (CE wt): ${ALPHA}"
echo "  log_every:     ${APR_DISTILL_LOG_EVERY}"
echo "  ckpt_every:    ${APR_DISTILL_CHECKPOINT_EVERY}"
echo "  backend:       ${APR_DISTILL_TEACHER_BACKEND}"
echo "  max_steps:     ${APR_DISTILL_MAX_STEPS:-<full run>}"
echo "  dataset:       ${DATASET_DIR:-(synthetic -- operator must set DATASET_DIR for real Phase 4)}"
echo "  run name:      ${RUN_NAME}"
echo "  evidence:      ${EVIDENCE_DIR}"
echo

STEPS_PER_EPOCH=$(( 1000 / BATCH_SIZE ))
if [ "${STEPS_PER_EPOCH}" -lt 1 ]; then STEPS_PER_EPOCH=1; fi
EPOCHS=$(( (STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
echo "  derived:       ${EPOCHS} epochs (~ ${STEPS} steps at ${STEPS_PER_EPOCH} steps/epoch)"
echo

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY-RUN] would dispatch; exiting before remote work."
    exit 0
fi

# ============================================================================
# Remote preflight: disk + repo + teacher + student
# ============================================================================
echo "=== remote preflight ==="
ssh "${GX10_USER}@${GX10_HOST}" bash <<REMOTE_PREFLIGHT
    set -e
    cd '${GX10_REPO_PATH}'

    FREE_GB=\$(df -BG "\$HOME" | awk 'NR==2 {gsub("G","",\$4); print \$4}')
    echo "disk free on \$HOME: \${FREE_GB} GB (require >= ${DISK_FREE_REQUIRED_GB})"
    if [ "\${FREE_GB}" -lt "${DISK_FREE_REQUIRED_GB}" ]; then
        echo "ERROR: insufficient disk space" >&2
        echo "  cleanup candidates:" >&2
        du -h --max-depth=1 "\$HOME/runs" 2>/dev/null | sort -hr | head -10 >&2
        exit 1
    fi

    if [ ! -f "${TEACHER_APR}" ]; then
        echo "ERROR: teacher .apr not found at ${TEACHER_APR}" >&2
        exit 1
    fi
    if [ ! -f "${TEACHER_TOKENIZER}" ]; then
        echo "ERROR: teacher tokenizer.json not found at ${TEACHER_TOKENIZER}" >&2
        exit 1
    fi

    if [ ! -d "${STUDENT_DIR_SRC}" ]; then
        echo "ERROR: student dir not found at ${STUDENT_DIR_SRC}" >&2
        exit 1
    fi
    if [ ! -f "${STUDENT_DIR_SRC}/source.safetensors" ] && [ ! -L "${STUDENT_DIR_SRC}/source.safetensors" ]; then
        echo "ERROR: student dir missing source.safetensors symlink" >&2
        echo "  CudaStudentProvider::for_training will fail without it." >&2
        exit 1
    fi

    if [ ! -x ./target/release/apr ]; then
        echo "ERROR: ./target/release/apr not built -- run cargo build --release --features cuda -p apr-cli --bin apr" >&2
        exit 1
    fi
    echo "apr version: \$(./target/release/apr --version)"
    echo "preflight OK"
REMOTE_PREFLIGHT

# ============================================================================
# Dispatch
# ============================================================================
echo "=== dispatching Stage D run on gx10 ==="
RUN_DIR_REMOTE="${GX10_RUN_PREFIX}/${RUN_NAME}"
LOG_REMOTE="${RUN_DIR_REMOTE}/launch.log"

ssh "${GX10_USER}@${GX10_HOST}" bash <<REMOTE_DISPATCH
    set -e
    cd '${GX10_REPO_PATH}'

    mkdir -p '${RUN_DIR_REMOTE}/teacher' '${RUN_DIR_REMOTE}/student'

    ln -sf '${TEACHER_APR}'       '${RUN_DIR_REMOTE}/teacher/model.apr'
    ln -sf '${TEACHER_TOKENIZER}' '${RUN_DIR_REMOTE}/teacher/tokenizer.json'

    for f in model.apr tokenizer.json config.json tokenizer_config.json source.safetensors; do
        if [ -e '${STUDENT_DIR_SRC}/'"\$f" ]; then
            ln -sf '${STUDENT_DIR_SRC}/'"\$f" '${RUN_DIR_REMOTE}/student/'"\$f"
        fi
    done

    DATASET_FLAG=''
    if [ -n '${DATASET_DIR}' ]; then
        if [ ! -d '${DATASET_DIR}' ]; then
            echo "ERROR: DATASET_DIR=${DATASET_DIR} does not exist" >&2
            exit 1
        fi
        DATASET_FLAG='--dataset ${DATASET_DIR}'
    fi

    nohup env APR_DISTILL_LOG_EVERY=${APR_DISTILL_LOG_EVERY} APR_DISTILL_CHECKPOINT_EVERY=${APR_DISTILL_CHECKPOINT_EVERY} APR_DISTILL_TEACHER_BACKEND=${APR_DISTILL_TEACHER_BACKEND} APR_DISTILL_MAX_STEPS=${APR_DISTILL_MAX_STEPS} ./target/release/apr distill '${RUN_DIR_REMOTE}/teacher/model.apr' --student '${RUN_DIR_REMOTE}/student/model.apr' --epochs ${EPOCHS} --temperature ${T} --alpha ${ALPHA} --backend cuda \$DATASET_FLAG --output '${RUN_DIR_REMOTE}/student-trained.apr' > '${LOG_REMOTE}' 2>&1 &
    DISPATCH_PID=\$!
    disown
    echo "dispatched PID \${DISPATCH_PID}"

    sleep 10
    if ! kill -0 \${DISPATCH_PID} 2>/dev/null; then
        echo "EARLY EXIT -- tail of log:" >&2
        tail -40 '${LOG_REMOTE}' >&2 || true
        exit 1
    fi
    echo "PID alive after 10 s -- Stage D training underway"
    head -30 '${LOG_REMOTE}'
REMOTE_DISPATCH

mkdir -p "${EVIDENCE_DIR}"
# DET002: reproducible under SOURCE_DATE_EPOCH (falls back to the real wall
# clock when it's unset -- this is an audit timestamp of when the run was
# actually dispatched, so it must stay real time by default).
DISPATCHED_AT="$(date -u -d "@${SOURCE_DATE_EPOCH:-$(date +%s)}" +%Y-%m-%dT%H:%M:%SZ)"
cat > "${EVIDENCE_DIR}/dispatch.json" <<JSON
{
  "ticket": "SPEC-DISTILL-001 Phase 4 Stage D -- post-PMAT-701 cascade",
  "run_name": "${RUN_NAME}",
  "host": "${GX10_HOST}",
  "teacher_apr": "${TEACHER_APR}",
  "student_dir_src": "${STUDENT_DIR_SRC}",
  "steps": ${STEPS},
  "epochs": ${EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "learning_rate": "${LR}",
  "kd_temperature": "${T}",
  "kd_alpha": "${ALPHA}",
  "log_every": ${APR_DISTILL_LOG_EVERY},
  "checkpoint_every": ${APR_DISTILL_CHECKPOINT_EVERY},
  "backend": "${APR_DISTILL_TEACHER_BACKEND}",
  "max_steps": "${APR_DISTILL_MAX_STEPS}",
  "dataset_dir": "${DATASET_DIR}",
  "remote_run_dir": "${RUN_DIR_REMOTE}",
  "remote_log": "${LOG_REMOTE}",
  "dispatched_at": "${DISPATCHED_AT}"
}
JSON

echo
echo "=== dispatch complete ==="
echo "  evidence manifest:  ${EVIDENCE_DIR}/dispatch.json"
echo "  remote log:         ssh ${GX10_HOST} 'tail -f ${LOG_REMOTE}'"
echo "  monitor progress:   ssh ${GX10_HOST} 'grep -E \"Step|Epoch\" ${LOG_REMOTE} | tail -20'"
echo
echo "Estimated wall time: ~${EPOCHS} epochs × ~1-2 s/step with cuBLAS = ~$(( EPOCHS * STEPS_PER_EPOCH * 2 / 60 )) minutes lower bound."
echo "Real wall time depends on dataset I/O + checkpoint write latency. Monitor via the log tail above."
