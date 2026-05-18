#!/usr/bin/env bash
# SPEC-DISTILL-001 Phase 3 - 500-step E2E smoke run on gx10 (Blackwell GB10).
#
# Falsifier: F-DISTILL-SMOKE-001 - val_loss at step 500 < step 0.
#
# Prerequisites
# =============
# 1. PRs #1787, #1788, #1791, #1792, #1793 merged to main (Phase 1b → 2d).
# 2. gx10 has a recent aprender checkout. The script pulls main + builds
#    --features cuda fresh.
# 3. MODEL-1 teacher is cached on gx10 at
#    ~/.cache/huggingface/hub/models--paiml--qwen2.5-coder-7b-apache-q4k-v1/
#    (apr pull will fetch it if missing).
# 4. HF_TOKEN exported in the local shell - propagated via ssh -A or
#    explicit `ssh gx10 "export HF_TOKEN=...; ./run.sh"` patterns.
#
# Constraints
# ===========
# - gx10 is aarch64 + Blackwell (sm_121). The trueno backward path on
#   Blackwell may hit the JIT pre-warming bug per memory rule
#   `feedback_blackwell_jit_blocked_training.md`. Phase 3 specifically
#   tolerates this - if the fused-NF4 path crashes, we fall back to the
#   in-tree forward-only smoke (proves teacher + kd_step orchestration
#   without weight updates) and document Phase 3a as partial.
# - Real Phase 3 discharge of F-DISTILL-SMOKE-001 needs a working
#   backward, which on Blackwell requires trueno 0.4.36 (per
#   `project_pmat_587_exec_params_blocker.md` lineage). Until then, the
#   smoke runs on RTX 4090 (lambda-vector) for a definitive verdict.
#
# Usage
# =====
#   ./scripts/dispatch-distill-phase-3-gx10.sh                # default
#   STEPS=50  ./scripts/dispatch-distill-phase-3-gx10.sh       # quick check
#   STEPS=500 ./scripts/dispatch-distill-phase-3-gx10.sh       # full smoke
#   DRY_RUN=1 ./scripts/dispatch-distill-phase-3-gx10.sh       # plan only

set -euo pipefail

# --------------------------------------------------------------------------
# Config (override via env)
# --------------------------------------------------------------------------
GX10_HOST="${GX10_HOST:-gx10}"
GX10_USER="${GX10_USER:-noah}"
GX10_REPO_PATH="${GX10_REPO_PATH:-/home/noah/src/aprender}"
TEACHER_REPO="${TEACHER_REPO:-paiml/qwen2.5-coder-7b-apache-q4k-v1}"
STUDENT_INIT="${STUDENT_INIT:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
STEPS="${STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1.5e-5}"
T="${T:-4.0}"
ALPHA="${ALPHA:-0.3}"
RUN_NAME="distill-smoke-$(date +%Y%m%d-%H%M%S)"
EVIDENCE_DIR="${EVIDENCE_DIR:-evidence/distill-phase-3-${RUN_NAME}}"
DRY_RUN="${DRY_RUN:-0}"

# --------------------------------------------------------------------------
# Pre-flight checks (local)
# --------------------------------------------------------------------------
echo "=== Phase 3 smoke dispatch ==="
echo "  target:        ${GX10_USER}@${GX10_HOST}"
echo "  teacher:       ${TEACHER_REPO}"
echo "  student init:  ${STUDENT_INIT}"
echo "  steps:         ${STEPS}"
echo "  batch_size:    ${BATCH_SIZE}"
echo "  LR:            ${LR}"
echo "  T (KD temp):   ${T}"
echo "  alpha (CE wt): ${ALPHA}"
echo "  run name:      ${RUN_NAME}"
echo "  evidence:      ${EVIDENCE_DIR}"
echo

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY-RUN] would dispatch; exiting before remote work."
    exit 0
fi

# --------------------------------------------------------------------------
# Preflight remote
# --------------------------------------------------------------------------
echo "=== remote preflight ==="
ssh "${GX10_USER}@${GX10_HOST}" "
    set -e
    cd '${GX10_REPO_PATH}' || { echo 'aprender checkout missing on gx10' >&2; exit 1; }
    git fetch --quiet origin
    git checkout main
    git pull --ff-only
    echo 'aprender HEAD:' \$(git log --oneline -1)
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"

# --------------------------------------------------------------------------
# Remote build (cuda)
# --------------------------------------------------------------------------
echo "=== remote build (cuda) ==="
ssh "${GX10_USER}@${GX10_HOST}" "
    set -e
    cd '${GX10_REPO_PATH}'
    cargo build --release -p apr-cli --features cuda --bin apr 2>&1 | tail -5
    ./target/release/apr --version
"

# --------------------------------------------------------------------------
# Pull artifacts on gx10 (idempotent)
# --------------------------------------------------------------------------
echo "=== remote artifact staging ==="
ssh "${GX10_USER}@${GX10_HOST}" "
    set -e
    cd '${GX10_REPO_PATH}'
    # Teacher: probably cached; pull is idempotent if HF_TOKEN is exported.
    ./target/release/apr pull ${TEACHER_REPO} || true
    # Student init: needed for the starting weights.
    ./target/release/apr pull ${STUDENT_INIT} || true
"

# --------------------------------------------------------------------------
# Dispatch the smoke run
# --------------------------------------------------------------------------
echo "=== dispatching smoke run on gx10 ==="
RUN_DIR_REMOTE="/mnt/nvme-raid0/runs/${RUN_NAME}"
LOG_REMOTE="${RUN_DIR_REMOTE}/launch.log"

# SPEC-DISTILL-001 Phase 3 dispatch - CLI-flag-aligned invocation (PMAT-698b).
# The post-#1797 `apr distill` surface is:
#   - positional <TEACHER>           (PathBuf - directory containing model.apr or model.safetensors)
#   - --student <STUDENT>            (PathBuf - same shape)
#   - --epochs <N>                   (no --num-steps; pipeline runs ~31 steps/epoch at default batch=32)
#   - --temperature, --alpha, --backend cuda, --output <FILE>
# Per evidence/distill-phase-3-readiness/findings.md, the earlier script used
# aspirational flags (--num-steps/--batch-size/--learning-rate/--student-init/
# --output-dir/--device) that do not exist on the post-Phase-3-prep CLI.
#
# Map the user-facing knobs to the real CLI:
#   - STEPS=500 (default) → --epochs 17 (~527 steps at default batch=32)
#   - BATCH_SIZE / LR are NOT yet exposed on the CLI; documented but unused
#     here. A follow-up PMAT-698c adds --batch-size / --learning-rate / --max-steps.
#   - TEACHER_REPO / STUDENT_INIT → resolved to ~/.cache/huggingface/hub snapshot
#     dirs via shell expansion (apr pull populates the cache).

EPOCHS_FROM_STEPS=$(( (STEPS + 30) / 31 ))  # round up: 500 → 17 epochs

ssh "${GX10_USER}@${GX10_HOST}" "
    set -e
    mkdir -p '${RUN_DIR_REMOTE}'
    cd '${GX10_REPO_PATH}'

    # Resolve HF repo → local cache snapshot dir. The hub layout is
    # models--<org>--<name>/snapshots/<sha>/, with one snapshot per pull.
    hf_repo_to_dir() {
        local repo=\"\$1\"
        local sanitized=\"\${repo//\\//--}\"
        local cache_root=\"\$HOME/.cache/huggingface/hub/models--\${sanitized}\"
        ls -td \"\${cache_root}/snapshots/\"*/ 2>/dev/null | head -1 | sed 's:/\$::'
    }
    TEACHER_DIR=\$(hf_repo_to_dir '${TEACHER_REPO}')
    STUDENT_DIR=\$(hf_repo_to_dir '${STUDENT_INIT}')
    if [ -z \"\$TEACHER_DIR\" ] || [ ! -d \"\$TEACHER_DIR\" ]; then
        echo \"teacher cache dir not found for '${TEACHER_REPO}' - apr pull failed?\" >&2
        exit 1
    fi
    if [ -z \"\$STUDENT_DIR\" ] || [ ! -d \"\$STUDENT_DIR\" ]; then
        echo \"student cache dir not found for '${STUDENT_INIT}' - apr pull failed?\" >&2
        exit 1
    fi
    echo \"teacher dir: \$TEACHER_DIR\"
    echo \"student dir: \$STUDENT_DIR\"

    nohup ./target/release/apr distill \"\$TEACHER_DIR\" \\
        --student \"\$STUDENT_DIR\" \\
        --epochs ${EPOCHS_FROM_STEPS} \\
        --temperature ${T} \\
        --alpha ${ALPHA} \\
        --backend cuda \\
        --output '${RUN_DIR_REMOTE}/student.apr' \\
        > '${LOG_REMOTE}' 2>&1 &
    DISPATCH_PID=\$!
    echo \"dispatched PID \${DISPATCH_PID}\"
    sleep 5
    if ! kill -0 \${DISPATCH_PID} 2>/dev/null; then
        echo 'EARLY EXIT - capturing tail of log:' >&2
        tail -40 '${LOG_REMOTE}' >&2 || true
        exit 1
    fi
    echo \"PID alive after 5s - training underway\"
"

# --------------------------------------------------------------------------
# Pull evidence
# --------------------------------------------------------------------------
mkdir -p "${EVIDENCE_DIR}"
cat > "${EVIDENCE_DIR}/dispatch.json" <<JSON
{
  "ticket": "PMAT-697",
  "phase": "SPEC-DISTILL-001 Phase 3 - E2E smoke",
  "falsifier": "F-DISTILL-SMOKE-001",
  "run_name": "${RUN_NAME}",
  "host": "${GX10_HOST}",
  "teacher": "${TEACHER_REPO}",
  "student_init": "${STUDENT_INIT}",
  "steps": ${STEPS},
  "batch_size": ${BATCH_SIZE},
  "learning_rate": "${LR}",
  "kd_temperature": "${T}",
  "kd_alpha": "${ALPHA}",
  "remote_run_dir": "${RUN_DIR_REMOTE}",
  "remote_log": "${LOG_REMOTE}",
  "dispatched_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo
echo "=== dispatch complete ==="
echo "  evidence/manifest: ${EVIDENCE_DIR}/dispatch.json"
echo "  remote log:        ssh ${GX10_HOST} 'tail -f ${LOG_REMOTE}'"
echo "  watch:             scripts/watch-distill-phase-3-gx10.sh ${RUN_NAME}"
echo
echo "Falsifier F-DISTILL-SMOKE-001 verdict will be in ${LOG_REMOTE}'s"
echo "metrics block once training completes. Plan: ~${STEPS} * ~1.5s/step on"
echo "RTX 4090 ≈ $((STEPS * 3 / 2 / 60))min ; expect ~$((STEPS * 4 / 60))min on"
echo "Blackwell pending trueno 0.4.36 JIT fix."
