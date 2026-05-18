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
# PMAT-698d: gx10 has no /mnt/nvme-raid0 (that's lambda-vector layout).
# Default to $HOME/runs which exists on most setups; override via env.
GX10_RUN_PREFIX="${GX10_RUN_PREFIX:-/home/noah/runs}"
# PMAT-698d: paiml/qwen2.5-coder-7b-apache-q4k-v1 is GGUF, which apr
# distill cuda backend (CudaTrainerTeacher::for_inference) does NOT
# load — it only handles APR + SafeTensors. The original MODEL-1
# teacher needs an apr import GGUF->APR step before distill (deferred
# to PMAT-698e); for the Phase 3 smoke we default to a SafeTensors
# teacher that loads directly.
TEACHER_REPO="${TEACHER_REPO:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
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
# PMAT-698d: configurable run prefix; gx10 has no /mnt/nvme-raid0/.
RUN_DIR_REMOTE="${GX10_RUN_PREFIX}/${RUN_NAME}"
LOG_REMOTE="${RUN_DIR_REMOTE}/launch.log"

# SPEC-DISTILL-001 Phase 3 dispatch - CLI-flag-aligned invocation (PMAT-698b
# + PMAT-698d cache staging).
#
# The post-#1797 apr distill cuda surface is:
#   - positional <TEACHER>       (PathBuf - directory with model.apr or model.safetensors)
#   - --student <STUDENT>        (PathBuf - same shape)
#   - --epochs <N>               (no --num-steps; pipeline runs ~31 steps/epoch at default batch=32)
#   - --temperature, --alpha, --backend cuda, --output <FILE>
#
# Per evidence/distill-phase-3-readiness/findings.md, the original script used
# aspirational flags (--num-steps / --batch-size / --learning-rate / --student-init /
# --output-dir / --device) and an HF-cache lookup path. The real `apr pull`
# layout is pacha-based:
#
#   /home/noah/.cache/pacha/models/<sha>.safetensors
#   /home/noah/.cache/pacha/models/<sha>.tokenizer.json
#   /home/noah/.cache/pacha/models/<sha>.config.json
#   /home/noah/.cache/pacha/models/<sha>.tokenizer_config.json
#
# But `apr distill --backend cuda` expects a DIRECTORY containing model.apr or
# model.safetensors. So PMAT-698d adds a stage_repo helper that:
#   1. captures `apr pull` Path: from stdout
#   2. mkdirs a stage subdir under RUN_DIR_REMOTE/teacher /student
#   3. symlinks model.<ext> + companion tokenizer/config files
#   4. for GGUF teachers, runs `apr import --preserve-q4k` to convert to APR
#      (Phase 3 default avoids this by using a SafeTensors teacher.)
#
# Map the user-facing knobs to the real CLI:
#   - STEPS=500 (default) → --epochs 17 (~527 steps at default batch=32)
#   - BATCH_SIZE / LR are NOT yet exposed on the CLI; documented but unused.
#     A follow-up PMAT-698c adds --batch-size / --learning-rate / --max-steps.

EPOCHS_FROM_STEPS=$(( (STEPS + 30) / 31 ))  # round up: 500 -> 17 epochs

ssh "${GX10_USER}@${GX10_HOST}" "
    set -e
    mkdir -p '${RUN_DIR_REMOTE}'
    cd '${GX10_REPO_PATH}'

    # PMAT-698d: stage pacha-cached repo into a directory layout that
    # CudaTrainerTeacher::for_inference expects (model.apr or model.safetensors
    # at the dir root). Captures the cache path from \`apr pull\` stdout.
    stage_repo() {
        local repo=\"\$1\"
        local stage_dir=\"\$2\"
        local arch_hint=\"\$3\"
        mkdir -p \"\$stage_dir\"
        local pull_out
        pull_out=\$(./target/release/apr pull \"\$repo\" 2>&1)
        local cache_path
        cache_path=\$(echo \"\$pull_out\" | grep -E '^[[:space:]]+Path:' | head -1 | awk '{print \$2}')
        if [ -z \"\$cache_path\" ] || [ ! -f \"\$cache_path\" ]; then
            echo \"failed to resolve cache for \$repo (Path: not found in apr pull output)\" >&2
            echo \"\$pull_out\" | tail -20 >&2
            return 1
        fi
        local ext=\"\${cache_path##*.}\"
        local sha
        sha=\$(basename \"\$cache_path\" \".\$ext\")
        local cache_dir
        cache_dir=\$(dirname \"\$cache_path\")
        case \"\$ext\" in
            safetensors)
                ln -sf \"\$cache_path\" \"\$stage_dir/model.safetensors\"
                ;;
            apr)
                ln -sf \"\$cache_path\" \"\$stage_dir/model.apr\"
                ;;
            gguf)
                echo \"GGUF detected for \$repo - converting to APR via apr import (--arch \$arch_hint)\" >&2
                ./target/release/apr import \"\$cache_path\" --arch \"\$arch_hint\" --preserve-q4k -o \"\$stage_dir/model.apr\" >&2
                ;;
            *)
                echo \"unknown model file extension: \$ext (cache_path=\$cache_path)\" >&2
                return 1
                ;;
        esac
        for companion in tokenizer.json config.json tokenizer_config.json; do
            local src=\"\${cache_dir}/\${sha}.\${companion}\"
            if [ -f \"\$src\" ]; then
                ln -sf \"\$src\" \"\$stage_dir/\${companion}\"
            fi
        done
        echo \"staged \$repo -> \$stage_dir\" >&2
    }

    TEACHER_DIR=\"${RUN_DIR_REMOTE}/teacher\"
    STUDENT_DIR=\"${RUN_DIR_REMOTE}/student\"
    stage_repo '${TEACHER_REPO}' \"\$TEACHER_DIR\" qwen2
    stage_repo '${STUDENT_INIT}' \"\$STUDENT_DIR\" qwen2
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
