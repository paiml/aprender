#!/usr/bin/env bash
# SPEC-DISTILL-001 Phase 6 — publish v2 trained student to HuggingFace Hub.
#
# Falsifier: F-DISTILL-PUBLISH-001 — round-trip works (apr pull <published>
# loads the same checkpoint that was uploaded).
#
# Per SPEC-HF-PUBLISH-001 (the canonical 12-file minimum publish workflow):
#   - model.safetensors      (or model.apr)
#   - config.json
#   - tokenizer.json
#   - tokenizer_config.json
#   - generation_config.json
#   - README.md (model card)
#   - LICENSE
#
# Prerequisites
# =============
# 1. Phase 5 completed: pass@1 verdict from F-DISTILL-HUMANEVAL-001.
# 2. HF_TOKEN exported with `write` scope for paiml/ namespace.
# 3. Trained student checkpoint at MODEL_DIR.
#
# Usage
# =====
#   ./scripts/dispatch-phase6-publish.sh                       # default
#   REPO_ID=paiml/qwen2.5-coder-0.5b-distilled-v2 \
#     ./scripts/dispatch-phase6-publish.sh
#   DRY_RUN=1 ./scripts/dispatch-phase6-publish.sh             # plan only

set -euo pipefail

# --------------------------------------------------------------------------
# Config (override via env)
# --------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-/home/noah/runs/distill-smoke-20260520-124239/student-trained.apr}"
REPO_ID="${REPO_ID:-paiml/qwen2.5-coder-0.5b-distilled-v2}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-Coder-0.5B distilled (SPEC-DISTILL-001 v2)}"
LICENSE="${LICENSE:-apache-2.0}"
LIBRARY_NAME="${LIBRARY_NAME:-aprender}"
PIPELINE_TAG="${PIPELINE_TAG:-text-generation}"
TAGS="${TAGS:-distillation,qwen2.5,code,blackwell-gb10}"
COMMIT_MSG="${COMMIT_MSG:-SPEC-DISTILL-001 v2 distilled student — Stage D 50K-step on GB10}"
DRY_RUN="${DRY_RUN:-0}"
GX10_HOST="${GX10_HOST:-gx10}"
GX10_USER="${GX10_USER:-noah}"
# Whether to publish from gx10 (where the model lives) or locally after scp.
PUBLISH_HOST="${PUBLISH_HOST:-gx10}"  # gx10 | local

# --------------------------------------------------------------------------
# Pre-flight (local)
# --------------------------------------------------------------------------
echo "=== Phase 6 publish to HuggingFace ==="
echo "  repo_id:       ${REPO_ID}"
echo "  model_dir:     ${MODEL_DIR}"
echo "  model_name:    ${MODEL_NAME}"
echo "  license:       ${LICENSE}"
echo "  library:       ${LIBRARY_NAME}"
echo "  pipeline_tag:  ${PIPELINE_TAG}"
echo "  tags:          ${TAGS}"
echo "  publish_host:  ${PUBLISH_HOST}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "  mode:          DRY-RUN (plan only)"
fi
echo

if [ -z "${HF_TOKEN:-}" ] && [ "${PUBLISH_HOST}" = "local" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "HF_TOKEN not set (required for non-dry-run local publish)" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Validate model directory exists on the publish host (skip on dry-run)
# --------------------------------------------------------------------------
if [ "${DRY_RUN}" = "1" ]; then
    echo "=== remote model validation: SKIPPED (dry-run) ==="
    echo "[DRY-RUN] would publish; exiting before remote work."
    exit 0
fi
if [ "${PUBLISH_HOST}" = "gx10" ]; then
    echo "=== remote model validation ==="
    ssh "${GX10_USER}@${GX10_HOST}" "
        set -e
        if [ ! -d '${MODEL_DIR}' ] && [ ! -f '${MODEL_DIR}' ]; then
            echo 'MODEL_DIR not found at ${MODEL_DIR}' >&2
            exit 1
        fi
        # apr publish expects a DIRECTORY containing model files.
        if [ -f '${MODEL_DIR}' ]; then
            echo 'MODEL_DIR points to a FILE; apr publish needs a directory' >&2
            exit 1
        fi
        ls -lh '${MODEL_DIR}/' | head -10
    "
fi

# --------------------------------------------------------------------------
# Publish
# --------------------------------------------------------------------------
APR_PUBLISH_ARGS=(
    "${MODEL_DIR}"
    "${REPO_ID}"
    --model-name "${MODEL_NAME}"
    --license "${LICENSE}"
    --library-name "${LIBRARY_NAME}"
    --pipeline-tag "${PIPELINE_TAG}"
    --tags "${TAGS}"
    --message "${COMMIT_MSG}"
    --json
)
if [ "${DRY_RUN}" = "1" ]; then
    APR_PUBLISH_ARGS+=(--dry-run)
fi

if [ "${PUBLISH_HOST}" = "gx10" ]; then
    echo "=== dispatching apr publish on gx10 ==="
    # Quote each arg for the SSH heredoc.
    PRINTABLE_ARGS=""
    for a in "${APR_PUBLISH_ARGS[@]}"; do
        PRINTABLE_ARGS+=" $(printf '%q' "$a")"
    done
    ssh "${GX10_USER}@${GX10_HOST}" "
        set -e
        cd '${GX10_REPO_PATH:-/home/noah/src/aprender}'
        ./target/release/apr publish ${PRINTABLE_ARGS} 2>&1 | tail -40
    "
else
    echo "=== dispatching apr publish locally ==="
    # The local branch hardcoded /mnt/nvme-raid0/targets/aprender/release/apr
    # while the gx10 branch three lines up correctly used a checkout-relative
    # ./target/release/apr - the same publish, one path per host, only one of
    # them pinned. That /mnt path is orphaned (nothing writes it), so a local
    # publish shipped a model to Hugging Face using a binary of unknown age
    # (#2358). Ask cargo instead.
    . "$(dirname "$0")/apr_bin.sh" || exit 1
    "$APR" publish "${APR_PUBLISH_ARGS[@]}"
fi

echo
echo "=== publish complete ==="
echo "Verify: https://huggingface.co/${REPO_ID}"
echo "Round-trip falsifier (F-DISTILL-PUBLISH-001):"
echo "  apr pull ${REPO_ID} && apr qa <cached-path>"
