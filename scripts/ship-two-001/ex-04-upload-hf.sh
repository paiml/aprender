#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SHIP-TWO-001 EX-04 -- Upload teacher to HF Hub via `apr publish`
# ─────────────────────────────────────────────────────────────
# Dogfood-first: this script invokes OUR OWN product -- `apr publish`
# with `--manifest` + `--extra-file` -- for every format we ship.
# No Python, no `huggingface-cli`, no `uv run`, no `pip install`.
#
# Contract: contracts/apr-cli-publish-extra-v1.yaml (F-PUBLISH-EXTRA-001)
#   - manifest_upload_roundtrip: per-format manifest, sha256 pre-flight guard
#   - extra_file_passthrough:    tokenizer.json uploaded verbatim
#   - no_readme_when_manifest:   manifest.yaml IS the provenance doc
#   - dogfood_shell_script:      this script -- no Python uploaders
#   - three_format_preference:   .apr + .safetensors (fp16) + .gguf
#   - safetensors_dtype_fp16:    enforced upstream at export time
#   - preflight_validate_manifest: PM-001..007 must PASS BEFORE any network I/O
#     (discharges FALSIFY-PUB-EXTRA-009; covers FALSIFY-PM-007 §12.7.2 ship-blocker)
#
# REQUIRES:
#   - HF_TOKEN env var with write access to paiml org
#   - All three artifacts + tokenizer staged in STAGING_DIR (below)
#   - Per-format manifests at contracts/publish-manifests/*-{apr,safetensors,gguf}.yaml
#   - Canonical release binary `/mnt/nvme-raid0/targets/aprender/release/apr`
#     built with `--features cuda` including F-PUBLISH-EXTRA-001 surgery
#
# Falsifies: FALSIFY-PUB-EXTRA-005 (dogfood gate); discharges -001/-006/-007/-009
#
# Next step after this: bash scripts/ship-two-001/ex-05-verify-manifest.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN env var required (write access to paiml)}"

# Default: the binary THIS CHECKOUT builds (#2358). The old default named one
# machine's build output; this script UPLOADS A MODEL TO HUGGING FACE, so the
# binary that produced the artifacts must be the one under test. APR_BIN still
# overrides.
APR_BIN="${APR_BIN:-}"
if [ -z "$APR_BIN" ] && . "$(dirname "$0")/../apr_bin.sh" 2>/dev/null; then
    APR_BIN="$APR"
fi
STAGING_DIR="${STAGING_DIR:-/mnt/nvme-raid0/models/ship-two-001}"
MODEL_ID="paiml/qwen2.5-coder-7b-apache-q4k-v1"
MANIFEST_DIR="contracts/publish-manifests"
TOKENIZER="${STAGING_DIR}/tokenizer.json"

if [[ ! -x "$APR_BIN" ]]; then
    echo "ABORT: canonical apr binary not found at $APR_BIN" >&2
    echo "       build with: cargo build --release -p apr-cli --bin apr --features cuda" >&2
    exit 1
fi

if [[ ! -f "$TOKENIZER" ]]; then
    echo "ABORT: tokenizer not staged at $TOKENIZER" >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# PRE-FLIGHT: FALSIFY-PUB-EXTRA-009 + FALSIFY-PM-007 §12.7.2 gate
# ─────────────────────────────────────────────────────────────
# Runs `apr validate-manifest --artifact <local_file>` against every
# per-format manifest BEFORE any network I/O. Any FAIL aborts the ship.
# This catches:
#   PM-001  schema drift
#   PM-002  sha256 mismatch (local file vs manifest)
#   PM-004  bad SPDX id
#   PM-005  recipe sha256 drift
#   PM-006  broken parent chain
#   PM-007  safetensors dtype Poka-Yoke (fp16 manifest + F32 header = FAIL)
# before we upload 30+ GiB of the wrong bytes.
preflight_validate_manifest() {
    local fmt="$1"
    local manifest="${MANIFEST_DIR}/paiml-qwen2.5-coder-7b-apache-q4k-v1-${fmt}.yaml"
    local url
    url=$(awk '/^artifact_url:/ {print $2}' "$manifest")
    local base
    base=$(basename "$url")
    local artifact="${STAGING_DIR}/${base}"
    if [[ ! -f "$artifact" ]]; then
        echo "ABORT: staged artifact not found: $artifact" >&2
        exit 1
    fi
    echo "pre-flight: apr validate-manifest ${manifest##*/} --artifact ${base}"
    if ! "$APR_BIN" validate-manifest "$manifest" --artifact "$artifact" >/dev/null; then
        echo "ABORT: pre-flight validation FAILED for .${fmt} -- no network I/O performed" >&2
        echo "       re-run without --json to see details:" >&2
        echo "         $APR_BIN validate-manifest $manifest --artifact $artifact" >&2
        exit 2
    fi
    echo "  PASS"
}

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  PRE-FLIGHT: FALSIFY-PM-001..007 on all three formats"
echo "══════════════════════════════════════════════════════════════"
preflight_validate_manifest apr
preflight_validate_manifest safetensors
preflight_validate_manifest gguf

# Publish each format with its per-format manifest. The pre-flight sha256
# guard inside `apr publish` aborts before any network I/O if the local
# artifact's hash disagrees with the manifest (FALSIFY-PUB-EXTRA-002).
publish_format() {
    local fmt="$1"  # apr | safetensors | gguf
    local manifest="${MANIFEST_DIR}/paiml-qwen2.5-coder-7b-apache-q4k-v1-${fmt}.yaml"

    if [[ ! -f "$manifest" ]]; then
        echo "ABORT: manifest not found: $manifest" >&2
        exit 1
    fi

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  apr publish → ${MODEL_ID} (.${fmt})"
    echo "  manifest: ${manifest}"
    echo "══════════════════════════════════════════════════════════════"

    "$APR_BIN" publish \
        "$STAGING_DIR" \
        "$MODEL_ID" \
        --manifest "$manifest" \
        --extra-file "$TOKENIZER" \
        --license apache-2.0 \
        --message "SHIP-TWO-001 EX-04: publish .${fmt} via apr publish (F-PUBLISH-EXTRA-001)"
}

# .apr first (native), then .safetensors (fp16), then .gguf.
# Each publish adds to the same HF repo (create_repo=true is idempotent on HF).
publish_format apr
publish_format safetensors
publish_format gguf

echo ""
echo "✓ EX-04 complete: all three formats uploaded to https://huggingface.co/${MODEL_ID}"
echo "  next: bash scripts/ship-two-001/ex-05-verify-manifest.sh"
