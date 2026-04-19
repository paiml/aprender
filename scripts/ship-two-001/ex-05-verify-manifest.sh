#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SHIP-TWO-001 EX-05 — Verify publish manifests against live URLs
# ─────────────────────────────────────────────────────────────
# Spec §12.2 EX-05 — runs FALSIFY-PM-002-live and FALSIFY-PM-003 against
# each uploaded artifact, discharging RJ-PM-001 and AC-PM-004.
#
# F-PUBLISH-EXTRA-001::dogfood_ex05 — native `apr validate-manifest --live`
# supersedes the previous external-interpreter block. Zero ext dependencies.
#
# Per SPEC-SHIP-TWO-001 §12.7, we ship THREE artifacts per release:
#   - .apr            (paiml-qwen2.5-coder-7b-apache-q4k-v1-apr.yaml)
#   - .safetensors    (-safetensors.yaml, fp16 per F-PUBLISH-EXTRA-001)
#   - .gguf           (-gguf.yaml, Q4_K_M)
# Each manifest is validated independently.
#
# Inputs:
#   MANIFESTS      space-separated list (default: all three per-format manifests)
#   EVIDENCE_DIR   output directory (default: evidence/ship-two-001/)
#   APR_BIN        path to apr binary (default: $REPO/target/release/apr)
#
# Outputs:
#   evidence/ship-two-001/ex-05-manifest-verify-<format>.json   per manifest
#   evidence/ship-two-001/ex-05-manifest-verify.json            top-level summary
# ─────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

declare -a DEFAULT_MANIFESTS=(
    "contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-apr.yaml"
    "contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-safetensors.yaml"
    "contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-gguf.yaml"
)

declare -a MANIFESTS_ARR
if [[ -n "${MANIFESTS:-}" ]]; then
    # shellcheck disable=SC2206  # word-splitting is intentional for user override
    MANIFESTS_ARR=(${MANIFESTS})
else
    MANIFESTS_ARR=("${DEFAULT_MANIFESTS[@]}")
fi

EVIDENCE_DIR="${EVIDENCE_DIR:-evidence/ship-two-001}"
DEFAULT_APR_BIN="${REPO_ROOT}/target/release/apr"
APR_BIN="${APR_BIN:-$DEFAULT_APR_BIN}"
mkdir -p "$EVIDENCE_DIR"

if [[ ! -x "$APR_BIN" ]]; then
    echo "ERROR: apr binary not found at $APR_BIN" >&2
    echo "       Build with: cargo build -p apr-cli --release --features inference" >&2
    exit 2
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOP_OUT="${EVIDENCE_DIR}/ex-05-manifest-verify.json"

overall="PASS"
per_manifest_json=()

for MF in "${MANIFESTS_ARR[@]}"; do
    if [[ ! -f "$MF" ]]; then
        echo "ERROR: manifest not found: $MF" >&2
        exit 2
    fi
    base="$(basename "$MF" .yaml)"
    out="${EVIDENCE_DIR}/ex-05-manifest-verify-${base}.json"
    echo "──── $MF ────"
    # --live discharges FALSIFY-PM-003 (HEAD) + FALSIFY-PM-002-live (streaming sha256)
    # directly inside apr via ureq — no external interpreters, no extra deps.
    if "$APR_BIN" validate-manifest "$MF" --live --json > "$out"; then
        echo "  PASS"
    else
        echo "  FAIL"
        overall="FAIL"
    fi
    per_manifest_json+=("$(printf '{"manifest":"%s","report_file":"%s"}' "$MF" "$out")")
done

# Top-level summary file — keeps ex-05 output schema stable for downstream tools
{
    printf '{\n'
    printf '  "timestamp_utc": "%s",\n' "$TIMESTAMP"
    printf '  "tool": "apr validate-manifest --live",\n'
    printf '  "manifests": [\n'
    last_idx=$(( ${#per_manifest_json[@]} - 1 ))
    for i in "${!per_manifest_json[@]}"; do
        printf '    %s' "${per_manifest_json[i]}"
        if (( i < last_idx )); then
            printf ','
        fi
        printf '\n'
    done
    printf '  ],\n'
    printf '  "overall": "%s"\n' "$overall"
    printf '}\n'
} > "$TOP_OUT"

echo ""
echo "EX-05 verification archived: $TOP_OUT"
echo "overall: $overall"

[[ "$overall" == "PASS" ]]
