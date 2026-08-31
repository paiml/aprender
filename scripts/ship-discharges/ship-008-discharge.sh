#!/usr/bin/env bash
# ship-008-discharge.sh - LIVE dispatch for GATE-CHAT-SHIP-008
#
# Contract: contracts/chat-template-v1.yaml (GATE-CHAT-SHIP-008)
# AC: AC-SHIP1-008 - MODEL-1 teacher (Qwen2.5-Coder-7B-Instruct, ChatML)
#     renders canonical (system, user) messages to a byte-exact golden.
#
# Canonical command:
#   apr run paiml/qwen2.5-coder-7b-apache-q4k-v1 --prompt <canonical>
#
# Pass criterion: byte-exact match between the prompt-as-rendered and the
# golden defined in crates/aprender-core/src/text/chat_template/ship_008.rs
#   AC_SHIP1_008_CANONICAL_GOLDEN.
# Algorithm-level proof: crates/aprender-core/src/text/chat_template/ship_008.rs
#   ::verdict_from_chat_template_render(rendered, golden) -> Pass
#
# Note: this script verifies the *rendered prompt* (what the model sees,
# i.e. the ChatML wrapping) is byte-exact to the spec golden - that is
# what AC-SHIP1-008 binds. The completion bytes that follow are content,
# which is verified separately by SHIP-002 (Python syntax) and SHIP-005
# (HumanEval pass@1).
#
# Usage: bash scripts/ship-discharges/ship-008-discharge.sh \
#            [--apr-binary <path>] [--model <path-or-hf-id>]
#
# Exit 0 on Pass, 1 on Fail. Writes evidence to
#   evidence/ship-008-full-discharge/discharge-evidence-v1.json

set -euo pipefail

# --- Defaults -----------------------------------------------------------
# Default: the binary THIS CHECKOUT builds (#2358). The previous default was a
# hardcoded /mnt/nvme-raid0/targets/aprender/release/apr - a path nothing writes
# any more, and one that on 2026-08-01 was two minor versions stale while docs
# still called it canonical. A discharge script signs off a SHIP; signing it off
# with a binary of unknown provenance certifies nothing. `--apr-binary` and the
# APR_BINARY env var both still override.
APR_BINARY="${APR_BINARY:-}"
if [ -z "$APR_BINARY" ] && . "$(dirname "$0")/../apr_bin.sh" 2>/dev/null; then
    APR_BINARY="$APR"
fi
MODEL="${MODEL:-paiml/qwen2.5-coder-7b-apache-q4k-v1}"
SYSTEM_MSG="You are a helpful coding assistant."
USER_MSG="Write a Python function to compute the nth Fibonacci number."
EVIDENCE_DIR="evidence/ship-008-full-discharge"
EVIDENCE_FILE="${EVIDENCE_DIR}/discharge-evidence-v1.json"
GOLDEN_FILE="${EVIDENCE_DIR}/golden.txt"
RENDERED_FILE="${EVIDENCE_DIR}/rendered.txt"
DIFF_FILE="${EVIDENCE_DIR}/byte-diff.txt"

# --- Arg parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apr-binary)  APR_BINARY="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//' >&2
            exit 0
            ;;
        *)
            echo "FAIL: unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$EVIDENCE_DIR"

if [[ ! -x "$APR_BINARY" ]]; then
    echo "FAIL: apr binary not executable at: $APR_BINARY" >&2
    exit 1
fi

echo "SHIP-008 dispatch - LIVE discharge"
echo "  apr binary : $APR_BINARY"
echo "  model      : $MODEL"
echo "  system msg : $SYSTEM_MSG"
echo "  user msg   : $USER_MSG"
echo ""

# --- Step 1: write canonical golden -------------------------------------
# Mirrors AC_SHIP1_008_CANONICAL_GOLDEN in
# crates/aprender-core/src/text/chat_template/ship_008.rs (byte-exact).
# DO NOT edit by hand - must track the constant. printf preserves the
# trailing newline expected by ChatML's `<|im_start|>assistant\n`.
printf '%s\n%s%s\n%s\n%s%s\n%s\n' "<|im_start|>system" "$SYSTEM_MSG" "<|im_end|>" "<|im_start|>user" "$USER_MSG" "<|im_end|>" "<|im_start|>assistant" >"$GOLDEN_FILE"

GOLDEN_SHA="$(sha256sum "$GOLDEN_FILE" | awk '{print $1}')"
GOLDEN_BYTES="$(wc -c < "$GOLDEN_FILE")"
GOLDEN_SHA_SHORT="${GOLDEN_SHA:0:16}"
echo "Step 1: wrote golden - ${GOLDEN_BYTES} bytes, sha256=${GOLDEN_SHA_SHORT}..."

# --- Step 2: dispatch apr run + capture rendered prompt -----------------
# `apr run` should expose a flag to print the rendered prompt without
# generating tokens. If it does not, we instead run with --max-tokens 1
# and capture the rendered prompt via apr's debug channel.
echo "Step 2: apr run + capture rendered prompt"

CAPTURE_MODE=""
START_EPOCH=$(date -u +%s)
RUN_EXIT=0
if grep -q -- '--print-prompt' <<< "$("$APR_BINARY" run --help 2>&1)" ; then
    "$APR_BINARY" run "$MODEL" --system "$SYSTEM_MSG" --prompt "$USER_MSG" --print-prompt >"$RENDERED_FILE" 2>&1 || RUN_EXIT=$?
    CAPTURE_MODE="--print-prompt"
else
    # Fallback: short generation, recover rendered prompt from a known
    # marker (apr emits the rendered ChatML when --debug is passed).
    "$APR_BINARY" run "$MODEL" --system "$SYSTEM_MSG" --prompt "$USER_MSG" --max-tokens 1 --debug >"$RENDERED_FILE" 2>&1 || RUN_EXIT=$?
    CAPTURE_MODE="--debug --max-tokens 1 (fallback)"
fi
END_EPOCH=$(date -u +%s)
DURATION_SEC=$(( END_EPOCH - START_EPOCH ))

echo "  capture mode: $CAPTURE_MODE"
echo "  rendered  -> $RENDERED_FILE (${DURATION_SEC} sec, exit=$RUN_EXIT)"

# --- Step 3: byte-diff golden vs rendered -------------------------------
echo "Step 3: byte-diff golden vs rendered"

# Use cmp for byte-exactness (load-bearing per AC-SHIP1-008).
BYTE_EQUAL="false"
DIFF_DETAILS=""
if cmp -s "$GOLDEN_FILE" "$RENDERED_FILE"; then
    BYTE_EQUAL="true"
    DIFF_DETAILS="byte-identical"
    : > "$DIFF_FILE"
else
    # Capture first divergent byte for the evidence record.
    cmp "$GOLDEN_FILE" "$RENDERED_FILE" > "$DIFF_FILE" 2>&1 || true
    DIFF_DETAILS="$(head -1 "$DIFF_FILE")"
fi

RENDERED_SHA="$(sha256sum "$RENDERED_FILE" | awk '{print $1}')"
RENDERED_BYTES="$(wc -c < "$RENDERED_FILE")"
RENDERED_SHA_SHORT="${RENDERED_SHA:0:16}"
echo "  rendered : ${RENDERED_BYTES} bytes, sha256=${RENDERED_SHA_SHORT}..."
echo "  byte-equal=$BYTE_EQUAL ($DIFF_DETAILS)"

# --- Step 4: verdict ----------------------------------------------------
if [[ "$RUN_EXIT" -eq 0 && "$BYTE_EQUAL" == "true" ]]; then
    VERDICT="PASS"
    EXIT_CODE=0
else
    VERDICT="FAIL"
    EXIT_CODE=1
fi

# --- Step 5: emit evidence JSON -----------------------------------------
HOSTNAME_VAL="$(hostname)"
APR_VERSION="$( "$APR_BINARY" --version 2>/dev/null || echo "unknown" )"
DATE_UTC="$(date -u +%Y-%m-%d)"

# JSON-encode the diff details (single-line excerpt).
DIFF_DETAILS_JSON="$(printf '%s' "$DIFF_DETAILS" | jq -Rs . 2>/dev/null || printf '"%s"' "$DIFF_DETAILS")"

cat > "$EVIDENCE_FILE" <<JSON
{
  "schema_ref": "contracts/chat-template-v1.yaml#GATE-CHAT-SHIP-008.discharged_evidence",
  "evidence_id": "GATE-CHAT-SHIP-008-DISCHARGE-DISPATCH-V1",
  "binds_to": "AC-SHIP1-008",
  "falsification_id": "GATE-CHAT-SHIP-008",
  "discharge_date": "${DATE_UTC}",
  "host": {
    "hostname": "${HOSTNAME_VAL}",
    "apr_binary": "${APR_BINARY}",
    "apr_version": "${APR_VERSION}"
  },
  "command": "apr run ${MODEL} --system <canonical-system> --prompt <canonical-user> ${CAPTURE_MODE}",
  "model": "${MODEL}",
  "capture_mode": "${CAPTURE_MODE}",
  "duration_seconds": ${DURATION_SEC},
  "apr_run_exit_code": ${RUN_EXIT},
  "golden_file": "${GOLDEN_FILE}",
  "golden_sha256": "${GOLDEN_SHA}",
  "golden_bytes": ${GOLDEN_BYTES},
  "rendered_file": "${RENDERED_FILE}",
  "rendered_sha256": "${RENDERED_SHA}",
  "rendered_bytes": ${RENDERED_BYTES},
  "byte_equal": ${BYTE_EQUAL},
  "diff_details": ${DIFF_DETAILS_JSON},
  "verdict_from_chat_template_render": "${VERDICT}",
  "overall": "${VERDICT}"
}
JSON

# --- Step 6: report -----------------------------------------------------
echo ""
echo "Verdict: $VERDICT"
echo "Evidence: $EVIDENCE_FILE"

if [[ "$VERDICT" == "PASS" ]]; then
    echo "SHIP-008 DISCHARGED (live): rendered prompt is byte-exact to golden"
else
    echo "SHIP-008 still PARTIAL_ALGORITHM_LEVEL: byte-equal=$BYTE_EQUAL, exit=$RUN_EXIT"
    echo "  see $DIFF_FILE for first divergence"
fi

exit "$EXIT_CODE"
