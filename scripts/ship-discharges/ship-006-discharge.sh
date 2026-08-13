#!/usr/bin/env bash
# ship-006-discharge.sh - LIVE dispatch for FALSIFY-QA-SHIP-006
#
# Contract: contracts/apr-model-qa-v1.yaml (FALSIFY-QA-SHIP-006)
# AC: AC-SHIP1-006 - MODEL-1 teacher must pass all 8 apr qa gates
#     (golden, throughput, ollama parity, gpu speedup, tensor contracts,
#      format parity, ptx parity, metadata).
#
# Canonical command:
#   apr qa paiml/qwen2.5-coder-7b-apache-q4k-v1 --json
#
# Pass criterion: aggregate-AND over 8 gate booleans - every gate
# reports `"pass": true` AND the array has exactly 8 entries.
# Algorithm-level proof: crates/aprender-core/src/qa/ship_006.rs
#   ::verdict_from_qa_gates(&[bool; 8]) -> Pass
#
# Usage: bash scripts/ship-discharges/ship-006-discharge.sh \
#            [--apr-binary <path>] [--model <path-or-hf-id>]
#
# Exit 0 on Pass, 1 on Fail. Writes evidence to
#   evidence/ship-006-full-discharge/discharge-evidence-v1.json

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
REQUIRED_GATE_COUNT=8
EVIDENCE_DIR="evidence/ship-006-full-discharge"
EVIDENCE_FILE="${EVIDENCE_DIR}/discharge-evidence-v1.json"
QA_RAW_FILE="${EVIDENCE_DIR}/qa-raw.json"

# --- Arg parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apr-binary) APR_BINARY="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
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

if ! command -v jq >/dev/null 2>&1; then
    echo "FAIL: jq required for JSON parsing" >&2
    exit 1
fi

echo "SHIP-006 dispatch - LIVE discharge"
echo "  apr binary           : $APR_BINARY"
echo "  model                : $MODEL"
echo "  required gate count  : $REQUIRED_GATE_COUNT"
echo ""

# --- Step 1: run apr qa --json ------------------------------------------
echo "Step 1: apr qa $MODEL --json"
START_EPOCH=$(date -u +%s)
QA_EXIT=0
"$APR_BINARY" qa "$MODEL" --json > "$QA_RAW_FILE" 2>&1 || QA_EXIT=$?
END_EPOCH=$(date -u +%s)
DURATION_SEC=$(( END_EPOCH - START_EPOCH ))

echo "  raw output -> $QA_RAW_FILE (${DURATION_SEC} sec, exit=$QA_EXIT)"

# --- Step 2: parse 8-gate boolean array ---------------------------------
echo "Step 2: parse 8 gates from JSON"

# apr qa --json shape varies; try common shapes:
#   { "gates": [ {"name": "...", "pass": true}, ... ] }
#   { "gates": { "golden": {"pass": true}, ... } }
#   { "results": [ {"name": "...", "pass": true}, ... ] }
# Normalize to a flat boolean array via jq.
JQ_QUERY_GATES='( .gates // .results // [] ) as $g | if ($g | type) == "array" then ($g | map(.pass // .passed // false)) elif ($g | type) == "object" then ($g | to_entries | map(.value.pass // .value.passed // false)) else [] end'
GATE_BOOLS_JSON="$(jq -r "$JQ_QUERY_GATES" "$QA_RAW_FILE" 2>/dev/null || echo '[]')"

GATE_COUNT="$(printf '%s' "$GATE_BOOLS_JSON" | jq 'length')"
PASS_COUNT="$(printf '%s' "$GATE_BOOLS_JSON" | jq '[.[] | select(. == true)] | length')"

echo "  gate_count=$GATE_COUNT (required=$REQUIRED_GATE_COUNT)"
echo "  pass_count=$PASS_COUNT"
echo "  gates: $GATE_BOOLS_JSON"

# --- Step 3: verdict (aggregate-AND) ------------------------------------
if [[ "$QA_EXIT" -eq 0 \
   && "$GATE_COUNT" == "$REQUIRED_GATE_COUNT" \
   && "$PASS_COUNT" == "$REQUIRED_GATE_COUNT" ]]; then
    VERDICT="PASS"
    EXIT_CODE=0
else
    VERDICT="FAIL"
    EXIT_CODE=1
fi

# --- Step 4: emit evidence JSON -----------------------------------------
HOSTNAME_VAL="$(hostname)"
APR_VERSION="$( "$APR_BINARY" --version 2>/dev/null || echo "unknown" )"
DATE_UTC="$(date -u +%Y-%m-%d)"

cat > "$EVIDENCE_FILE" <<JSON
{
  "schema_ref": "contracts/apr-model-qa-v1.yaml#FALSIFY-QA-SHIP-006.discharged_evidence",
  "evidence_id": "FALSIFY-SHIP-006-DISCHARGE-DISPATCH-V1",
  "binds_to": "AC-SHIP1-006",
  "falsification_id": "FALSIFY-QA-SHIP-006",
  "discharge_date": "${DATE_UTC}",
  "host": {
    "hostname": "${HOSTNAME_VAL}",
    "apr_binary": "${APR_BINARY}",
    "apr_version": "${APR_VERSION}"
  },
  "command": "apr qa ${MODEL} --json",
  "model": "${MODEL}",
  "raw_qa_output": "${QA_RAW_FILE}",
  "duration_seconds": ${DURATION_SEC},
  "apr_qa_exit_code": ${QA_EXIT},
  "required_gate_count": ${REQUIRED_GATE_COUNT},
  "gate_count": ${GATE_COUNT},
  "pass_count": ${PASS_COUNT},
  "gate_pass_array": ${GATE_BOOLS_JSON},
  "verdict_from_qa_gates": "${VERDICT}",
  "overall": "${VERDICT}"
}
JSON

# --- Step 5: report -----------------------------------------------------
echo ""
echo "Verdict: $VERDICT"
echo "Evidence: $EVIDENCE_FILE"

if [[ "$VERDICT" == "PASS" ]]; then
    echo "SHIP-006 DISCHARGED (live): all 8 qa gates pass"
else
    echo "SHIP-006 still PARTIAL_ALGORITHM_LEVEL: gate_count=$GATE_COUNT pass_count=$PASS_COUNT exit=$QA_EXIT"
fi

exit "$EXIT_CODE"
