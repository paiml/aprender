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
# #3965: the gates AC-SHIP1-006 REQUIRES, by the names apr qa reports (qa.md §3:
# golden, throughput, ollama parity, gpu speedup, tensor contracts, format parity,
# ptx parity, metadata). This used to be REQUIRED_GATE_COUNT=8, a count, while apr qa
# emitted 12. So the discharge compared 12 to 8 and could never pass, and when it did
# "pass" (the contract's 2026-05-10 note: "All 12 gates pass (6 executed, 6 skipped)")
# the skips were being counted as passes. Now: every REQUIRED gate must be REGISTERED
# by the binary, EMITTED, EXECUTED and PASSED. Every OTHER registered gate must be
# emitted and not FAILED; a skip there is neutral (cop ruling, 2026-09-23). The count
# is derived, never typed.
REQUIRED_GATES=(golden_output throughput ollama_parity gpu_speedup tensor_contract format_parity ptx_parity metadata_plausibility)
REQUIRED_GATE_COUNT=${#REQUIRED_GATES[@]}
EVIDENCE_DIR="evidence/ship-006-full-discharge"
EVIDENCE_FILE="${EVIDENCE_DIR}/discharge-evidence-v1.json"
QA_RAW_FILE="${EVIDENCE_DIR}/qa-raw.json"
QA_ERR_FILE="${EVIDENCE_DIR}/qa-stderr.log"

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
# SOURCE_DATE_EPOCH-derived (reproducible-builds convention): unset, this is
# exactly `date -u +%s` as before; set, both timestamps pin to it and
# DURATION_SEC reads 0, which is the expected/documented behavior of asking
# for a reproducible discharge run rather than a live-timed one.
START_EPOCH="${SOURCE_DATE_EPOCH:-$(date -u +%s)}"
QA_EXIT=0
# #3965: stderr goes to its OWN file. `2>&1` merged diagnostics into the JSON, so
# one stray stderr line made jq fail and every gate read as absent.
"$APR_BINARY" qa "$MODEL" --json > "$QA_RAW_FILE" 2> "$QA_ERR_FILE" || QA_EXIT=$?
END_EPOCH="${SOURCE_DATE_EPOCH:-$(date -u +%s)}"
DURATION_SEC=$(( END_EPOCH - START_EPOCH ))

echo "  raw output -> $QA_RAW_FILE (${DURATION_SEC} sec, exit=$QA_EXIT)"

# --- Step 2: judge against the binary's own gate registry (#3965) --------
echo "Step 2: required gates vs the registry apr qa publishes"
REGISTERED_JSON="$(jq -c '.gates_registered // []' "$QA_RAW_FILE" 2>/dev/null || echo '[]')"
REQUIRED_JSON="$(printf '%s\n' "${REQUIRED_GATES[@]}" | jq -R . | jq -s -c .)"
# One verdict object; every reason it is not a PASS is named in `why`.
# The judge is jq, kept in a quoted heredoc so no shell tool mistakes jq's `$req[]`
# for a bash array (bashrs did).
JQ_JUDGE=$(cat <<'JQ'
  (.gates // []) as $g
  | def one($n): ($g | map(select(.name == $n)));
  { registered: ($reg | length),
    why: (
      (if ($reg | length) == 0 then ["the report has no gates_registered: this apr predates the registry, so what it should have run cannot be derived"] else [] end)
      + [ $req[] | select(. as $n | ($reg | index($n)) == null) | "required gate \(.) is not in the binary registry" ]
      + [ $reg[] as $n | select((one($n) | length) != 1) | "registered gate \($n) was emitted \(one($n) | length) times, not once" ]
      + [ $req[] as $n | one($n) | select(length == 1) | .[0] | select(.skipped == true) | "required gate \(.name) was SKIPPED: a skip is not a pass" ]
      + [ $req[] as $n | one($n) | select(length == 1) | .[0] | select(.skipped != true and .passed != true) | "required gate \(.name) FAILED" ]
      + [ $g[] | select((.name as $n | $req | index($n)) == null) | select(.skipped != true and .passed != true) | "gate \(.name) FAILED" ]
    ),
    required_pass: [ $req[] as $n | one($n) | (length == 1 and .[0].passed == true and .[0].skipped != true) ]
  }
JQ
)
JUDGE="$(jq -c --argjson reg "$REGISTERED_JSON" --argjson req "$REQUIRED_JSON" "$JQ_JUDGE" "$QA_RAW_FILE" 2>/dev/null || echo '{"registered":0,"why":["qa output is not valid JSON (see '"$QA_ERR_FILE"')"],"required_pass":[]}')"
GATE_COUNT="$(printf '%s' "$JUDGE" | jq '.registered')"
GATE_BOOLS_JSON="$(printf '%s' "$JUDGE" | jq -c '.required_pass')"
PASS_COUNT="$(printf '%s' "$GATE_BOOLS_JSON" | jq '[.[] | select(. == true)] | length')"
WHY="$(printf '%s' "$JUDGE" | jq -r '.why[]')"

echo "  registered=$GATE_COUNT required=$REQUIRED_GATE_COUNT required_passing=$PASS_COUNT"
echo "  required gates: ${REQUIRED_GATES[*]}"
[ -n "$WHY" ] && printf '  NOT PASS: %s\n' "$WHY" | sed '2,$s/^/  NOT PASS: /'

# --- Step 3: verdict (aggregate-AND) ------------------------------------
if [[ "$QA_EXIT" -eq 0 \
   && -z "$WHY" \
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
# Same SOURCE_DATE_EPOCH convention as START_EPOCH/END_EPOCH above.
DATE_UTC="$(date -u -d "@${SOURCE_DATE_EPOCH:-$(date -u +%s)}" +%Y-%m-%d)"

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
  "required_gates": ${REQUIRED_JSON},
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
    echo "SHIP-006 DISCHARGED (live): all ${REQUIRED_GATE_COUNT} required qa gates executed and passed; no registered gate failed"
else
    echo "SHIP-006 still PARTIAL_ALGORITHM_LEVEL: gate_count=$GATE_COUNT pass_count=$PASS_COUNT exit=$QA_EXIT"
fi

exit "$EXIT_CODE"
