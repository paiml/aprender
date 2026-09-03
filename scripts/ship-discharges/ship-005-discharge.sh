#!/usr/bin/env bash
# ship-005-discharge.sh - LIVE dispatch for FALSIFY-QW2E-SHIP-005
#
# Contract: contracts/qwen2-e2e-verification-v1.yaml (FALSIFY-QW2E-SHIP-005)
# AC: AC-SHIP1-005 - MODEL-1 HumanEval pass@1 >= 86.00% nominal,
#     >= 84.80% effective (after 1.2 pp noise allowance).
#
# Canonical command (3 runs, seed=0, take median; subcommand is $APR_SUBCMD
# below):
#   apr <subcommand> --benchmark humaneval paiml/qwen2.5-coder-7b-apache-q4k-v1 \
#       --json --features cuda
#
# Pass criterion: median pass@1 across 3 seed=0 runs >= 86.00 nominal
# (or >= 84.80 within spec's 1.2 pp noise allowance).
# Algorithm-level proof: crates/aprender-core/src/metrics/ship_005.rs
#   ::verdict_from_pass_at_1(correct, total, threshold_pct) -> Pass
#
# Usage: bash scripts/ship-discharges/ship-005-discharge.sh \
#            [--apr-binary <path>] [--model <path-or-hf-id>] \
#            [--threshold <pct>] [--runs <n>]
#
# Exit 0 on Pass, 1 on Fail. Writes evidence to
#   evidence/ship-005-full-discharge/discharge-evidence-v1.json

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
NOMINAL_THRESHOLD="86.00"
EFFECTIVE_THRESHOLD="84.80"  # nominal - 1.2 pp noise allowance
RUNS=3
EVIDENCE_DIR="evidence/ship-005-full-discharge"
EVIDENCE_FILE="${EVIDENCE_DIR}/discharge-evidence-v1.json"

# --- Arg parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apr-binary) APR_BINARY="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --threshold)  NOMINAL_THRESHOLD="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
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

echo "SHIP-005 dispatch - LIVE discharge"
echo "  apr binary       : $APR_BINARY"
echo "  model            : $MODEL"
echo "  runs             : $RUNS (seed=0)"
echo "  nominal threshold: ${NOMINAL_THRESHOLD}%"
echo "  effective floor  : ${EFFECTIVE_THRESHOLD}% (nominal - 1.2 pp noise allowance)"
echo ""

# --- Step 1: run N evals, capture pass@1 from each ----------------------
PASS_AT_1_VALUES=()

APR_SUBCMD="eval"
for ((i=1; i<=RUNS; i++)); do
    LOG_FILE="${EVIDENCE_DIR}/eval-run-${i}.json"
    echo "Run $i/$RUNS: apr ${APR_SUBCMD} --benchmark humaneval $MODEL --json --seed 0"
    # Elapsed-time measurement, not a timestamp: $SECONDS is bash's
    # monotonic seconds-since-shell-start counter, so this never reads the
    # wall clock the way `date -u +%s` twice would.
    RUN_START_SECONDS=$SECONDS
    SUBCMD_EXIT=0
    "$APR_BINARY" "$APR_SUBCMD" --benchmark humaneval "$MODEL" --json --seed 0 >"$LOG_FILE" 2>&1 || SUBCMD_EXIT=$?
    if [[ "$SUBCMD_EXIT" -ne 0 ]]; then
        echo "  WARN: apr ${APR_SUBCMD} exit=$SUBCMD_EXIT on run $i (continuing for evidence)"
    fi
    DURATION=$(( SECONDS - RUN_START_SECONDS ))

    # Extract pass@1 from JSON (try multiple keys; output format may use
    # `pass_at_1`, `pass@1`, or nested `metrics.pass_at_1`).
    JQ_QUERY='(.["pass_at_1"] // .["pass@1"] // .metrics["pass_at_1"] // .metrics["pass@1"] // empty) | tostring'
    PASS_AT_1="$(jq -r "$JQ_QUERY" "$LOG_FILE" 2>/dev/null || true)"

    if [[ -z "$PASS_AT_1" || "$PASS_AT_1" == "null" ]]; then
        echo "  FAIL: could not extract pass@1 from $LOG_FILE"
        PASS_AT_1="0.0"
    fi
    echo "  pass@1=$PASS_AT_1 (run took ${DURATION} sec)"
    PASS_AT_1_VALUES+=("$PASS_AT_1")
done

# --- Step 2: compute median ---------------------------------------------
echo ""
echo "Step 2: compute median across $RUNS runs"

# Sort numerically and pick the middle element.
COUNT="${#PASS_AT_1_VALUES[@]}"
SORTED_LIST="$(printf '%s\n' "${PASS_AT_1_VALUES[@]}" | sort -g)"
MEDIAN_IDX=$(( COUNT / 2 ))
MEDIAN="$(printf '%s\n' "$SORTED_LIST" | sed -n "$((MEDIAN_IDX + 1))p")"

# If pass@1 was reported as a fraction (0.0-1.0), promote to percent.
MEDIAN_PCT="$(awk -v m="$MEDIAN" 'BEGIN{ if (m+0 <= 1.0 && m+0 > 0) print m*100; else print m+0 }')"

echo "  values   : ${PASS_AT_1_VALUES[*]}"
echo "  median   : $MEDIAN (= ${MEDIAN_PCT}%)"

# --- Step 3: verdict ----------------------------------------------------
NOMINAL_PASS="$(awk -v m="$MEDIAN_PCT" -v t="$NOMINAL_THRESHOLD" 'BEGIN{ print (m+0 >= t+0) ? "true" : "false" }')"
EFFECTIVE_PASS="$(awk -v m="$MEDIAN_PCT" -v t="$EFFECTIVE_THRESHOLD" 'BEGIN{ print (m+0 >= t+0) ? "true" : "false" }')"

if [[ "$NOMINAL_PASS" == "true" ]]; then
    VERDICT="PASS"
    BAND="nominal (>=${NOMINAL_THRESHOLD}%)"
    EXIT_CODE=0
elif [[ "$EFFECTIVE_PASS" == "true" ]]; then
    VERDICT="PASS"
    BAND="noise-allowance (>=${EFFECTIVE_THRESHOLD}%, < nominal ${NOMINAL_THRESHOLD}%)"
    EXIT_CODE=0
else
    VERDICT="FAIL"
    BAND="below-floor (< ${EFFECTIVE_THRESHOLD}%)"
    EXIT_CODE=1
fi

# --- Step 4: emit evidence JSON -----------------------------------------
HOSTNAME_VAL="$(hostname)"
APR_VERSION="$( "$APR_BINARY" --version 2>/dev/null || echo "unknown" )"
# Records the real day the discharge ran (audit trail). SOURCE_DATE_EPOCH,
# when a caller sets it (e.g. to reproduce byte-identical evidence in a
# test), pins the date instead of reading the live clock.
DATE_UTC="$(date -u -d "@${SOURCE_DATE_EPOCH:-$(date -u +%s)}" +%Y-%m-%d)"

# JSON-encode the array of pass@1 values
PASS_AT_1_JSON="$(printf '%s\n' "${PASS_AT_1_VALUES[@]}" | jq -R . | jq -s 'map(tonumber)' )"

cat > "$EVIDENCE_FILE" <<JSON
{
  "schema_ref": "contracts/qwen2-e2e-verification-v1.yaml#FALSIFY-QW2E-SHIP-005.discharged_evidence",
  "evidence_id": "FALSIFY-SHIP-005-DISCHARGE-DISPATCH-V1",
  "binds_to": "AC-SHIP1-005",
  "falsification_id": "FALSIFY-QW2E-SHIP-005",
  "discharge_date": "${DATE_UTC}",
  "host": {
    "hostname": "${HOSTNAME_VAL}",
    "apr_binary": "${APR_BINARY}",
    "apr_version": "${APR_VERSION}"
  },
  "command": "apr ${APR_SUBCMD} --benchmark humaneval ${MODEL} --json --seed 0",
  "model": "${MODEL}",
  "runs": ${RUNS},
  "seed": 0,
  "pass_at_1_values": ${PASS_AT_1_JSON},
  "median_pass_at_1_pct": ${MEDIAN_PCT},
  "thresholds": {
    "nominal_pct": ${NOMINAL_THRESHOLD},
    "effective_pct": ${EFFECTIVE_THRESHOLD},
    "noise_allowance_pp": 1.2
  },
  "band": "${BAND}",
  "verdict_from_pass_at_1": "${VERDICT}",
  "overall": "${VERDICT}"
}
JSON

# --- Step 5: report -----------------------------------------------------
echo ""
echo "Verdict: $VERDICT (band=$BAND)"
echo "Evidence: $EVIDENCE_FILE"

if [[ "$VERDICT" == "PASS" ]]; then
    echo "SHIP-005 DISCHARGED (live): median pass@1=${MEDIAN_PCT}%"
else
    echo "SHIP-005 still PARTIAL_ALGORITHM_LEVEL: median pass@1=${MEDIAN_PCT}% < ${EFFECTIVE_THRESHOLD}%"
fi

exit "$EXIT_CODE"
