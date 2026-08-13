#!/usr/bin/env bash
# ship-007-discharge.sh - LIVE dispatch for FALSIFY-QW2E-SHIP-007
#
# Contract: contracts/qwen2-e2e-verification-v1.yaml (FALSIFY-QW2E-SHIP-007)
# AC: AC-SHIP1-007 - MODEL-1 7B Q4_K decode throughput on RTX 4090
#     must be >= 30.0 tok/s (Ollama-parity-class floor).
#
# Canonical command:
#   apr bench --iterations 5 --max-tokens 128 \
#       paiml/qwen2.5-coder-7b-apache-q4k-v1 --features cuda
#
# Pass criterion: median tok/s across 5 iterations >= 30.0 AND finite.
# Algorithm-level proof: crates/aprender-core/src/bench/ship_007.rs
#   ::verdict_from_decode_tps(measured) -> Pass iff finite && >= 30.0
#
# Usage: bash scripts/ship-discharges/ship-007-discharge.sh \
#            [--apr-binary <path>] [--model <path-or-hf-id>] \
#            [--iterations <n>] [--max-tokens <n>] [--threshold <tps>]
#
# Exit 0 on Pass, 1 on Fail. Writes evidence to
#   evidence/ship-007-full-discharge/discharge-evidence-v1.json

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
ITERATIONS=5
MAX_TOKENS=128
THRESHOLD_TPS="30.0"
EVIDENCE_DIR="evidence/ship-007-full-discharge"
EVIDENCE_FILE="${EVIDENCE_DIR}/discharge-evidence-v1.json"
BENCH_RAW_FILE="${EVIDENCE_DIR}/bench-raw.json"

# --- Arg parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apr-binary)  APR_BINARY="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --iterations)  ITERATIONS="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --threshold)   THRESHOLD_TPS="$2"; shift 2 ;;
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

echo "SHIP-007 dispatch - LIVE discharge"
echo "  apr binary  : $APR_BINARY"
echo "  model       : $MODEL"
echo "  iterations  : $ITERATIONS"
echo "  max-tokens  : $MAX_TOKENS"
echo "  threshold   : ${THRESHOLD_TPS} tok/s"
echo ""

# --- Step 1: dispatch apr bench -----------------------------------------
echo "Step 1: apr bench --iterations $ITERATIONS --max-tokens $MAX_TOKENS $MODEL --json"
START_EPOCH=$(date -u +%s)
BENCH_EXIT=0
"$APR_BINARY" bench \
    --iterations "$ITERATIONS" \
    --max-tokens "$MAX_TOKENS" \
    "$MODEL" \
    --json > "$BENCH_RAW_FILE" 2>&1 || BENCH_EXIT=$?
END_EPOCH=$(date -u +%s)
DURATION_SEC=$(( END_EPOCH - START_EPOCH ))

echo "  raw output -> $BENCH_RAW_FILE (${DURATION_SEC} sec, exit=$BENCH_EXIT)"

# --- Step 2: extract median tok/s ---------------------------------------
echo "Step 2: extract median tok/s"

# Try several known shapes from `apr bench --json`:
#   { "median_tps": 38.4, ... }
#   { "decode": { "median_tps": 38.4 } }
#   { "tps_samples": [37.0, 38.4, 39.1, 38.5, 38.9] }
JQ_QUERY_TPS='( .median_tps // .median_tok_s // .decode.median_tps // (( .tps_samples // .iterations_tps // [] ) as $a | if ($a | length) > 0 then ($a | sort_by(.) | .[($a | length) / 2 | floor]) else null end) // empty ) | tostring'
MEDIAN_TPS="$(jq -r "$JQ_QUERY_TPS" "$BENCH_RAW_FILE" 2>/dev/null || true)"

# Fallback: if iterations are objects with `.tps`, sort-by-tps and pick median.
JQ_QUERY_ITERS='( .iterations // [] ) | if (length) > 0 and (first | type) == "object" then (map(.tps // .tok_s // .throughput // 0) | sort_by(.) | .[(length / 2 | floor)]) else empty end'
if [[ -z "$MEDIAN_TPS" || "$MEDIAN_TPS" == "null" ]]; then
    MEDIAN_TPS="$(jq -r "$JQ_QUERY_ITERS" "$BENCH_RAW_FILE" 2>/dev/null || true)"
fi

if [[ -z "$MEDIAN_TPS" || "$MEDIAN_TPS" == "null" ]]; then
    echo "  FAIL: could not extract median tok/s from $BENCH_RAW_FILE"
    MEDIAN_TPS="0.0"
fi

# Detect non-finite (NaN / Infinity) - verdict_from_decode_tps Fail-closed
IS_FINITE="$(awk -v v="$MEDIAN_TPS" 'BEGIN{
    if (v == "NaN" || v == "nan" || v == "Infinity" || v == "-Infinity" || v == "inf" || v == "-inf") print "false";
    else print "true"
}')"

echo "  median_tps=$MEDIAN_TPS (finite=$IS_FINITE, threshold=$THRESHOLD_TPS)"

# --- Step 3: verdict ----------------------------------------------------
if [[ "$BENCH_EXIT" -eq 0 && "$IS_FINITE" == "true" ]]; then
    PASS_NUM="$(awk -v m="$MEDIAN_TPS" -v t="$THRESHOLD_TPS" 'BEGIN{ print (m+0 >= t+0) ? "true" : "false" }')"
else
    PASS_NUM="false"
fi

if [[ "$PASS_NUM" == "true" ]]; then
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
  "schema_ref": "contracts/qwen2-e2e-verification-v1.yaml#FALSIFY-QW2E-SHIP-007.discharged_evidence",
  "evidence_id": "FALSIFY-SHIP-007-DISCHARGE-DISPATCH-V1",
  "binds_to": "AC-SHIP1-007",
  "falsification_id": "FALSIFY-QW2E-SHIP-007",
  "discharge_date": "${DATE_UTC}",
  "host": {
    "hostname": "${HOSTNAME_VAL}",
    "apr_binary": "${APR_BINARY}",
    "apr_version": "${APR_VERSION}"
  },
  "command": "apr bench --iterations ${ITERATIONS} --max-tokens ${MAX_TOKENS} ${MODEL} --json",
  "model": "${MODEL}",
  "iterations": ${ITERATIONS},
  "max_tokens": ${MAX_TOKENS},
  "raw_bench_output": "${BENCH_RAW_FILE}",
  "duration_seconds": ${DURATION_SEC},
  "apr_bench_exit_code": ${BENCH_EXIT},
  "median_tok_per_sec": ${MEDIAN_TPS},
  "is_finite": ${IS_FINITE},
  "threshold_tok_per_sec": ${THRESHOLD_TPS},
  "verdict_from_decode_tps": "${VERDICT}",
  "overall": "${VERDICT}"
}
JSON

# --- Step 5: report -----------------------------------------------------
echo ""
echo "Verdict: $VERDICT"
echo "Evidence: $EVIDENCE_FILE"

if [[ "$VERDICT" == "PASS" ]]; then
    echo "SHIP-007 DISCHARGED (live): median=${MEDIAN_TPS} tok/s >= ${THRESHOLD_TPS} tok/s"
else
    echo "SHIP-007 still PARTIAL_ALGORITHM_LEVEL: median=${MEDIAN_TPS} tok/s, finite=$IS_FINITE, exit=$BENCH_EXIT"
fi

exit "$EXIT_CODE"
