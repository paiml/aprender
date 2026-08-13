#!/usr/bin/env bash
# ship-002-discharge.sh - LIVE dispatch for FALSIFY-QW2E-SHIP-002
#
# Contract: contracts/qwen2-e2e-verification-v1.yaml (FALSIFY-QW2E-SHIP-002)
# AC: AC-SHIP1-002 - MODEL-1 teacher emits syntactically valid Python on
#     canonical `def fib(n):` prompt (zero syntax-error tolerance).
#
# Canonical command:
#   apr run paiml/qwen2.5-coder-7b-apache-q4k-v1.safetensors \
#       --prompt "def fib(n):"
#
# Pass criterion: rustpython/ruff parses completion with zero syntax errors.
# Algorithm-level proof: crates/aprender-core/src/qa/ship_002.rs
#   ::verdict_from_syntax_error_count(0) -> Pass
#
# Usage: bash scripts/ship-discharges/ship-002-discharge.sh \
#            [--apr-binary <path>] [--model <path-or-hf-id>]
#
# Exit 0 on Pass, 1 on Fail. Writes evidence to
#   evidence/ship-002-full-discharge/discharge-evidence-v1.json

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
MODEL="${MODEL:-paiml/qwen2.5-coder-7b-apache-q4k-v1.safetensors}"
PROMPT="def fib(n):"
EVIDENCE_DIR="evidence/ship-002-full-discharge"
EVIDENCE_FILE="${EVIDENCE_DIR}/discharge-evidence-v1.json"
COMPLETION_FILE="${EVIDENCE_DIR}/completion.txt"

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

# --- Preflight ----------------------------------------------------------
if [[ ! -x "$APR_BINARY" ]]; then
    echo "FAIL: apr binary not executable at: $APR_BINARY" >&2
    exit 1
fi

PARSER=""
if command -v ruff >/dev/null 2>&1; then
    PARSER="ruff"
elif command -v rustpython >/dev/null 2>&1; then
    PARSER="rustpython"
elif command -v python3 >/dev/null 2>&1; then
    PARSER="python3"
else
    echo "FAIL: need ruff, rustpython, or python3 to parse completion" >&2
    exit 1
fi

echo "SHIP-002 dispatch - LIVE discharge"
echo "  apr binary : $APR_BINARY"
echo "  model      : $MODEL"
echo "  prompt     : $PROMPT"
echo "  parser     : $PARSER"
echo ""

# --- Step 1: dispatch apr run -------------------------------------------
echo "Step 1: apr run --prompt '$PROMPT'"
START_EPOCH=$(date -u +%s)
RUN_EXIT=0
COMPLETION="$( "$APR_BINARY" run "$MODEL" --prompt "$PROMPT" 2>&1 )" || RUN_EXIT=$?
END_EPOCH=$(date -u +%s)
DURATION_SEC=$(( END_EPOCH - START_EPOCH ))

printf "%s\n" "$COMPLETION" > "$COMPLETION_FILE"
echo "  completion saved -> $COMPLETION_FILE ($DURATION_SEC sec, exit=$RUN_EXIT)"

if [[ "$RUN_EXIT" -ne 0 ]]; then
    echo "WARN: apr run exited non-zero ($RUN_EXIT)" >&2
fi

# --- Step 2: parse completion as Python ---------------------------------
echo "Step 2: parse completion via $PARSER"

# Reconstruct Python: prompt + completion
FULL_PY_FILE="$(mktemp --suffix=.py)"
PARSER_LOG="$(mktemp --suffix=.log)"
trap 'rm -f "$FULL_PY_FILE" "$PARSER_LOG"' EXIT

{
    printf "%s\n" "$PROMPT"
    cat "$COMPLETION_FILE"
} > "$FULL_PY_FILE"

SYNTAX_ERRORS=0
case "$PARSER" in
    ruff)
        # ruff check --select E9 (syntax-only). Any non-zero exit means at
        # least one diagnostic; we promote that to >=1 to honor zero-tolerance.
        if ! ruff check --select E9 --no-cache "$FULL_PY_FILE" > "$PARSER_LOG" 2>&1; then
            SYNTAX_ERRORS="$(grep -c '^' "$PARSER_LOG" || true)"
            if [[ "$SYNTAX_ERRORS" == "0" ]]; then SYNTAX_ERRORS=1; fi
        fi
        ;;
    rustpython)
        if ! rustpython -c "import ast; ast.parse(open('$FULL_PY_FILE').read())" > "$PARSER_LOG" 2>&1; then
            SYNTAX_ERRORS=1
        fi
        ;;
    python3)
        if ! python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$FULL_PY_FILE" > "$PARSER_LOG" 2>&1; then
            SYNTAX_ERRORS=1
        fi
        ;;
esac

echo "  syntax_errors=$SYNTAX_ERRORS (threshold=0)"

# --- Step 3: verdict ----------------------------------------------------
if [[ "$SYNTAX_ERRORS" -eq 0 && "$RUN_EXIT" -eq 0 ]]; then
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
  "schema_ref": "contracts/qwen2-e2e-verification-v1.yaml#FALSIFY-QW2E-SHIP-002.discharged_evidence",
  "evidence_id": "FALSIFY-SHIP-002-DISCHARGE-DISPATCH-V1",
  "binds_to": "AC-SHIP1-002",
  "falsification_id": "FALSIFY-QW2E-SHIP-002",
  "discharge_date": "${DATE_UTC}",
  "host": {
    "hostname": "${HOSTNAME_VAL}",
    "apr_binary": "${APR_BINARY}",
    "apr_version": "${APR_VERSION}"
  },
  "command": "apr run ${MODEL} --prompt \"${PROMPT}\"",
  "model": "${MODEL}",
  "prompt": "${PROMPT}",
  "parser": "${PARSER}",
  "completion_file": "${COMPLETION_FILE}",
  "duration_seconds": ${DURATION_SEC},
  "apr_run_exit_code": ${RUN_EXIT},
  "syntax_errors": ${SYNTAX_ERRORS},
  "syntax_error_threshold": 0,
  "verdict_from_syntax_error_count": "${VERDICT}",
  "overall": "${VERDICT}"
}
JSON

# --- Step 5: report -----------------------------------------------------
echo ""
echo "Verdict: $VERDICT"
echo "Evidence: $EVIDENCE_FILE"

if [[ "$VERDICT" == "PASS" ]]; then
    echo "SHIP-002 DISCHARGED (live): syntax_errors=0"
else
    echo "SHIP-002 still PARTIAL_ALGORITHM_LEVEL: syntax_errors=$SYNTAX_ERRORS or apr-run failed"
fi

exit "$EXIT_CODE"
