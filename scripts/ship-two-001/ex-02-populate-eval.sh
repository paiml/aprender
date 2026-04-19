#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SHIP-TWO-001 EX-02 completion — populate eval_results in manifest
# ─────────────────────────────────────────────────────────────
# After eval-pass-at-k.sh emits humaneval_*.json, extract pass_at_1
# and rewrite the manifest's eval_results.humaneval block.
#
# AC-EX-003 falsifier: pass_at_1 < 84.5 (noise band = baseline 85.98 - 1.5)
# ─────────────────────────────────────────────────────────────

set -euo pipefail

EVAL_JSON="${1:?Usage: ex-02-populate-eval.sh <path-to-humaneval_*.json>}"
MANIFEST="${MANIFEST:-contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1.yaml}"

[[ -f "$EVAL_JSON" ]] || { echo "eval json not found: $EVAL_JSON"; exit 1; }
[[ -f "$MANIFEST" ]]  || { echo "manifest not found: $MANIFEST"; exit 1; }

PASS1=$(python3 -c "import json; d=json.load(open('$EVAL_JSON')); print(d['results']['pass_at_1'])")
TS=$(python3 -c "import json; d=json.load(open('$EVAL_JSON')); print(d['timestamp'])")
PASSED=$(python3 -c "import json; d=json.load(open('$EVAL_JSON')); print(d['results']['passed'])")
TOTAL=$(python3 -c "import json; d=json.load(open('$EVAL_JSON')); print(d['results']['total'])")

# AC-EX-003 falsifier check
if python3 -c "import sys; sys.exit(0 if float('$PASS1') >= 84.5 else 1)"; then
    AC_EX_003=PASS
else
    AC_EX_003=FAIL
fi

echo "pass_at_1:    $PASS1"
echo "passed:       $PASSED / $TOTAL"
echo "timestamp:    $TS"
echo "AC-EX-003:    $AC_EX_003 (threshold 84.5)"

# Drift vs 85.98 baseline (§12.5)
DRIFT=$(python3 -c "print(abs(float('$PASS1') - 85.98))")
echo "baseline drift: $DRIFT pp (falsifier if >1.2)"

# Rewrite manifest eval_results block via sed (minimal invasive)
sed -i "s|    pass_at_1: null.*|    pass_at_1: $PASS1  # passed=$PASSED/$TOTAL|" "$MANIFEST"
sed -i "s|    timestamp: null.*|    timestamp: $TS|" "$MANIFEST"

echo ""
echo "manifest updated: $MANIFEST"
grep -A3 'humaneval:' "$MANIFEST" | head -5
