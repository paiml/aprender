#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SHIP-TWO-001 EX-06 — Roundtrip via apr pull
# ─────────────────────────────────────────────────────────────
# Spec §12.2 EX-06 — discharges AC-EX-005 (pull + sha match) and
# AC-EX-006 (apr run emits valid Python).
#
# Inputs:
#   MODEL_ID     default paiml/qwen2.5-coder-7b-apache-q4k-v1
#   MANIFEST     default contracts/publish-manifests/.../v1.yaml
# Outputs:
#   evidence/ship-two-001/ex-06-pull-rerun.json
# ─────────────────────────────────────────────────────────────

set -euo pipefail

MODEL_ID="${MODEL_ID:-paiml/qwen2.5-coder-7b-apache-q4k-v1}"
MANIFEST="${MANIFEST:-contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1.yaml}"
EVIDENCE="evidence/ship-two-001/ex-06-pull-rerun.json"
mkdir -p "$(dirname "$EVIDENCE")"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "[EX-06] apr pull $MODEL_ID"
apr pull "$MODEL_ID" -o "$TMPDIR/pulled.apr"

DECLARED_SHA=$(grep '^sha256:' "$MANIFEST" | awk '{print $2}')
COMPUTED_SHA=$(sha256sum "$TMPDIR/pulled.apr" | awk '{print $1}')

if [[ "$DECLARED_SHA" == "$COMPUTED_SHA" ]]; then
    SHA_VERDICT=PASS
else
    SHA_VERDICT=FAIL
fi

echo "[EX-06] apr run --prompt 'def fib(n):'"
OUTPUT_FILE="$TMPDIR/apr_run.out"
apr run "$TMPDIR/pulled.apr" --prompt 'def fib(n):' --max-tokens 64 --temperature 0.0 --top-k 1 > "$OUTPUT_FILE" 2>&1 || true

# AC-EX-006 (spec §12.3 literal): "emits syntactically valid Python"
# We extract the generated body between the "Output:\n" banner and the
# "Completed in …" trailer, then find the longest leading-line prefix that
# parses with `ast.parse`. PASS requires ≥ 1 non-trivial Python statement
# (assignment / function def / class / non-docstring expression) in the
# parsed prefix — this rules out "empty string is valid Python" trivial PASS
# while staying faithful to the spec literal. It does NOT require 'def fib'
# to appear in the completion (Instruct models with raw prompts don't
# reliably autocomplete — the spec does not mandate they do).
PY_CHECK="$TMPDIR/check.py"
cat > "$PY_CHECK" << 'PYEOF'
import ast, sys
raw = sys.stdin.read()
# Extract the generated body: between "Output:\n" banner and "Completed in …" trailer
marker = 'Output:'
idx = raw.find(marker)
body = raw[idx + len(marker):] if idx >= 0 else raw
lines = body.strip().split('\n')
# Drop leading blank lines
while lines and not lines[0].strip():
    lines.pop(0)
# Drop trailing "Completed in …" line and trailing blanks
while lines and (lines[-1].startswith('Completed in') or not lines[-1].strip()):
    lines.pop()
if not lines:
    print('EMPTY')
    sys.exit(3)
# Find longest leading-line prefix that parses as Python
parsed = None
for n in range(len(lines), 0, -1):
    candidate = '\n'.join(lines[:n])
    try:
        parsed = ast.parse(candidate)
        kept = n
        kept_source = candidate
        break
    except SyntaxError:
        continue
if parsed is None:
    print('INVALID: no leading-line prefix parses')
    sys.exit(1)
# Require ≥ 1 non-trivial statement
trivial_exprs = (ast.Constant,)  # bare docstrings / constants
nontrivial = 0
for stmt in parsed.body:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, trivial_exprs):
        continue
    nontrivial += 1
if nontrivial < 1:
    print(f'TRIVIAL: parsed {kept} lines, 0 non-trivial statements')
    sys.exit(4)
print(f'VALID: {kept}/{len(lines)} lines, {nontrivial} non-trivial stmt(s)')
sys.exit(0)
PYEOF

if python3 "$PY_CHECK" < "$OUTPUT_FILE"; then
    PY_VERDICT=PASS
else
    PY_VERDICT=FAIL
fi

OUTPUT_HEAD=$(head -c 500 "$OUTPUT_FILE")

# Archive with jq (standard tool)
jq -n \
    --arg ts "$(date -Iseconds)" \
    --arg model "$MODEL_ID" \
    --arg sha_v "$SHA_VERDICT" \
    --arg decl "$DECLARED_SHA" \
    --arg comp "$COMPUTED_SHA" \
    --arg py_v "$PY_VERDICT" \
    --arg prompt 'def fib(n):' \
    --arg out "$OUTPUT_HEAD" \
    '{
        timestamp_utc: $ts,
        model_id: $model,
        ac_ex_005_sha256: $sha_v,
        ac_ex_005_declared: $decl,
        ac_ex_005_computed: $comp,
        ac_ex_006_python_valid: $py_v,
        prompt: $prompt,
        output_excerpt: $out,
        overall: (if $sha_v == "PASS" and $py_v == "PASS" then "PASS" else "FAIL" end)
    }' > "$EVIDENCE"

echo "[EX-06] sha256:    $SHA_VERDICT"
echo "[EX-06] python ok: $PY_VERDICT"
echo "[EX-06] evidence:  $EVIDENCE"
