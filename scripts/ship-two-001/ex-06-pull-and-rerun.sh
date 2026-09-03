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

# This script discharges AC-EX-005/006 by pulling the PUBLISHED model and
# re-running it. Both acceptance criteria are claims about this commit's `apr`,
# so the binary must be this commit's (#2358) - a bare `apr` here would have
# discharged the criteria against whatever was installed.
. "$(dirname "$0")/../apr_bin.sh" || exit 1

MODEL_ID="${MODEL_ID:-paiml/qwen2.5-coder-7b-apache-q4k-v1}"
MANIFEST_DIR="${MANIFEST_DIR:-contracts/publish-manifests}"
MANIFEST_PREFIX="${MANIFEST_PREFIX:-paiml-qwen2.5-coder-7b-apache-q4k-v1}"
EVIDENCE="evidence/ship-two-001/ex-06-pull-rerun.json"
mkdir -p "$(dirname "$EVIDENCE")"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "[EX-06] apr pull $MODEL_ID"
# `apr pull` has no -o flag; it caches into ~/.cache/pacha/models/ and prints
# the path on a `Path: <path>` line. NO_COLOR=1 strips ANSI so awk is clean.
PULL_LOG="$TMPDIR/pull.log"
NO_COLOR=1 "$APR" pull "$MODEL_ID" 2>&1 | tee "$PULL_LOG"
PULLED_PATH=$(awk '/^  *Path:/ {print $2; exit}' "$PULL_LOG")
if [[ -z "$PULLED_PATH" || ! -f "$PULLED_PATH" ]]; then
    echo "ABORT: could not parse pulled path from apr pull output" >&2
    exit 3
fi

# `apr pull` caches files with a hashed stem (e.g. 7bcabb852fedb36b.gguf);
# only the extension survives.  Derive the manifest format from extension:
# .gguf → -gguf.yaml, .apr → -apr.yaml, .safetensors → -safetensors.yaml.
PULLED_EXT="${PULLED_PATH##*.}"
case "$PULLED_EXT" in
    gguf)        MANIFEST_FORMAT=gguf ;;
    apr)         MANIFEST_FORMAT=apr ;;
    safetensors) MANIFEST_FORMAT=safetensors ;;
    *)
        echo "ABORT: unsupported pulled extension '.$PULLED_EXT' (expected .gguf/.apr/.safetensors)" >&2
        exit 4
        ;;
esac
MANIFEST="${MANIFEST:-${MANIFEST_DIR}/${MANIFEST_PREFIX}-${MANIFEST_FORMAT}.yaml}"
echo "[EX-06] pulled format: $MANIFEST_FORMAT → manifest: $MANIFEST"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ABORT: manifest not found: $MANIFEST" >&2
    exit 5
fi

DECLARED_SHA=$(grep '^sha256:' "$MANIFEST" | awk '{print $2}')
COMPUTED_SHA=$(sha256sum "$PULLED_PATH" | awk '{print $1}')

if [[ "$DECLARED_SHA" == "$COMPUTED_SHA" ]]; then
    SHA_VERDICT=PASS
else
    SHA_VERDICT=FAIL
fi

echo "[EX-06] apr run --prompt 'def fib(n):'"
OUTPUT_FILE="$TMPDIR/apr_run.out"
"$APR" run "$PULLED_PATH" --prompt 'def fib(n):' --max-tokens 64 --temperature 0.0 --top-k 1 > "$OUTPUT_FILE" 2>&1 || true

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

# Archive with jq (standard tool). SOURCE_DATE_EPOCH, when set, pins the
# recorded timestamp for a reproducible/test invocation; otherwise this is
# real wall-clock time, as befits an evidence record of a live discharge run.
jq -n \
    --arg ts "$(date -u -d "@${SOURCE_DATE_EPOCH:-$(date +%s)}" -Iseconds)" \
    --arg model "$MODEL_ID" \
    --arg fmt "$MANIFEST_FORMAT" \
    --arg manifest "$MANIFEST" \
    --arg pulled "$PULLED_PATH" \
    --arg sha_v "$SHA_VERDICT" \
    --arg decl "$DECLARED_SHA" \
    --arg comp "$COMPUTED_SHA" \
    --arg py_v "$PY_VERDICT" \
    --arg prompt 'def fib(n):' \
    --arg out "$OUTPUT_HEAD" \
    '{
        timestamp_utc: $ts,
        model_id: $model,
        pulled_format: $fmt,
        pulled_path: $pulled,
        manifest_matched: $manifest,
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
