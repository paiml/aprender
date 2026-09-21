#!/usr/bin/env bash
# check_apr_code_edit_verify.sh - case table for the #3719 `apr code` judge
# (scripts/lib/apr_code_edit_verify.py). No model, no GPU, no inputs.
#
# Each row builds a synthetic artifact directory in the shape
# scripts/apr_code_edit_verify.sh leaves behind and asserts the verdict and
# the FIRST failing mechanism the judge names. Two rows are real failures the
# judge once got wrong:
#
#   zero     - the serve child printed `CUDA optimized model ready` for a
#              model with `Model ready: 0 layers`, then answered HTTP 500
#              (Qwen3.5-4B on lambda, apr 0.69.0 856009cc9, #3571 step 1).
#              The judge called it backend=cuda / "tool call not parsed".
#   no-test  - the --emit-trace file holds one text block and never a tool
#              call, so a judge reading tool_use blocks from it could never
#              pass; "the agent ran the test" now comes from python shims.
#
# Exit 0 = every row got the verdict and mechanism it expects; 1 = a row did
# not (named); 2 = python3 missing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$ROOT/tests/fixtures/apr-code-edit-verify"
JUDGE="$ROOT/scripts/lib/apr_code_edit_verify.py"

if ! command -v python3 >/dev/null 2>&1; then
    printf 'DECLINE  python3 not found\n' >&2
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK:?}"' EXIT

LEGACY=$'Model ready: 36 layers, vocab_size=248320, hidden_dim=2560\ngpu-layers: requested=all resolved=36 total=36 (backend=cuda)\nCUDA optimized model ready'
HYBRID=$'Model ready: Qwen3.5 hybrid, 32 layers resident on the GPU, declared context 262144 tokens\nchat template: Qwen3NoThink (thinking off)'
FIXED_LINE='return sum(values) / len(values)'
BUGGY_LINE='return sum(values) / (len(values) - 1)'
FAILED=0

# make_case NAME CHILD_STDOUT - a run that did everything right
make_case() {
    local d="$WORK/$1"
    mkdir -p "$d"
    cp -R "$FIXTURE/project" "$d/project"
    printf 'abc\n' > "$d/model.sha256"
    printf 'apr 0.69.0 (case)\n' > "$d/apr-version.txt"
    : > "$d/lock-acquired"
    printf 'apr serve ready (3.1s)\n' > "$d/stderr.txt"
    printf '%s\n' "$2" > "$d/serve-child.stdout"
    : > "$d/serve-child.stderr"
    sed -i "s|$BUGGY_LINE|$FIXED_LINE|" "$d/project/stats.py"
    printf '%s\t-m unittest test_stats -v\n' "$d/project" > "$d/agent-python.log"
    set_result "$1" 'Fixed the denominator; all tests pass.'
}

set_result() {
    python3 -c 'import json,sys; print(json.dumps({"type": "result", "subtype": "success", "result": sys.argv[1], "session_id": "case", "duration_ms": 1}))' "$2" > "$WORK/$1/stdout.json"
}

unfix() { cp "$FIXTURE/project/stats.py" "$WORK/$1/project/stats.py"; }

# expect NAME VERDICT MECHANISM_SUBSTRING [APR_RC]
expect() {
    local d="$WORK/$1" test_rc=0 line
    (cd "$d/project" && python3 -m unittest test_stats >/dev/null 2>&1) || test_rc=$?
    line=$(python3 "$JUDGE" --out "$d" --fixture "$FIXTURE" --model /m/Qwen3.5-4B-Q4_K_M.gguf \
        --host case --rc "${4:-0}" --test-rc "$test_rc" --lock-wait 5 --timeout 9 | head -1) || true
    if printf '%s\n' "$line" | grep -q "^$2" && printf '%s\n' "$line" | grep -qF -- "$3"; then
        printf 'ok    %s\n' "$1"
    else
        printf 'FAIL  %s: want %s / %s, got: %s\n' "$1" "$2" "$3" "$line"
        FAILED=1
    fi
}

make_case pass "$LEGACY"
expect pass PASS "task completed"

make_case hybrid "$HYBRID"
printf 'prompt_tokens=4380\n' >> "$WORK/hybrid/serve-child.stderr"
expect hybrid PASS "task completed"
if python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if (c["thinking"], c["context"], c["prompt_tokens"], c["backend"]) == ("off", "4k", 4380, "cuda") else 1)' "$WORK/hybrid/cell.json"; then
    printf 'ok    hybrid row keys: thinking=off context=4k prompt_tokens=4380 backend=cuda\n'
else
    printf 'FAIL  hybrid row keys: %s\n' "$(cat "$WORK/hybrid/cell.json")"
    FAILED=1
fi

make_case pass-keys "$LEGACY"
expect pass-keys PASS "task completed"
if python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if (c["thinking"], c["context"], c["prompt_tokens"]) == ("unknown", "task", None) else 1)' "$WORK/pass-keys/cell.json"; then
    printf 'ok    unmeasured row keys: thinking=unknown context=task prompt_tokens=null\n'
else
    printf 'FAIL  unmeasured row keys: %s\n' "$(cat "$WORK/pass-keys/cell.json")"
    FAILED=1
fi

make_case hybrid-cpu "${HYBRID/GPU/CPU}"
expect hybrid-cpu FAIL "fell back to CPU"

make_case zero $'Model ready: 0 layers, vocab_size=248320, hidden_dim=2560\ngpu-layers: requested=all resolved=0 total=0 (backend=cuda)\nCUDA optimized model ready'
: > "$WORK/zero/stdout.json"
: > "$WORK/zero/agent-python.log"
unfix zero
printf '%s\n' 'Error: driver error: network error: apr serve HTTP 500: {"error":"Model architecture not supported for GPU-resident path"}' >> "$WORK/zero/stderr.txt"
expect zero FAIL "serve child did not load: the child reported ready with 0 layers" 1

make_case cpu "$LEGACY"
printf '[GPU->CPU FALLBACK] out of memory\n' > "$WORK/cpu/serve-child.stderr"
expect cpu FAIL "fell back to CPU: serve child printed a CPU path"

make_case no-load ""
printf 'Error: apr serve exited\n' > "$WORK/no-load/stderr.txt"
expect no-load FAIL "serve child did not load: the driver never reported" 1

make_case partial "${LEGACY/resolved=36/resolved=20}"
expect partial FAIL "never showed every layer resident on CUDA"

make_case refused "$LEGACY"
: > "$WORK/refused/stdout.json"
printf '%s\n' 'Error: driver error: network error: apr serve HTTP 500: {"error":"boom"}' >> "$WORK/refused/stderr.txt"
expect refused FAIL "serve child refused the forward" 1

make_case unparsed "$LEGACY"
unfix unparsed
: > "$WORK/unparsed/agent-python.log"
set_result unparsed $'I will fix it.\n<tool_call>\n{"name": "file_edit", "input": {}}\n</tool_call>'
expect unparsed FAIL "tool call not parsed"

make_case no-edit "$LEGACY"
unfix no-edit
set_result no-edit 'The bug is the denominator.'
expect no-edit FAIL "wrong edit: stats.py was not changed"

make_case test-edited "$LEGACY"
printf '# edited\n' >> "$WORK/test-edited/project/test_stats.py"
expect test-edited FAIL "wrong edit: test_stats.py was modified"

make_case two-lines "$LEGACY"
sed -i 's|"""Small statistics helpers."""|"""Stats."""|' "$WORK/two-lines/project/stats.py"
expect two-lines FAIL "wrong edit: stats.py changed by more than one line"

make_case still-fails "$LEGACY"
unfix still-fails
sed -i 's|(len(values) - 1)|(len(values) + 1)|' "$WORK/still-fails/project/stats.py"
expect still-fails FAIL "independent test re-run still fails"

make_case extra-file "$LEGACY"
: > "$WORK/extra-file/project/notes.txt"
expect extra-file FAIL "wrong edit: files added"

make_case no-test "$LEGACY"
: > "$WORK/no-test/agent-python.log"
expect no-test FAIL "test not run"

make_case answer "$LEGACY"
set_result answer 'I changed a line.'
expect answer FAIL "wrong final answer: the final answer does not report"

make_case no-envelope "$LEGACY"
: > "$WORK/no-envelope/stdout.json"
expect no-envelope FAIL "wrong final answer: no"

make_case lock "$LEGACY"
rm "$WORK/lock/lock-acquired"
expect lock DECLINE "gpu lock not acquired" 75

if [ "$FAILED" -ne 0 ]; then
    printf 'FAIL: the apr code judge named the wrong verdict or mechanism\n'
    exit 1
fi
printf 'OK: every row got its verdict and first failing mechanism\n'
