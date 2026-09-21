#!/usr/bin/env bash
# check_apr_code_edit_verify.sh - case table for the #3719 `apr code` judge
# (scripts/lib/apr_code_edit_verify.py) and harness
# (scripts/apr_code_edit_verify.sh). No model, no GPU, no inputs.
#
# The harness rows (h-*) run the real harness end to end against a fake `apr`
# and a scratch lock, never the fleet lock. aprender-62 (#3712): the release
# ladder calls the harness OUTSIDE its own lock, so "every GPU apr call is
# locked" rests on the harness. The rows prove its apr call runs with the lock
# held and oom_score_adj 1000, in both gate modes, and that a held lock gives a
# DECLINE within the bound, never a hang.
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
    if [[ "$line" == "$2"* && "$line" == *"$3"* ]]; then
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

make_case model-choice "${HYBRID/thinking off/thinking the model\'s choice}"
expect model-choice PASS "task completed"
if python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if (c["thinking"], c["thinking_raw"]) == ("unknown", "the model'"'"'s choice") else 1)' "$WORK/model-choice/cell.json"; then
    printf 'ok    a thinking value other than on/off keys onto no cell: thinking=unknown\n'
else
    printf 'FAIL  model-choice row keys: %s\n' "$(cat "$WORK/model-choice/cell.json")"
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
set_result no-edit ''
printf 'gpu-q: waiting (prio 1, 2/5 in queue)\n%s\n' "$(cat "$WORK/no-edit/stderr.txt")" > "$WORK/no-edit/stderr.txt"
expect no-edit FAIL "wrong edit: stats.py was not changed"
if python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(1 if "gpu-q:" in c["evidence"] else 0)' "$WORK/no-edit/cell.json"; then
    printf 'ok    gpu-q queue lines are not quoted as evidence\n'
else
    printf 'FAIL  no-edit evidence quotes the gpu-q queue: %s\n' "$(cat "$WORK/no-edit/cell.json")"
    FAILED=1
fi

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

# ---- harness rows: the real harness, a fake apr, a scratch lock -------------
HARNESS="$ROOT/scripts/apr_code_edit_verify.sh"
BIN="$WORK/bin"
mkdir -p "$BIN"
HEAD_SHA=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || printf 'nogit')
# The fake answers --version with HEAD's sha (scripts/apr_bin.sh checks it),
# plays the serve child through the harness's APR_BIN wrapper, records whether
# the GPU lock is held and its own oom_score_adj, then does the task right.
cat > "$BIN/fake-apr" <<FAKE
#!/usr/bin/env bash
case "\$1" in
    --version) printf 'apr 0.69.0 (%s)\n' "$HEAD_SHA" ;;
    serve)
        printf 'Model ready: Qwen3.5 hybrid, 32 layers resident on the GPU, declared context 262144 tokens\n'
        printf 'chat template: Qwen3NoThink (thinking off)\n'
        printf 'gpu-layers: requested=all resolved=32 total=32 (backend=cuda)\n' ;;
    code)
        if flock -n "\$APR_GPU_LOCK" true; then held=no; else held=yes; fi
        printf 'held=%s oom=%s\n' "\$held" "\$(cat /proc/self/oom_score_adj)" > "\$FAKE_PROBE"
        "\$APR_BIN" serve --fake-child > /dev/null
        printf 'apr serve ready (0.1s)\n' >&2
        sed -i 's|$BUGGY_LINE|$FIXED_LINE|' stats.py
        python3 -m unittest test_stats > /dev/null 2>&1
        printf '{"type": "result", "subtype": "success", "result": "Fixed the denominator; all tests pass."}\n' ;;
esac
FAKE
# The stub keeps gpu-q v3's contract: `--caps` lists prio and wait, GPUQ_WAIT
# bounds the wait with exit 75, then it takes the lock and choom and runs the
# job. OLD is a gpu-q without `wait`, whose wait has no bound at all.
cat > "$BIN/gpu-q" <<'STUB'
#!/usr/bin/env bash
if [ "${1:-}" = "--caps" ]; then printf 'prio\nwait\n'; exit 0; fi
[ "${1:-}" = "--prio" ] && shift 2
[ "${1:-}" = "--" ] && shift
if [ -n "${GPUQ_WAIT:-}" ]; then
    exec flock -w "$GPUQ_WAIT" -E 75 "${GPUQ_LOCK:?}" choom -n 1000 -- "$@"
fi
exec flock "${GPUQ_LOCK:?}" choom -n 1000 -- "$@"
STUB
OLD="$WORK/old-gpu-q"
mkdir -p "$OLD"
cat > "$OLD/gpu-q" <<'STUB'
#!/usr/bin/env bash
if [ "${1:-}" = "--caps" ]; then exit 2; fi
[ "${1:-}" = "--prio" ] && shift 2
[ "${1:-}" = "--" ] && shift
exec flock "${GPUQ_LOCK:?}" choom -n 1000 -- "$@"
STUB
chmod +x "$BIN/fake-apr" "$BIN/gpu-q" "$OLD/gpu-q"
printf 'not a model\n' > "$WORK/model.gguf"

# harness_row NAME WANT_RC WANT_PROBE MAX_SECONDS [harness args...]
# A harness that hangs (a double lock, a lost bound) is killed and fails the row.
harness_row() {
    local name="$1" want_rc="$2" want_probe="$3" max_s="$4" rc=0 t0 took probe
    shift 4
    t0=$(date +%s)
    (cd "$ROOT" && PATH="${ROW_PATH:-$BIN}:$PATH" APR_BIN="$BIN/fake-apr" APR_GPU_LOCK="$WORK/gpu.lock" \
        FAKE_PROBE="$WORK/$name.probe" \
        timeout "$((max_s + 5))" bash "$HARNESS" --model "$WORK/model.gguf" --host case --out "$WORK/$name" "$@") \
        > "$WORK/$name.log" 2>&1 || rc=$?
    took=$(( $(date +%s) - t0 ))
    probe=$(cat "$WORK/$name.probe" 2>/dev/null || printf 'never-ran')
    if [ "$rc" = "$want_rc" ] && [ "$probe" = "$want_probe" ] && [ "$took" -le "$max_s" ]; then
        printf 'ok    %s (rc=%s, %s, %ss)\n' "$name" "$rc" "$probe" "$took"
    else
        printf 'FAIL  %s: want rc=%s probe=%s within %ss, got rc=%s probe=%s in %ss\n' \
            "$name" "$want_rc" "$want_probe" "$max_s" "$rc" "$probe" "$took"
        sed 's/^/        /' "$WORK/$name.log" | tail -5
        FAILED=1
    fi
}

harness_row h-flock-free 0 "held=yes oom=1000" 60
harness_row h-gpuq-free 0 "held=yes oom=1000" 60 --gpu-q 1
ROW_PATH="$OLD:$BIN" harness_row h-old-gpuq-free 0 "held=yes oom=1000" 60 --gpu-q 1
flock "$WORK/gpu.lock" sleep 60 &
holder=$!
sleep 0.5
harness_row h-flock-held 2 never-ran 30 --lock-wait 2
harness_row h-gpuq-held 2 never-ran 30 --gpu-q 1 --lock-wait 2
ROW_PATH="$OLD:$BIN" harness_row h-old-gpuq-held 2 never-ran 30 --gpu-q 1 --lock-wait 2
kill "$holder" 2>/dev/null || true
wait "$holder" 2>/dev/null || true

if [ "$FAILED" -ne 0 ]; then
    printf 'FAIL: the apr code judge or harness got a row wrong\n'
    exit 1
fi
printf 'OK: every row got its verdict and first failing mechanism\n'
