#!/usr/bin/env bash
# apr_code_edit_verify.sh - does `apr code` finish a scripted edit-and-verify
# task on a real model, on CUDA? (#3719)
#
# One run is one cell: one model on one host. The fixture under
# tests/fixtures/apr-code-edit-verify/ is a small Python project whose
# test_mean fails; the fix is a one-line edit to stats.py. task.txt asks the
# agent to make that fix and run the test.
#
# Nothing the agent SAYS counts as evidence:
#   - the edit is judged by diffing the working copy against the fixture;
#   - the test is re-run HERE, after the agent exits;
#   - "the agent ran the test" is read from logging python3/python shims put
#     first on the agent's PATH (its shell tool runs `sh -c` with the
#     inherited environment). The --emit-trace file cannot answer it: it
#     holds one text block per run and never a tool call
#     (crates/aprender-orchestrate/src/agent/code.rs emit_ccpa_trace), and
#     the PreToolUse/PostToolUse hooks are not wired into the loop;
#   - the backend is read from the serve CHILD's own output. `apr code` has no
#     GPU flag: its driver spawns an `apr serve` child with `--gpu`
#     (crates/aprender-orchestrate/src/agent/driver/apr_serve.rs), and a
#     requested flag proves nothing. The driver pipes the child's output and
#     shows it only when startup fails, so APR_BIN points it at a wrapper that
#     runs the pinned binary and tees the child's stdout and stderr to files.
#
# The judge is scripts/lib/apr_code_edit_verify.py; it writes <out>/cell.json.
#
# Usage:
#   scripts/apr_code_edit_verify.sh --model FILE --host NAME --out DIR
#       [--max-turns N] [--lock-wait SEC] [--timeout SEC]
#   DIR must not exist yet; the run writes every artifact into it.
#
# Exit: 0 PASS; 1 FAIL (cell.json names the first failing mechanism);
#       2 decline (GPU lock not acquired, missing input); 3 usage error.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$ROOT/tests/fixtures/apr-code-edit-verify"
JUDGE="$ROOT/scripts/lib/apr_code_edit_verify.py"
GPU_LOCK="/tmp/apr-gpu.lock"

MODEL=""
HOST=""
OUT=""
MAX_TURNS=12
LOCK_WAIT=3600
TIMEOUT=900

usage() {
    sed -n '/^# Usage:/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 3
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --host) HOST="${2:-}"; shift 2 ;;
        --out) OUT="${2:-}"; shift 2 ;;
        --max-turns) MAX_TURNS="${2:-}"; shift 2 ;;
        --lock-wait) LOCK_WAIT="${2:-}"; shift 2 ;;
        --timeout) TIMEOUT="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$HOST" ] || [ -z "$OUT" ]; then
    usage
fi
if [ ! -f "$MODEL" ]; then
    printf 'DECLINE  model file not found: %s\n' "$MODEL" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    printf 'DECLINE  python3 not found: the fixture test and the judge need it\n' >&2
    exit 2
fi

# shellcheck source=scripts/apr_bin.sh
. "$ROOT/scripts/apr_bin.sh" || exit 2

if [ -e "$OUT" ]; then
    printf 'usage: --out %s already exists; give a fresh directory\n' "$OUT" >&2
    exit 3
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cp -R "$FIXTURE/project" "$OUT/project"

# The wrapper runs the pinned binary for the serve child and records what the
# child prints. tee writes back to the original streams, so the driver sees
# exactly what it would have seen without the wrapper.
wrapper="$OUT/apr-serve-child"
{
    printf '#!/usr/bin/env bash\n'
    printf 'exec %q "$@" > >(tee -a %q) 2> >(tee -a %q >&2)\n' \
        "$APR" "$OUT/serve-child.stdout" "$OUT/serve-child.stderr"
} > "$wrapper"
chmod +x "$wrapper"

# The shims record every python invocation the agent's shell makes (cwd and
# argv), then run the real interpreter. The harness's own re-run below calls
# the real interpreter by path, so it never appears in the log.
real_python="$(command -v python3)"
shim_dir="$OUT/shim"
mkdir -p "$shim_dir"
for name in python3 python; do
    {
        printf '#!/usr/bin/env bash\n'
        printf 'printf "%%s\\t%%s\\n" "$PWD" "$*" >> %q\n' "$OUT/agent-python.log"
        printf 'exec %q "$@"\n' "$real_python"
    } > "$shim_dir/$name"
    chmod +x "$shim_dir/$name"
done
: > "$OUT/agent-python.log"

"$APR" --version > "$OUT/apr-version.txt" 2>&1
sha256sum "$MODEL" | cut -d' ' -f1 > "$OUT/model.sha256"
hostname > "$OUT/hostname.txt"
date -u +%FT%TZ > "$OUT/started.txt"

prompt="$(cat "$FIXTURE/task.txt")"

# Every GPU call takes the fleet lock with a bounded wait and runs choom'd to
# 1000, so this run and never a CI job is the OOM victim. The lock-acquired
# marker separates "never got the GPU" from "apr exited 75".
set +e
(
    cd "$OUT/project" &&
    flock -w "$LOCK_WAIT" -E 75 "$GPU_LOCK" \
        choom -n 1000 -- \
        bash -c 'date -u +%FT%TZ > "$1"; shift; exec "$@"' _ "$OUT/lock-acquired" \
        env APR_BIN="$wrapper" PATH="$shim_dir:$PATH" timeout "$TIMEOUT" \
        "$APR" code -p \
            --model "$MODEL" \
            --project "$OUT/project" \
            --output-format json \
            --emit-trace "$OUT/trace.jsonl" \
            --max-turns "$MAX_TURNS" \
            -- "$prompt"
) > "$OUT/stdout.json" 2> "$OUT/stderr.txt"
rc=$?
set -e
date -u +%FT%TZ > "$OUT/finished.txt"

# The independent re-run: the agent's report of the test is not the test.
set +e
(cd "$OUT/project" && timeout 120 "$real_python" -m unittest test_stats) > "$OUT/unittest.txt" 2>&1
test_rc=$?
set -e

python3 "$JUDGE" \
    --out "$OUT" \
    --fixture "$FIXTURE" \
    --model "$MODEL" \
    --host "$HOST" \
    --rc "$rc" \
    --test-rc "$test_rc" \
    --lock-wait "$LOCK_WAIT" \
    --timeout "$TIMEOUT"
