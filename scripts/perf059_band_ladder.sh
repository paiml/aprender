#!/usr/bin/env bash
# perf059_band_ladder.sh - PERF-059 / #2785: does batched decode at band c agree
# with the c=1 reference, under a named CUDA-route configuration?
#
# WHY THIS EXISTS. #2785 recorded "the 7B CUDA path produces garbage text" and
# blamed a 7B-specific defect, because the 1.5B looked fine. The 1.5B only ever
# looked fine at c=1. The question a receipt needs answered is not "is the 7B
# broken" but "for THIS host, THIS model and THIS route configuration, which
# bands produce the same text a single request produces" -- because a band whose
# text differs from the c=1 text is measuring the throughput of a different
# computation, and per APR-PERF-GATE-001 I-9 a band may not be re-run to green.
#
# The comparison is against the SAME server's own c=1 output, never against a
# stored golden: a stored golden cannot tell "this build changed" from "this
# route is wrong", and the c=1 reference is exactly the arm the epic's scaling
# formula divides by.
#
# USAGE
#   scripts/perf059_band_ladder.sh --model M.gguf --bands 1,4,8,16 [--port N]
#                                  [--ctx N] [--max-tokens N]
#   scripts/perf059_band_ladder.sh --prove-can-fail
#
# Route configuration is passed through the ENVIRONMENT, so one script covers
# every arm and the arm is recorded in the output rather than hardcoded:
#   APR_STREAM_LEGACY=1     pins the #2767 legacy-vs-non-blocking stream race
#   CUBLAS_GEMM_THRESHOLD=N keeps m>=N off / on the cuBLAS decode route
#   FP8_DECODE=0            keeps m>=5 off the FP8 E4M3 decode route
#
# EXIT 0 only when every requested band matched the c=1 reference. A band that
# diverged, an empty completion, and a server that never came up are all exit 1
# -- there is no path where "I could not measure" reports success.
#
# DELIBERATELY NO `set -e`: every band is run and tallied, the way the CB-006
# probe is. Failures are counted and returned at the end.
set -uo pipefail

MODEL=""
BANDS="1,4,8,16"
PORT=18777
CTX=2048
MAX_TOKENS=40
PROMPT="Write a Python function that returns the sum of a list."
PROVE_CAN_FAIL=0
KEEP_SERVER=0

usage() {
    sed -n '2,30p' "$0"
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --bands) BANDS="${2:-}"; shift 2 ;;
        --port) PORT="${2:-}"; shift 2 ;;
        --ctx) CTX="${2:-}"; shift 2 ;;
        --max-tokens) MAX_TOKENS="${2:-}"; shift 2 ;;
        --prompt) PROMPT="${2:-}"; shift 2 ;;
        --keep-server) KEEP_SERVER=1; shift ;;
        --prove-can-fail) PROVE_CAN_FAIL=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/perf059.XXXXXX")"
# SEC011: the cleanup below is `rm -rf`, so refuse to proceed unless mktemp
# actually produced a directory under a temp root. An empty or `/` value here
# would delete far more than a scratch dir.
if [ -z "$WORKDIR" ] || [ ! -d "$WORKDIR" ]; then
    echo "error: could not create a scratch directory" >&2
    exit 2
fi
if [ "${WORKDIR#/}" = "$WORKDIR" ] || [ "${#WORKDIR}" -lt 12 ]; then
    echo "error: refusing to use scratch directory '$WORKDIR'" >&2
    exit 2
fi
cleanup_workdir() {
    if [ -n "${WORKDIR:-}" ] && [ -d "$WORKDIR" ]; then
        rm -rf "$WORKDIR"
    fi
}
trap cleanup_workdir EXIT

# --------------------------------------------------------------------------
# compare_band: the whole verdict, isolated so --prove-can-fail can exercise it
# without a GPU. Prints one PASS/FAIL line; returns 0 only on PASS.
# --------------------------------------------------------------------------
compare_band() {
    band_label="$1"
    reference="$2"
    shift 2
    n_diff=0
    n_empty=0
    n_total=0
    for observed in "$@"; do
        n_total=$((n_total + 1))
        if [ -z "$observed" ]; then
            n_empty=$((n_empty + 1))
        elif [ "$observed" != "$reference" ]; then
            n_diff=$((n_diff + 1))
        fi
    done
    if [ "$n_total" -eq 0 ]; then
        echo "  FAIL band=${band_label}: no completions returned"
        return 1
    fi
    if [ "$n_diff" -eq 0 ] && [ "$n_empty" -eq 0 ]; then
        echo "  PASS band=${band_label}: ${n_total}/${n_total} match the c=1 reference"
        return 0
    fi
    echo "  FAIL band=${band_label}: ${n_diff}/${n_total} diverged, ${n_empty}/${n_total} empty"
    return 1
}

# --------------------------------------------------------------------------
# --prove-can-fail: a verdict function that cannot report FAIL is theater, and
# this one is the only thing standing between a corrupted band and a receipt.
# Exercise BOTH polarities before trusting any run: identical inputs must PASS,
# and each way of being wrong must FAIL. GPU-free, model-free, seconds.
# --------------------------------------------------------------------------
if [ "$PROVE_CAN_FAIL" -eq 1 ]; then
    echo "PERF-059 harness self-test (no GPU, no model)"
    failures=0

    echo "case 1: all slots identical to the reference -- must PASS"
    if compare_band "self-1" "hello world" "hello world" "hello world"; then
        echo "    ok"
    else
        echo "    BROKEN: harness reported FAIL on identical input"
        failures=$((failures + 1))
    fi

    echo "case 2: one slot diverged -- must FAIL"
    if compare_band "self-2" "hello world" "hello world" "importimportimport"; then
        echo "    BROKEN: harness reported PASS on a diverged slot"
        failures=$((failures + 1))
    else
        echo "    ok"
    fi

    echo "case 3: one slot empty -- must FAIL"
    if compare_band "self-3" "hello world" "hello world" ""; then
        echo "    BROKEN: harness reported PASS on an empty completion"
        failures=$((failures + 1))
    else
        echo "    ok"
    fi

    echo "case 4: no completions at all -- must FAIL"
    if compare_band "self-4" "hello world"; then
        echo "    BROKEN: harness reported PASS on zero completions"
        failures=$((failures + 1))
    else
        echo "    ok"
    fi

    echo "case 5: a prefix of the reference -- must FAIL"
    if compare_band "self-5" "hello world" "hello"; then
        echo "    BROKEN: harness reported PASS on a truncated completion"
        failures=$((failures + 1))
    else
        echo "    ok"
    fi

    if [ "$failures" -eq 0 ]; then
        echo "SELF-TEST PASS: the verdict distinguishes all five cases"
        exit 0
    fi
    echo "SELF-TEST FAIL: ${failures} case(s) the verdict cannot see"
    exit 1
fi

if [ -z "$MODEL" ]; then
    echo "error: --model is required (or use --prove-can-fail)" >&2
    exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    exit 2
fi

# Step 0: pin the binary and prove it came from THIS commit.
#
# A pre-set $APR is honoured, because a CUDA ladder is normally run from a
# build placed in its own --target-dir (the dev shell's `cargo` FUNCTION exports
# one shared CARGO_TARGET_DIR for every worktree of this repo, so a plain
# `cargo build` and a scripted one land in DIFFERENT directories and neither is
# reliably "this worktree's"). Honoured is not trusted: a supplied path must
# still name a binary whose EMBEDDED build SHA is HEAD's, which is the same
# question scripts/apr_bin.sh asks. Unset $APR falls back to that script's full
# search-and-attribute protocol.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${APR:-}" ]; then
    head_sha="$(git -C "$REPO_ROOT" rev-parse --short=9 HEAD 2>/dev/null || true)"
    apr_version="$("$APR" --version 2>/dev/null || true)"
    case "$apr_version" in
        *"$head_sha"*)
            : ;;
        *)
            echo "error: \$APR ($APR) reports '${apr_version:-<no version>}'," >&2
            echo "       which does not carry the HEAD SHA ($head_sha)." >&2
            echo "       Rebuild, or unset APR to use scripts/apr_bin.sh." >&2
            exit 2
            ;;
    esac
else
    # shellcheck source=/dev/null
    . "$REPO_ROOT/scripts/apr_bin.sh" || exit 1
fi

# APR_LAYER_TRACE changes what the decode path does (#2764); a ladder run with
# it set measures the instrumented path, not the shipped one.
if [ -n "${APR_LAYER_TRACE:-}" ]; then
    echo "error: APR_LAYER_TRACE is set (#2764). Unset it and re-run." >&2
    exit 2
fi

echo "PERF-059 band ladder"
echo "  binary : $APR"
echo "  model  : $MODEL"
echo "  ctx    : $CTX   max_tokens: $MAX_TOKENS"
echo "  route  : APR_STREAM_LEGACY=${APR_STREAM_LEGACY:-unset}" \
     "CUBLAS_GEMM_THRESHOLD=${CUBLAS_GEMM_THRESHOLD:-default}" \
     "FP8_DECODE=${FP8_DECODE:-default}"

SERVER_LOG="$WORKDIR/server.log"
"$APR" serve run "$MODEL" --backend cuda --gpu-layers all \
    --port "$PORT" --context-length "$CTX" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup_all() {
    kill "$SERVER_PID" 2>/dev/null || true
    cleanup_workdir
}
trap cleanup_all EXIT

up=0
for _ in $(seq 1 60); do
    if curl -s -m 3 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        up=1
        break
    fi
    sleep 5
done
if [ "$up" -ne 1 ]; then
    echo "FAIL: server did not become ready" >&2
    tail -20 "$SERVER_LOG" >&2
    exit 1
fi

# Prove the accelerator request RESOLVED, rather than trusting --backend cuda.
# `--backend cuda` alone resolves gpu-layers=0 and runs on CPU; a ladder taken
# that way is a CPU measurement wearing a CUDA label.
resolution="$(grep -m1 'gpu-layers:' "$SERVER_LOG" || true)"
echo "  dispatch: ${resolution:-<not reported>}"
case "$resolution" in
    *resolved=0*|"")
        echo "FAIL: GPU layers did not resolve; this would be a CPU run" >&2
        exit 1
        ;;
esac

PROMPT_JSON="$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"

# The server is itself a background child of this shell, so a bare `wait` here
# waits for the SERVER and never returns. Wait on the curl PIDs only.
request_band() {
    n="$1"
    tag="$2"
    pids=()
    i=1
    while [ "$i" -le "$n" ]; do
        curl -s -m 900 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -H 'content-type: application/json' \
            -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":${PROMPT_JSON}}],\"max_tokens\":${MAX_TOKENS},\"temperature\":0}" \
            -o "$WORKDIR/${tag}_${i}.json" &
        pids+=("$!")
        i=$((i + 1))
    done
    wait "${pids[@]}"
}

extract() {
    python3 - "$1" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as fh:
        doc = json.load(fh)
    sys.stdout.write(doc["choices"][0]["message"]["content"])
except Exception:
    sys.stdout.write("")
PYEOF
}

echo "reference: c=1"
request_band 1 ref
REFERENCE="$(extract "$WORKDIR/ref_1.json")"
if [ -z "$REFERENCE" ]; then
    echo "FAIL: the c=1 reference itself is empty; there is nothing to compare to" >&2
    exit 1
fi
printf 'reference text: %s\n' "$(printf '%s' "$REFERENCE" | head -c 120)"

FAILED=0
for band in $(printf '%s' "$BANDS" | tr ',' ' '); do
    request_band "$band" "b${band}"
    # A completion contains newlines of its own (```python blocks), so it can
    # never be recovered by splitting a joined string on newline -- that would
    # turn one divergent completion into several "matching" lines and report
    # PASS. Collect into an array element by element instead.
    OBSERVED=()
    i=1
    while [ "$i" -le "$band" ]; do
        OBSERVED+=("$(extract "$WORKDIR/b${band}_${i}.json")")
        i=$((i + 1))
    done
    if ! compare_band "$band" "$REFERENCE" "${OBSERVED[@]}"; then
        FAILED=$((FAILED + 1))
        printf '    first divergent: %s\n' "$(printf '%s' "${OBSERVED[0]}" | head -c 120)"
    fi
done

echo "batch sizes actually formed:"
grep -oE '\[PMAT-044\] Batch m=[0-9]+' "$SERVER_LOG" | sort | uniq -c | sed 's/^/  /'

if [ "$FAILED" -eq 0 ]; then
    echo "LADDER PASS: every band matched the c=1 reference"
    exit 0
fi
echo "LADDER FAIL: ${FAILED} band(s) diverged from the c=1 reference"
exit 1
