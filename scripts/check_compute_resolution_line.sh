#!/usr/bin/env bash
# PERF-062 / #2790 — the compute class a receipt records must be the path TAKEN.
#
# WHAT THIS GUARD EXISTS FOR. `scripts/parity_host_receipt.sh` writes
# `provenance.compute_class` from `apr_class_from_log()`, which reads the
# server's own log. Its header says a lane is never labelled by intent. It was:
# every `Backend:` line the engine prints is printed by an ATTEMPT, and the
# reader's `elif grep -qi "wgpu"` arm matched the banner of a wgpu attempt that
# was rejected two lines later. Measured on lambda-vector against apr built
# from a866988e4 with --features cuda, qwen2.5-coder-7b Q4_K_M, `--gpu`: the
# run executed on CPU and the reader returned `wgpu`.
#
# So this is a CASE TABLE, not a review. Every `apr`-log-classification pattern
# in this repo has been wrong at least once, and none of those were caught by
# reading the pattern. The table is run against REAL logs captured from real
# runs (evidence/perf-062/), plus synthetic rows for the shapes a real log
# cannot conveniently produce.
#
# The function under test is EXTRACTED from parity_host_receipt.sh rather than
# copied here. A copy is a second definition, and a second definition is how the
# two band schemas happened (PERF-004).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRODUCER="$ROOT/scripts/parity_host_receipt.sh"
EVIDENCE="$ROOT/evidence/perf-062"
FAIL=0
SELFTEST=0
[ "${1:-}" = "--self-test" ] && SELFTEST=1

[ -f "$PRODUCER" ] || { printf 'FAIL  %s is missing\n' "$PRODUCER" >&2; exit 1; }

if [ "$SELFTEST" -eq 1 ]; then
    # THE COMMITTED DEFECTIVE FIXTURE: apr_class_from_log exactly as it stood at
    # a866988e4. Verbatim, so the self-test proves this table can go RED against
    # the real historical defect rather than against a strawman. If the table
    # ever stops failing here, it has stopped checking anything.
    apr_class_from_log() {
        if grep -qi "CUDA optimized model ready\|Enabling optimized CUDA acceleration" "$1"; then
            echo cuda
        elif grep -qi "metal.*ready\|Metal acceleration" "$1"; then
            echo metal
        elif grep -qi "wgpu" "$1"; then
            echo wgpu
        else
            echo cpu
        fi
    }
    printf 'SELF-TEST: the a866988e4 reader against this table; it MUST fail\n\n'
else
    # Extract the real function, brace-matched from its `() {` to the line that
    # is exactly `}`. Read rather than restated: a copy is a second definition,
    # and a second definition is how the two band schemas happened (PERF-004).
    FN=$(awk '/^apr_class_from_log\(\) \{$/,/^\}$/' "$PRODUCER")
    case "$FN" in
        *apr_class_from_log*) : ;;
        *) printf 'FAIL  apr_class_from_log() not found in %s\n' "$PRODUCER" >&2; exit 1 ;;
    esac
    eval "$FN"
fi

check() { # check <expected> <label> <file>
    local want="$1" label="$2" file="$3" got
    got=$(apr_class_from_log "$file")
    if [ "$got" = "$want" ]; then
        printf '  ok    %-52s -> %s\n' "$label" "$got"
    else
        printf '  FAIL  %-52s -> %s (want %s)\n' "$label" "$got" "$want"
        FAIL=1
    fi
}

WORK=$(mktemp -d)
trap 'rm -rf "${WORK:?}"' EXIT

write() { printf '%s\n' "$2" > "$WORK/$1"; }

printf 'MUST classify as the path TAKEN\n'

# --- the real logs -------------------------------------------------------
#
# A MISSING FIXTURE IS A FAILURE, not a skip. An earlier draft guarded each of
# these with `[ -f ... ] &&`, and the fixtures were named `*.log` -- which
# .gitignore line 38 excludes. The three rows that carry the whole point would
# have been silently absent in CI while the guard reported PASS. They are
# `.stderr.txt` now, and a missing one stops the run.
require() { # require <path>
    [ -f "$1" ] || {
        printf 'FAIL  fixture missing: %s\n' "$1" >&2
        printf '      (check .gitignore -- a fixture the repo does not track is a\n' >&2
        printf '      row this table silently stops checking)\n' >&2
        exit 1
    }
}

BEFORE="$EVIDENCE/apr-run-7b-gpu-BEFORE-a866988e4.stderr.txt"
AFTER="$EVIDENCE/apr-run-7b-gpu-AFTER.stderr.txt"
HONOURED="$EVIDENCE/apr-run-1.5b-gpu-honoured-AFTER.stderr.txt"
require "$BEFORE"; require "$AFTER"; require "$HONOURED"

# The BEFORE log carries no resolution line, so it exercises the banner
# fallback -- which is exactly the arm that returned `wgpu` for a CPU run.
check cpu "real 7B --gpu, F2+wgpu rejected (pre-PERF-062 log)" "$BEFORE"
check cpu "real 7B --gpu, F2+wgpu rejected (resolution line)" "$AFTER"
# THE DISCRIMINATION CASE. Without a row that must come back `cuda`, the whole
# table is satisfied by `echo cpu`.
check cuda "real 1.5B --gpu, CUDA accepted" "$HONOURED"

# --- synthetic rows ------------------------------------------------------
write line_cpu   'apr-compute: requested=accelerator resolved=cpu honoured=false refused=cuda,wgpu'
check cpu "resolution line, resolved=cpu" "$WORK/line_cpu"

write line_cuda  'apr-compute: requested=accelerator resolved=cuda honoured=true refused=-'
check cuda "resolution line, resolved=cuda" "$WORK/line_cuda"

write line_wgpu  'apr-compute: requested=accelerator resolved=wgpu honoured=true refused=cuda'
check wgpu "resolution line, resolved=wgpu" "$WORK/line_wgpu"

write line_metal 'apr-compute: requested=accelerator resolved=metal honoured=true refused=-'
check metal "resolution line, resolved=metal" "$WORK/line_metal"

# THE LAST LINE WINS. One process settles once; an earlier line is a different
# run appended to the same file, and a reader that took the FIRST would report
# the previous lane's class for this one.
printf '%s\n%s\n' \
    'apr-compute: requested=accelerator resolved=cuda honoured=true refused=-' \
    'apr-compute: requested=cpu resolved=cpu honoured=true refused=-' > "$WORK/two"
check cpu "two resolution lines, the LAST wins" "$WORK/two"

# The line OUTRANKS a banner that contradicts it. This is the whole point: the
# banner is an attempt, the line is an outcome.
printf '%s\n%s\n' \
    'CUDA optimized model ready' \
    'apr-compute: requested=accelerator resolved=cpu honoured=false refused=cuda' > "$WORK/banner_vs_line"
check cpu "resolution line beats a contradicting CUDA banner" "$WORK/banner_vs_line"

printf '\nMUST NOT let a REFUSAL spell a class into existence\n'

write rej_wgpu 'warning: GPU (wgpu) path rejected, attempting fallback: cosine vs CPU = 0.722249'
check cpu "a rejected wgpu attempt is not a wgpu run" "$WORK/rej_wgpu"

write unavail 'Backend: CPU (wgpu unavailable: Inference error: wgpu parity gate)'
check cpu "'wgpu unavailable' is not a wgpu run" "$WORK/unavail"

write notavail 'Backend: CPU (wgpu not available)'
check cpu "'wgpu not available' is not a wgpu run" "$WORK/notavail"

write failed 'Backend: CPU (wgpu failed: adapter request)'
check cpu "'wgpu failed' is not a wgpu run" "$WORK/failed"

printf '\nMUST still classify a pre-PERF-062 log by its banners\n'

write old_cuda 'CUDA optimized model ready'
check cuda "legacy log, CUDA banner" "$WORK/old_cuda"

write old_wgpu 'Backend: wgpu (Vulkan)'
check wgpu "legacy log, wgpu banner with no refusal" "$WORK/old_wgpu"

write old_cpu 'Backend: CPU (SIMD-accelerated)'
check cpu "legacy log, CPU banner" "$WORK/old_cpu"

if [ "$SELFTEST" -eq 1 ]; then
    if [ "$FAIL" -eq 0 ]; then
        printf '\nFAIL  SELF-TEST: the historical reader PASSED this table, so the table\n' >&2
        printf '      cannot detect the defect it was written for.\n' >&2
        exit 1
    fi
    printf '\nPASS  SELF-TEST: the table goes RED against the a866988e4 reader\n'
    exit 0
fi

if [ "$FAIL" -ne 0 ]; then
    printf '\nFAIL  apr_class_from_log() mislabels a lane; provenance.compute_class would be fabricated\n' >&2
    exit 1
fi
printf '\nPASS  every row classifies by the path taken\n'
