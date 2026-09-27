#!/usr/bin/env bash
# ttop_soak_gate.sh - headless RSS soak: ttop's steady state must not grow (#4511).
#
# WHY. ttop is the fleet's monitor and the proof of the TUI stack; the operator saw it
# leak. `ttop --soak-frames N` runs the real loop (collector thread, snapshot apply,
# draw, diff) with no terminal and prints {"frame":N,"rss_kib":K} every 10 frames.
# This gate reads that series and fails when RSS after warm-up keeps growing.
#
# RULE. Drop samples before WARMUP frames. growth = median(last 5) - median(first 5)
# of what remains. RED when growth > LIMIT_KIB. Fewer than 10 usable samples, or no
# RSS (non-Linux), is CANNOT MEASURE (rc 2), never green.
#
# Usage:
#   ttop_soak_gate.sh <ttop-binary> [frames]   run the soak and judge it
#   ttop_soak_gate.sh --judge <series.jsonl>   judge a recorded series
#   ttop_soak_gate.sh --self-test              classifier case table (no build)
#   ttop_soak_gate.sh --mutant                 plant a 64 KiB/frame leak in ttop,
#                                              rebuild, and require RED
# Env: TTOP_SOAK_WARMUP (default 100 frames), TTOP_SOAK_LIMIT_KIB (default 8192).
set -euo pipefail

WARMUP="${TTOP_SOAK_WARMUP:-100}"
LIMIT_KIB="${TTOP_SOAK_LIMIT_KIB:-8192}"

# judge <series.jsonl> -> prints a verdict line; rc 0 green, 1 red, 2 cannot measure
judge() {
    awk -v warm="$WARMUP" -v limit="$LIMIT_KIB" '
        function med5(a, off,   i, j, t, b) {
            for (i = 0; i < 5; i++) b[i] = a[off + i]
            for (i = 0; i < 5; i++) for (j = i + 1; j < 5; j++) if (b[j] < b[i]) { t = b[i]; b[i] = b[j]; b[j] = t }
            return b[2]
        }
        match($0, /"frame":[0-9]+/) {
            f = substr($0, RSTART + 8, RLENGTH - 8) + 0
            if (f < warm) next
            if (!match($0, /"rss_kib":[0-9]+/)) next
            s[n++] = substr($0, RSTART + 10, RLENGTH - 10) + 0
        }
        END {
            if (n < 10) { printf "CANNOT MEASURE: %d usable samples after frame %d (need 10)\n", n, warm; exit 2 }
            g = med5(s, n - 5) - med5(s, 0)
            if (g > limit) { printf "RED: RSS grew %d KiB after warm-up (limit %d, %d samples)\n", g, limit, n; exit 1 }
            printf "GREEN: RSS growth %d KiB after warm-up (limit %d, %d samples)\n", g, limit, n
        }' "$1"
}

soak() {
    local bin="$1" frames="${2:-900}" out rc
    out=$(mktemp)
    set +e
    "$bin" --soak-frames "$frames" --refresh 50 > "$out"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then echo "RED: ttop --soak-frames exited $rc"; rm -f -- "${out:?}"; return 1; fi
    set +e
    judge "$out"
    rc=$?
    set -e
    rm -f -- "${out:?}"
    return "$rc"
}

self_test() {
    local d fails=0 rc
    d=$(mktemp -d)
    series() { # frames step base per-frame-growth noise
        awk -v n="$1" -v b="$2" -v g="$3" -v z="$4" 'BEGIN { for (f = 10; f <= n; f += 10) printf "{\"frame\":%d,\"rss_kib\":%d}\n", f, b + g * f + ((f / 10) % 2) * z }'
    }
    row() { # label file want-rc
        set +e; judge "$2" > "$d/out" 2>&1; rc=$?; set -e
        if [ "$rc" -eq "$3" ]; then echo "  ok   $1 (rc=$rc: $(cat "$d/out"))"; else echo "  FAIL $1 rc=$rc want $3: $(cat "$d/out")"; fails=1; fi
    }
    series 900 30000 0 0 > "$d/flat";       row 'flat RSS is green' "$d/flat" 0
    series 900 30000 0 3000 > "$d/noisy";   row 'noise under the limit (3 MiB sawtooth) is green' "$d/noisy" 0
    series 900 30000 64 0 > "$d/leak";      row 'a 64 KiB/frame leak is RED' "$d/leak" 1
    series 900 30000 8 0 > "$d/slow";       row 'an 8 KiB/frame leak (6.4 MiB over 800 frames) under the limit is green' "$d/slow" 0
    series 900 30000 12 0 > "$d/slow2";     row 'a 12 KiB/frame leak (9.6 MiB) is RED' "$d/slow2" 1
    awk 'BEGIN { for (f = 10; f <= 900; f += 10) printf "{\"frame\":%d,\"rss_kib\":%d}\n", f, (f < 100 ? 90000 : 30000) }' > "$d/warm"
    row 'a warm-up spike before frame 100 is ignored' "$d/warm" 0
    series 150 30000 0 0 > "$d/short";      row 'too few samples after warm-up is CANNOT MEASURE' "$d/short" 2
    sed 's/"rss_kib":[0-9]*/"rss_kib":null/' "$d/flat" > "$d/null"; row 'no RSS (non-Linux) is CANNOT MEASURE' "$d/null" 2
    : > "$d/empty";                         row 'empty output is CANNOT MEASURE' "$d/empty" 2
    rm -rf -- "${d:?}"
    if [ "$fails" -eq 0 ]; then echo "ttop_soak_gate self-test: PASS"; else echo "ttop_soak_gate self-test: FAIL"; return 1; fi
}

# The real mutant: plant a per-frame leak in ttop's soak loop, rebuild, require RED.
mutant() {
    local root main anchor bin rc sum
    root=$(git rev-parse --show-toplevel)
    main="$root/crates/aprender-viz-ttop/src/main.rs"
    # the soak loop's per-10-frame report line: the leak goes in once per frame, before it
    anchor='        if frame % 10 == 0 || frame == frames {'
    [ "$(grep -cxF "$anchor" "$main")" -eq 1 ] || { echo "FAIL: mutant anchor not found exactly once in $main"; return 1; }
    sum=$(cksum < "$main")
    cp -- "$main" "$main.orig"
    trap 'mv -f -- "'"$main"'.orig" "'"$main"'"' EXIT
    awk -v a="$anchor" '$0 == a { print "        std::mem::forget(vec![7u8; 64 * 1024]);" } { print }' "$main.orig" > "$main"
    cmp -s -- "$main" "$main.orig" && { echo "FAIL: the mutant changed nothing"; return 1; }
    cargo build -q -p aprender-viz-ttop --bin aprender-viz-ttop
    bin="${CARGO_TARGET_DIR:-$root/target}/debug/aprender-viz-ttop"
    # `soak` re-enables errexit inside itself, so `set +e` around the call does not
    # hold: a RED soak would exit the script before the verdict. Capture with ||.
    rc=0; soak "$bin" 600 || rc=$?
    mv -f -- "$main.orig" "$main"; trap - EXIT
    [ "$(cksum < "$main")" = "$sum" ] || { echo "FAIL: main.rs not restored"; return 1; }
    if [ "$rc" -eq 1 ]; then echo "mutant (64 KiB/frame planted leak): RED as required - killed"; else echo "FAIL: planted leak survived (rc=$rc)"; return 1; fi
}

case "${1:-}" in
    --self-test) self_test ;;
    --judge) judge "$2" ;;
    --mutant) mutant ;;
    '' | -*) echo "usage: $0 <ttop-binary> [frames] | --judge FILE | --self-test | --mutant" >&2; exit 2 ;;
    *) soak "$1" "${2:-900}" ;;
esac
