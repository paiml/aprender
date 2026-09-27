#!/usr/bin/env bash
# ttop_timings_gate.sh - `ttop --timings` reports every phase, and a planted sleep
# shows up in the phase it was planted in (#4511 item 6).
#
# WHY. A timing report that names the wrong phase is worse than none: it sends the
# optimiser to the wrong code. So the gate does not trust the report's shape alone;
# `--mutant` plants a 25 ms sleep inside `layout_render`, rebuilds, and requires the
# report to put it there and nowhere else.
#
# RULES (judge)
#   T1  every frame phase (input apply layout_render diff write -- input only in the
#       tty loop, so the soak requires the other four) and every collector phase
#       (cpu mem process disk net gpu analyzers) has a line with n > 0.
#   T2  each line carries timings, n, p50_us, p99_us, max_us and p50 <= p99 <= max.
#   T3  without --timings the output has NO timings line (off means off).
#
# Usage:
#   ttop_timings_gate.sh <ttop-binary>        run the soak with and without --timings
#   ttop_timings_gate.sh --judge FILE         judge one --timings output
#   ttop_timings_gate.sh --self-test          judge case table (no build)
#   ttop_timings_gate.sh --mutant             plant a 25 ms sleep in layout_render, require it found
set -euo pipefail

PHASES="apply layout_render diff write cpu mem process disk net gpu analyzers"

judge() {
    awk -v need="$PHASES" '
        /"timings":/ {
            if (!match($0, /"timings":"[a-z_]+"/)) { bad = bad " unparsable:" $0; next }
            p = substr($0, RSTART + 11, RLENGTH - 12)
            n = v($0, "n"); a = v($0, "p50_us"); b = v($0, "p99_us"); c = v($0, "max_us")
            if (n == "" || a == "" || b == "" || c == "") { bad = bad " T2-missing-field:" p; next }
            if (a + 0 > b + 0 || b + 0 > c + 0) { bad = bad " T2-order:" p }
            if (n + 0 > 0) seen[p] = 1
        }
        function v(s, k,   r) { if (match(s, "\"" k "\":[0-9]+")) return substr(s, RSTART + length(k) + 3, RLENGTH - length(k) - 3); return "" }
        END {
            split(need, want, " ")
            for (i in want) if (!(want[i] in seen)) bad = bad " T1-missing:" want[i]
            if (bad != "") { print "RED:" bad; exit 1 }
            print "GREEN: every phase reported, fields ordered"
        }' "$1"
}

p50() { sed -n "s/.*\"timings\":\"$2\",\"n\":[0-9]*,\"p50_us\":\([0-9]*\).*/\1/p" "$1"; }

run() {
    local bin="$1" on off rc=0
    on=$(mktemp); off=$(mktemp)
    "$bin" --soak-frames 120 --refresh 50 --timings > "$on"
    "$bin" --soak-frames 20 --refresh 50 > "$off"
    judge "$on" || rc=1
    if grep -q '"timings":' "$off"; then echo "RED T3: timings lines printed without --timings"; rc=1; fi
    grep '"timings":' "$on" | sed 's/^/  /'
    rm -f -- "${on:?}" "${off:?}"
    return "$rc"
}

self_test() {
    local d fails=0 rc all
    d=$(mktemp -d)
    line() { printf '{"timings":"%s","n":%s,"p50_us":%s,"p99_us":%s,"max_us":%s}\n' "$@"; }
    all() { for p in $PHASES; do line "$p" 10 5 9 12; done; }
    row() {
        set +e; judge "$2" > "$d/out" 2>&1; rc=$?; set -e
        if [ "$rc" -eq "$3" ]; then echo "  ok   $1 (rc=$rc)"; else echo "  FAIL $1 rc=$rc want $3: $(cat "$d/out")"; fails=1; fi
    }
    all > "$d/good"; row 'every phase, ordered fields is green' "$d/good" 0
    all | grep -v '"gpu"' > "$d/nogpu"; row 'a missing collector phase is RED (T1)' "$d/nogpu" 1
    all | grep -v '"diff"' > "$d/nodiff"; row 'a missing frame phase is RED (T1)' "$d/nodiff" 1
    { all | grep -v '"net"'; line net 0 0 0 0; } > "$d/zero"; row 'a phase with n=0 is RED (T1)' "$d/zero" 1
    { all | grep -v '"cpu"'; line cpu 10 9 5 12; } > "$d/order"; row 'p50 > p99 is RED (T2)' "$d/order" 1
    { all | grep -v '"mem"'; echo '{"timings":"mem","n":3,"p50_us":1}'; } > "$d/field"; row 'a missing field is RED (T2)' "$d/field" 1
    : > "$d/empty"; row 'no output is RED' "$d/empty" 1
    rm -rf -- "${d:?}"
    if [ "$fails" -eq 0 ]; then echo "ttop_timings_gate self-test: PASS"; else echo "ttop_timings_gate self-test: FAIL"; return 1; fi
}

mutant() {
    local root main anchor bin out lr df rc=0 sum
    root=$(git rev-parse --show-toplevel)
    main="$root/crates/aprender-viz-ttop/src/main.rs"
    # the body of draw_diff's layout_render closure
    anchor='        ui::draw(app, &mut buffer)'
    [ "$(grep -cxF "$anchor" "$main")" -eq 1 ] || { echo "FAIL: mutant anchor not found exactly once in $main"; return 1; }
    sum=$(cksum < "$main")
    cp -- "$main" "$main.orig"
    trap 'mv -f -- "'"$main"'.orig" "'"$main"'"' EXIT
    awk -v a="$anchor" '$0 == a { print "        std::thread::sleep(Duration::from_millis(25));" } { print }' "$main.orig" > "$main"
    cmp -s -- "$main" "$main.orig" && { echo "FAIL: the mutant changed nothing"; return 1; }
    cargo build -q -p aprender-viz-ttop --bin aprender-viz-ttop
    bin="${CARGO_TARGET_DIR:-$root/target}/debug/aprender-viz-ttop"
    out=$(mktemp)
    "$bin" --soak-frames 40 --refresh 50 --timings > "$out"
    mv -f -- "$main.orig" "$main"; trap - EXIT
    [ "$(cksum < "$main")" = "$sum" ] || { echo "FAIL: main.rs not restored"; return 1; }
    lr=$(p50 "$out" layout_render); df=$(p50 "$out" diff)
    rm -f -- "${out:?}"
    echo "  planted 25 ms in layout_render: layout_render p50=${lr:-none}us diff p50=${df:-none}us"
    [ -n "$lr" ] && [ "$lr" -ge 25000 ] || { echo "FAIL: the planted sleep is not in layout_render"; rc=1; }
    [ -n "$df" ] && [ "$df" -lt 25000 ] || { echo "FAIL: the planted sleep leaked into diff"; rc=1; }
    [ "$rc" -eq 0 ] && echo "mutant (25 ms planted in layout_render): found in its own phase - killed"
    return "$rc"
}

case "${1:-}" in
    --self-test) self_test ;;
    --judge) judge "$2" ;;
    --mutant) mutant ;;
    '' | -*) echo "usage: $0 <ttop-binary> | --judge FILE | --self-test | --mutant" >&2; exit 2 ;;
    *) run "$1" ;;
esac
