#!/usr/bin/env bash
# Behavioural case table for scripts/complexity_delta_gate.sh (#2766).
#
# Builds a throwaway git repo with calibrated fixtures and drives BOTH gates
# over the same staged states:
#
#   NEW = scripts/complexity_delta_gate.sh
#   OLD = the absolute per-staged-file scan lifted verbatim from the
#         pmat-generated .git/hooks/pre-commit
#
# Every case asserts both verdicts, so the table records not just that the new
# gate is right but exactly which commits the old one froze.
#
# Usage:
#   scripts/tests/complexity_delta_gate_cases.sh
#   scripts/tests/complexity_delta_gate_cases.sh --calibrate   # print fixture metrics

set -uo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
GATE="$ROOT/scripts/complexity_delta_gate.sh"

MAX_CYC=${PMAT_MAX_CYCLOMATIC_COMPLEXITY:-30}
MAX_COG=${PMAT_MAX_COGNITIVE_COMPLEXITY:-25}

command -v pmat >/dev/null 2>&1 || { echo "SKIP: pmat not installed" >&2; exit 0; }
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq not installed" >&2; exit 1; }
[ -x "$GATE" ] || { echo "FAIL: $GATE not executable" >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf -- "$WORK"' EXIT INT TERM HUP
REPO="$WORK/repo"

# ---------------------------------------------------------------- fixtures --
# `nest` is the calibrated over-threshold body: cognitive 29 against a
# threshold of 25, deliberately close to gemm.rs::cublas_prefill_gemm (30).
nest() {
    cat <<EOF
pub fn $1(a: i32, b: i32, c: i32) -> i32 {
    let mut t = 0;
    for i in 0..a {
        if i > b {
            if i > c {
                if i % 2 == 0 {
                    if i % 3 == 0 {
                        t += 1;
                    } else if i % 5 == 0 {
                        t += 2;
                    } else {
                        t += 3;
                    }
                } else if i % 7 == 0 {
                    t += 4;
                } else {
                    t += 5;
                }
            } else if i > 0 {
                t += 6;
            } else {
                t += 7;
            }
        } else if b > c {
            t += 8;
        } else {
            t += 9;
        }
    }
    t
}
EOF
}

# One more nesting level than nest(): strictly worse.
nest_worse() {
    cat <<EOF
pub fn $1(a: i32, b: i32, c: i32) -> i32 {
    let mut t = 0;
    for i in 0..a {
        if a > 0 {
            if i > b {
                if i > c {
                    if i % 2 == 0 {
                        if i % 3 == 0 {
                            t += 1;
                        } else if i % 5 == 0 {
                            t += 2;
                        } else {
                            t += 3;
                        }
                    } else if i % 7 == 0 {
                        t += 4;
                    } else {
                        t += 5;
                    }
                } else if i > 0 {
                    t += 6;
                } else {
                    t += 7;
                }
            } else if b > c {
                t += 8;
            } else {
                t += 9;
            }
        }
    }
    t
}
EOF
}

# The refactor: same behaviour, the nest extracted into a helper, both parts
# under threshold.
nest_refactored() {
    cat <<EOF
pub fn $1(a: i32, b: i32, c: i32) -> i32 {
    let mut t = 0;
    for i in 0..a {
        t += ${1}_step(i, b, c);
    }
    t
}

fn ${1}_step(i: i32, b: i32, c: i32) -> i32 {
    if i <= b {
        return if b > c { 8 } else { 9 };
    }
    if i <= c {
        return if i > 0 { 6 } else { 7 };
    }
    if i % 2 != 0 {
        return if i % 7 == 0 { 4 } else { 5 };
    }
    if i % 3 == 0 {
        return 1;
    }
    if i % 5 == 0 {
        return 2;
    }
    3
}
EOF
}

calm() {
    printf 'pub fn %s(x: i32) -> i32 {\n    if x > 0 { x } else { -x }\n}\n' "$1"
}

# A shallower nest plus N flat top-level ifs: calibrated to land BETWEEN the
# other fixtures (k=8 -> cognitive 27, k=18 -> cognitive 37) so that two
# same-named functions can hold four distinct values.
flat() {
    local name=$1 k=$2 j
    printf 'pub fn %s(a: i32, b: i32, c: i32) -> i32 {\n' "$name"
    printf '    let mut t = 0;\n    for i in 0..a {\n        if i > b {\n'
    printf '            if i > c {\n'
    printf '                if i %% 2 == 0 { t += 1; } else if i %% 7 == 0 { t += 4; } else { t += 5; }\n'
    printf '            } else if i > 0 { t += 6; } else { t += 7; }\n'
    printf '        } else if b > c { t += 8; } else { t += 9; }\n    }\n'
    j=0
    while [ "$j" -lt "$k" ]; do
        printf '    if a == %s { t += 1; }\n' "$j"
        j=$((j + 1))
    done
    printf '    t\n}\n'
}

# pmat reports BOTH of these as the bare name `run` (verified), so the gate
# cannot key on (file, function, rule) alone.
dupfile() {
    printf 'pub mod one {\n'
    "$1" run ${2:-}
    printf '}\n\npub mod two {\n'
    "$3" run ${4:-}
    printf '}\n'
}

build_repo() {
    rm -rf -- "$REPO"
    mkdir -p "$REPO/src"
    git init -q -b main "$REPO"
    git -C "$REPO" config user.email t@example.com
    git -C "$REPO" config user.name Test
    git -C "$REPO" config commit.gpgsign false
    { nest hot; echo; calm calm; } >"$REPO/src/hot.rs"
    { nest alpha; echo; nest beta; } >"$REPO/src/multi.rs"
    { calm tidy; } >"$REPO/src/clean.rs"
    { nest buried; } >"$REPO/src/hot_inc.rs"
    { calm shim; echo; printf 'include!("hot_inc.rs");\n'; } >"$REPO/src/wrapper.rs"
    dupfile nest_worse "" flat 8 >"$REPO/src/dup.rs"
    git -C "$REPO" add -A
    git -C "$REPO" commit -q --no-verify -m "fixtures"
}

# The OLD gate, lifted from the pmat-generated hook: scan every staged source
# file whole, refuse on any violation.
old_gate() {
    local staged f out
    local pathpfx=${1:-}
    staged=$(git -C "$REPO" diff --cached --name-only --diff-filter=ACMR -- '*.rs' | awk 'NR <= 20')
    [ -n "$staged" ] || return 0
    for f in $staged; do
        [ -f "$REPO/$f" ] || continue
        out=$(cd "$REPO" && PATH="${pathpfx:+$pathpfx:}$PATH" NO_COLOR=1 \
            pmat analyze complexity --file "$f" \
            --max-cyclomatic "$MAX_CYC" --max-cognitive "$MAX_COG" 2>&1)
        # herestring, never a pipe: `producer | grep -q` returns 141 on SIGPIPE
        # even when grep MATCHED.
        if grep -qE 'Errors: *[1-9]' <<<"$out"; then return 1; fi
    done
    return 0
}

# ------------------------------------------------------------------- cases --
edit_C1() { { nest_worse hot; echo; calm calm; } >"$REPO/src/hot.rs"; }
edit_C2() { { nest hot; echo; calm calm; echo; nest blazing; } >"$REPO/src/hot.rs"; }
edit_C3() { { nest hot; echo; nest calm; } >"$REPO/src/hot.rs"; }
edit_C4() {
    { echo "// instrumentation for the investigation (#2753)"; nest hot; echo; calm calm; \
        echo; printf 'pub fn probe(x: i32) -> i32 {\n    x + 1\n}\n'; } >"$REPO/src/hot.rs"
}
edit_C5() { { nest_refactored alpha; echo; nest beta; } >"$REPO/src/multi.rs"; }
edit_C6() {
    { calm shim; echo; printf 'pub fn probe(x: i32) -> i32 {\n    x + 1\n}\n'; echo; \
        printf 'include!("hot_inc.rs");\n'; } >"$REPO/src/wrapper.rs"
}
edit_C7() { nest fresh >"$REPO/src/fresh.rs"; }
edit_C8() { { calm tidy; echo; printf 'pub fn probe(x: i32) -> i32 {\n    x + 1\n}\n'; } >"$REPO/src/clean.rs"; }
edit_C9() { nest_worse buried >"$REPO/src/hot_inc.rs"; }
edit_C10() { git -C "$REPO" mv src/hot.rs src/hot_renamed.rs; }
edit_C11() { printf 'notes\n' >"$REPO/README.md"; }
# one::run 39 -> 29 (better) while two::run 27 -> 37 (worse). No single scalar
# per (file, function, rule) can see this: the file max FALLS from 39 to 37.
edit_C12() { dupfile nest "" flat 18 >"$REPO/src/dup.rs"; }
# one::run 39 -> 35 (better), two::run untouched at 27. Correct verdict is
# PERMIT: nothing rose. A gate that compared against the file MINIMUM instead of
# pairing rank-for-rank would refuse this - the refactor #2766 exists to unblock.
edit_C13() { dupfile flat 16 flat 8 >"$REPO/src/dup.rs"; }

# A pmat that cannot produce a report. The gate must go RED, not silently green:
# a gate that cannot measure has not measured.
edit_C14() { { nest hot; echo; calm calm; echo; nest blazing; } >"$REPO/src/hot.rs"; }
stub_C14() {
    local d="$WORK/stub"
    mkdir -p "$d"
    printf '#!/bin/sh\necho "boom: no such subcommand" >&2\nexit 1\n' >"$d/pmat"
    chmod +x "$d/pmat"
    printf '%s' "$d"
}

# id | expect NEW | expect OLD | expected verdict word | description
# The verdict word is asserted present AND its opposite asserted absent, so a
# gate that refuses for the wrong reason is RED too.
CASES='
C1|1|1|WORSE|RATCHET: worsens an already-violating function (cognitive 29 -> 39)
C2|1|1|NEW|RATCHET: adds a NEW over-threshold function to a file that already violates
C3|1|1|NEW|RATCHET: pushes a previously-compliant function over the threshold
C4|0|1||UNFREEZE: edits a violating file without worsening it (the #2766 case)
C5|0|1||UNFREEZE: refactors one of two violations below threshold (the impossible refactor)
C6|0|1||UNFREEZE: edits a file whose only violation is reached through include! (forward_utils)
C7|1|1|NEW|RATCHET: brand-new file containing an over-threshold function
C8|0|0||CONTROL: edits a clean file - both gates permit, discriminates from C4
C9|1|1|WORSE|RATCHET: worsens a violation inside an include!-reached file
C10|0|1||UNFREEZE: renames a violating file with no content change
C11|0|0||CONTROL: no source files staged at all
C12|1|1|WORSE|RATCHET: two functions named `run`; the file max FALLS 39->37 while one of them rises 27->37
C13|0|1||UNFREEZE: two functions named `run`; the worse one is refactored 39->35, the other untouched
C14|1|0|ENVFAIL|ANTI-THEATER: pmat cannot report - the delta gate goes RED, the old gate silently PASSED
'

if [ "${1:-}" = "--calibrate" ]; then
    build_repo
    for f in hot multi clean hot_inc wrapper dup; do
        echo "== src/$f.rs =="
        (cd "$REPO" && NO_COLOR=1 pmat analyze complexity --file "src/$f.rs" \
            --max-cyclomatic "$MAX_CYC" --max-cognitive "$MAX_COG" 2>&1 |
            sed -n '/Functions in File/,$p')
    done
    echo "== nest_worse / nest_refactored =="
    nest_worse hot >"$REPO/src/probe_worse.rs"
    nest_refactored alpha >"$REPO/src/probe_ref.rs"
    for f in probe_worse probe_ref; do
        (cd "$REPO" && NO_COLOR=1 pmat analyze complexity --file "src/$f.rs" \
            --max-cyclomatic "$MAX_CYC" --max-cognitive "$MAX_COG" 2>&1 |
            sed -n '/Functions in File/,$p')
    done
    exit 0
fi

# Self-check: the fixtures must sit where the table assumes, or every verdict
# below is vacuous. A test whose fixture drifted under threshold would pass for
# the wrong reason.
build_repo
cal=$(cd "$REPO" && NO_COLOR=1 pmat analyze complexity --file src/hot.rs \
    --max-cyclomatic "$MAX_CYC" --max-cognitive "$MAX_COG" --format json 2>&1 |
    awk 'f || /^\{/ { f = 1; print }' |
    jq -r '[.violations[] | select(.function == "hot") | .value] | first // 0')
if [ "${cal:-0}" -le "$MAX_COG" ]; then
    echo "FAIL calibration: fixture hot() measures ${cal:-?}, needs > $MAX_COG" >&2
    echo "     the case table would be vacuous; re-calibrate nest()." >&2
    exit 1
fi
echo "calibration: fixture hot() cognitive=$cal > threshold $MAX_COG  ✓"
echo

pass=0
fail=0
while IFS='|' read -r id want_new want_old marker desc; do
    [ -n "$id" ] || continue
    build_repo
    "edit_$id"
    git -C "$REPO" add -A
    new_log="$WORK/$id.new.log"
    pathpfx=""
    if declare -F "stub_$id" >/dev/null 2>&1; then pathpfx=$("stub_$id"); fi
    (cd "$REPO" && PATH="${pathpfx:+$pathpfx:}$PATH" "$GATE") >"$new_log" 2>&1
    got_new=$?
    old_gate "$pathpfx"
    got_old=$?
    [ "$got_new" -eq 0 ] || got_new=1
    why=ok
    case "$marker" in
    NEW | WORSE)
        anti=NEW
        [ "$marker" = NEW ] && anti=WORSE
        grep -q "    $marker " "$new_log" || why="report is missing the '$marker' verdict"
        if grep -q "    $anti " "$new_log"; then why="report refused for the wrong reason ('$anti')"; fi
        ;;
    ENVFAIL)
        grep -q 'ENV-FAIL' "$new_log" || why="report is missing the ENV-FAIL classification"
        ;;
    esac
    if [ "$got_new" -eq "$want_new" ] && [ "$got_old" -eq "$want_old" ] && [ "$why" = ok ]; then
        pass=$((pass + 1))
        printf 'ok   %-4s NEW=%s OLD=%s %-5s %s\n' "$id" "$got_new" "$got_old" "$marker" "$desc"
    else
        fail=$((fail + 1))
        printf 'FAIL %-4s NEW=%s (want %s) OLD=%s (want %s) [%s]  %s\n' \
            "$id" "$got_new" "$want_new" "$got_old" "$want_old" "$why" "$desc"
        sed 's/^/       | /' "$new_log"
    fi
done <<EOF
$CASES
EOF

printf '\ncomplexity delta gate: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
