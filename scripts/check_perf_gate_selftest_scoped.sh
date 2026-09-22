#!/usr/bin/env bash
# check_perf_gate_selftest_scoped.sh -- run perf_gate.sh's case table on a pull
# request ONLY when the change touches something that table reads (#3676).
#
# THE COST. `perf_gate.sh --selftest` is APR-PERF-GATE-001's 29-case table. It
# was an explicit guard-tree step on every PR, merge group and push: 405 s on
# main run 35563537942. Its verdict can only change when perf_gate.sh or a file
# it reads changes; on every other PR it re-proves the same fact about the same
# bytes. Operator 2026-09-21: aprender releases as fast as possible, intel idle.
#
# THE SCOPE IS DERIVED, NOT LISTED. The files the table reads are the
# `$ROOT/<path>` references in perf_gate.sh, plus perf_gate.sh, followed through
# every shell script it references (transitively). A new reference widens the
# scope with no edit here, so the scope cannot silently go
# stale -- the failure a hand-kept path filter has (check_workflow_path_filters.sh
# RULE 2: a gate that runs code must watch the code it runs).
#
# THE DECISION, fail-closed toward RUNNING:
#   the diff vs origin/main (a tree diff, --no-renames: a rename lists both sides)
#   touches a scoped path (the path itself, or anything under a scoped directory)
#       -> run the table; its verdict is this guard's verdict
#   touches none of them -> skip, and SAY so on a SUMMARY line guard_tree.sh
#       surfaces (#3651): which diff, how many scoped paths, where it runs whole
#   origin/main cannot be read, or the diff fails -> RUN (never skip on unknown)
# The table still runs whole every night (.github/workflows/guards-nightly.yml).
#
#   check_perf_gate_selftest_scoped.sh              decide, and run it if in scope
#   check_perf_gate_selftest_scoped.sh --self-test  the decision's case table
#
# THE SMOKE ROW (the ticket's words: "The PR path keeps a smoke row"): on a skip,
# `perf_gate.sh --list-selftests` must still exit 0 and enumerate at least one
# case (109 today, ~60 ms). It proves the table still parses and registers on
# this runner; the full table runs when its inputs change, and nightly.
#
# Test seams: PERF_GATE_ROOT (the tree), PERF_GATE_SUBJECT (the script whose refs
# define the scope and which is run), PERF_GATE_BASE (the comparand, default origin/main).
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" || exit 2
ROOT="${PERF_GATE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}" || exit 2
SUBJECT="${PERF_GATE_SUBJECT:-$ROOT/scripts/perf_gate.sh}"
BASE="${PERF_GATE_BASE:-origin/main}"

# scope_of <subject> -> the repo-relative paths it reads, one per line (itself first).
# TRANSITIVE over shell scripts: a scoped .sh file's own $ROOT/ refs are in scope too
# (perf_gate.sh -> perf_receipt_sign.sh -> whatever that reads). The python libs it
# reads import only the standard library (measured), so .py files end the walk.
scope_of() {
    local s=$1 rel queue next f
    rel="${s#"$ROOT/"}"
    printf '%s\n' "$rel"
    local seen=" $rel "
    queue="$s"
    while [ -n "$queue" ]; do
        next=""
        for f in $queue; do
            [ -f "$f" ] || continue
            while IFS= read -r rel; do
                [ -n "$rel" ] || continue
                case "$seen" in *" $rel "*) continue ;; esac
                seen="$seen$rel "
                printf '%s\n' "$rel"
                case "$rel" in *.sh) next="$next $ROOT/$rel" ;; esac
            done < <(grep -oE '\$ROOT/[A-Za-z0-9_./-]+' "$f" | sed 's|^\$ROOT/||' | sort -u)
        done
        queue="$next"
    done
}

# in_scope <path> <scope-file> -> 0 iff path is a scoped path or under a scoped directory
in_scope() {
    local p=$1 s
    while IFS= read -r s; do
        [ -n "$s" ] || continue
        case "$p" in "$s"|"$s"/*) return 0 ;; esac
    done < "$2"
    return 1
}

rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }

# decide -> prints RUN <why> or SKIP <why>; never fails toward SKIP
decide() {
    local t scope touched p n_scope n_touched
    t=$(mktemp -d "${TMPDIR:-/tmp}/perf-gate-scope.XXXXXX") || { printf 'RUN cannot make a temp dir\n'; return 0; }
    scope_of "$SUBJECT" > "$t/scope"
    n_scope=$(grep -c . "$t/scope")
    if ! git -C "$ROOT" rev-parse -q --verify "$BASE^{tree}" >/dev/null 2>&1; then
        printf 'RUN the comparand %s cannot be read -- never skip on unknown\n' "$BASE"
        rmtree "$t"; return 0
    fi
    if ! git -C "$ROOT" diff --no-renames --name-only "$BASE" HEAD > "$t/touched" 2>/dev/null; then
        printf 'RUN git diff %s HEAD failed -- never skip on unknown\n' "$BASE"
        rmtree "$t"; return 0
    fi
    n_touched=$(grep -c . "$t/touched")
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        if in_scope "$p" "$t/scope"; then
            printf 'RUN the diff vs %s touches %s, which the table reads\n' "$BASE" "$p"
            rmtree "$t"; return 0
        fi
    done < "$t/touched"
    printf 'SKIP the diff vs %s touches %s path(s), none of the %s the table reads; guards-nightly.yml runs it whole\n' "$BASE" "$n_touched" "$n_scope"
    rmtree "$t"; return 0
}

case "${1:-}" in -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== perf_gate --selftest scoping: case table ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/perf-gate-scope-st.XXXXXX") || exit 2
    trap 'rmtree "${d:-}"' EXIT
    bad=0; n=0
    row() { n=$((n + 1)); if [ "$2" = 0 ]; then printf 'ok    row %-2s %s\n' "$n" "$1"; else printf 'FAIL  row %-2s %s -- got: %s\n' "$n" "$1" "$OUT" >&2; bad=1; fi; }
    r="$d/repo"; mkdir -p "$r/scripts/lib" "$r/tests/fixtures/perf-gate" "$r/src"
    git init -q -b main "$r"
    printf '#!/usr/bin/env bash\nROOT=x\n: python3 "$ROOT/scripts/lib/sig.py"\n: cat "$ROOT/tests/fixtures/perf-gate"\n: bash "$ROOT/scripts/helper.sh"\ncase "$1" in --list-selftests) printf "case_a\\ncase_b\\n" ;; --selftest) echo selftest-ran ;; esac\n' > "$r/scripts/perf_gate.sh"
    printf '#!/usr/bin/env bash\ncat "$ROOT/scripts/deep.txt"\n' > "$r/scripts/helper.sh"
    printf 'x\n' > "$r/scripts/lib/sig.py"; printf 'x\n' > "$r/tests/fixtures/perf-gate/a.json"; printf 'x\n' > "$r/src/lib.rs"; mkdir -p "$r/tests/fixtures/perf-gate-old"; printf 'x\n' > "$r/tests/fixtures/perf-gate-old/x.json"; printf 'x\n' > "$r/scripts/deep.txt"
    ( cd "$r" && export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null
      git add -A && git commit -q -m base && git branch base ) || exit 2
    edit() { # edit <path>... -> a fresh commit on main touching exactly those paths, vs branch base
        ( cd "$r" && git checkout -q -B main base && for f in "$@"; do printf '# changed\n' >> "$f"; done \
          && GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t git -c core.hooksPath=/dev/null commit -q -am e ) || exit 2
    }
    judge() { OUT=$(cd "$r" && ROOT="$r" PERF_GATE_SUBJECT="$r/scripts/perf_gate.sh" PERF_GATE_BASE="${1:-base}" bash -c "$(declare -f scope_of in_scope rmtree decide); SUBJECT=\$PERF_GATE_SUBJECT; BASE=\$PERF_GATE_BASE; decide"); }
    edit src/lib.rs;                          judge; [[ $OUT == SKIP* ]];                   row "a diff touching none of the table's inputs -> SKIP, and says where it runs whole" $?
    [[ $OUT == *"none of the 5 the table reads"* ]];                                          row "  ...the scope is DERIVED from the \$ROOT/ refs: 5 paths (itself, a lib, a fixture dir, a helper script and what IT reads)" $?
    edit scripts/deep.txt;                    judge; [[ $OUT == RUN* ]];                    row "a file read only by a HELPER script the table calls -> RUN (the scope is transitive over .sh)" $?
    edit scripts/perf_gate.sh;                judge; [[ $OUT == "RUN "*"scripts/perf_gate.sh"* ]]; row "a diff touching perf_gate.sh itself -> RUN" $?
    edit scripts/lib/sig.py;                  judge; [[ $OUT == RUN* ]];                    row "a diff touching a file the table reads (\$ROOT/scripts/lib/sig.py) -> RUN" $?
    edit tests/fixtures/perf-gate/a.json;     judge; [[ $OUT == RUN* ]];                    row "a diff under a scoped DIRECTORY (tests/fixtures/perf-gate/) -> RUN" $?
    edit tests/fixtures/perf-gate-old/x.json; judge; [[ $OUT == SKIP* ]];                   row "a sibling dir whose name only STARTS like a scoped dir (perf-gate-old/) is not in scope" $?
    edit src/lib.rs;                          judge no-such-ref; [[ $OUT == RUN* ]];        row "the comparand cannot be read -> RUN, never skip on unknown" $?
    ( cd "$r" && git checkout -q -B main base && git mv src/lib.rs tests/fixtures/perf-gate/moved.json && GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t git -c core.hooksPath=/dev/null commit -q -m mv ) || exit 2
    judge; [[ $OUT == RUN* ]];                                                                row "a rename INTO a scoped directory -> RUN" $?
    ( cd "$r" && git checkout -q -B main base && git mv scripts/lib/sig.py src/sig.py && GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t git -c core.hooksPath=/dev/null commit -q -m mv2 ) || exit 2
    judge; [[ $OUT == RUN* ]];                                                                row "a rename OUT of scope still lists the removed side -> RUN (--no-renames)" $?
    # END TO END: the whole guard, not just decide() -- the smoke row on a skip, the table on a RUN
    whole() { OUT=$(cd "$r" && PERF_GATE_ROOT="$r" PERF_GATE_SUBJECT="$r/scripts/perf_gate.sh" PERF_GATE_BASE=base bash "$SELF" 2>&1); RC=$?; }
    edit src/lib.rs; whole
    [ "$RC" = 0 ] && [[ $OUT == *"SUMMARY perf_gate --selftest skipped:"*"smoke: 2 case(s) enumerated"* ]]; row "end to end, out of scope: rc 0 and a SUMMARY line with the SMOKE row (2 cases enumerated)" $?
    edit scripts/perf_gate.sh; whole
    [ "$RC" = 0 ] && [[ $OUT == *"selftest-ran"* ]];                                             row "end to end, in scope: the full table runs" $?
    ( cd "$r" && export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t \
      && git checkout -q -B brk base && printf '#!/usr/bin/env bash\nexit 3\n' > scripts/perf_gate.sh \
      && git -c core.hooksPath=/dev/null commit -q -am brk1 && printf 'y\n' >> src/lib.rs \
      && git -c core.hooksPath=/dev/null commit -q -am brk2 ) || exit 2   # HEAD~1 already broken; HEAD touches only src/
    OUT=$(cd "$r" && PERF_GATE_ROOT="$r" PERF_GATE_SUBJECT="$r/scripts/perf_gate.sh" PERF_GATE_BASE=HEAD~1 bash "$SELF" 2>&1); RC=$?
    [ "$RC" = 1 ] && [[ $OUT == *"FAIL  smoke"* ]];                                              row "end to end, out of scope but the table cannot even enumerate -> the smoke row is RED" $?
    # the real subject's derived scope must include the dirs and files it is known to read
    OUT=$(scope_of "$ROOT/scripts/perf_gate.sh" | tr '\n' ' ')
    [[ $OUT == *"scripts/perf_gate.sh"* && $OUT == *"scripts/lib/receipt_sig.py"* && $OUT == *"tests/fixtures/perf-gate"* ]]
    row "the REAL perf_gate.sh's derived scope names itself, receipt_sig.py and its fixture dir" $?
    [ "$bad" = 0 ] && { printf 'SELF-TEST PASSED: %s rows\n' "$n"; exit 0; }
    printf 'SELF-TEST FAILED\n' >&2; exit 1
fi

echo "=== perf_gate.sh --selftest, path-scoped (check_perf_gate_selftest_scoped.sh) ==="
[ -f "$SUBJECT" ] || { printf 'ENV   %s is missing -- cannot judge, not a pass\n' "$SUBJECT" >&2; exit 2; }
verdict=$(decide)
case "$verdict" in
    SKIP*)
        cases=$(bash "$SUBJECT" --list-selftests 2>/dev/null); lrc=$?
        n_cases=$(grep -c . <<<"$cases")
        if [ "$lrc" -ne 0 ] || [ "$n_cases" -lt 1 ]; then
            printf 'FAIL  smoke: %s --list-selftests exited %s with %s case(s) -- the table does not even enumerate\n' "${SUBJECT#"$ROOT/"}" "$lrc" "$n_cases" >&2
            exit 1
        fi
        printf 'SUMMARY perf_gate --selftest skipped: %s; smoke: %s case(s) enumerated\n' "${verdict#SKIP }" "$n_cases"
        echo PASS; exit 0 ;;
    *)  printf '%s\n' "$verdict"; bash "$SUBJECT" --selftest; exit $? ;;
esac
