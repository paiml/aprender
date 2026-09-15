#!/usr/bin/env bash
# ci_run_explicit_test_commands.sh -- run ci/explicit-test-commands.txt (PMAT-3313).
#
# The workspace-test "Integration tests" step used to be ONE physical ci.yml line,
# `bash -c 'cmd && cmd && ...'` (~4000 chars). Every PR adding a test target edited
# that line, so any two such PRs conflicted. The commands now live one per line in
# a file; this script is the only thing that turns that file into execution.
#
# Semantics, held equal to the old `&&` chain:
#   * blank lines and `#` comment lines are skipped; surrounding space trimmed
#   * commands run IN FILE ORDER, each as `bash -c "$cmd"`, stdin from /dev/null
#     (the old `docker run` had no -i; and a command reading stdin must never
#     swallow the lines after it -- the file is read fully BEFORE anything runs)
#   * FAIL-FAST: the first non-zero command ends the run with THAT exit status
#   * VACUITY: a missing file, or a file yielding zero commands, exits 2 --
#     an empty list is a broken wiring, never a pass
#
# Usage:
#   scripts/ci_run_explicit_test_commands.sh --list [FILE]   print the parsed commands
#   scripts/ci_run_explicit_test_commands.sh --run  [FILE]   execute them
# FILE defaults to ci/explicit-test-commands.txt. The case table for this script
# lives in scripts/check_explicit_test_commands.sh --self-test.
set -euo pipefail

DEFAULT_FILE="ci/explicit-test-commands.txt"

# parse FILE -> one trimmed command per line on stdout; rc 2 when missing/empty.
parse() {
    local file=$1 raw line n=0
    if [ ! -f "$file" ]; then
        printf 'ENV   %s: no such file -- refusing to run zero commands\n' "$file" >&2
        return 2
    fi
    while IFS= read -r raw || [ -n "$raw" ]; do
        line="${raw#"${raw%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -n "$line" ] || continue
        case "$line" in \#*) continue ;; esac
        printf '%s\n' "$line"
        n=$((n + 1))
    done < "$file"
    if [ "$n" -eq 0 ]; then
        printf 'ENV   %s yielded ZERO commands -- an empty list is broken wiring, not a pass\n' "$file" >&2
        return 2
    fi
}

run() {
    local file=$1 out cmd i=0 total rc
    local -a cmds
    out=$(parse "$file") || return $?
    mapfile -t cmds <<< "$out"
    total=${#cmds[@]}
    for cmd in "${cmds[@]}"; do
        i=$((i + 1))
        printf '::group::[%s/%s] %s\n' "$i" "$total" "$cmd"
        rc=0
        bash -c "$cmd" < /dev/null || rc=$?
        printf '::endgroup::\n'
        if [ "$rc" -ne 0 ]; then
            printf 'FAIL  [%s/%s] exit %s: %s\n' "$i" "$total" "$rc" "$cmd" >&2
            printf '      fail-fast: the %s command(s) after it did not run\n' "$((total - i))" >&2
            return "$rc"
        fi
    done
    printf 'PASS  %s/%s explicit test command(s) from %s\n' "$i" "$total" "$file"
}

case "${1:-}" in
    --list) parse "${2:-$DEFAULT_FILE}" ;;
    --run) run "${2:-$DEFAULT_FILE}" ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) printf 'usage: %s --list|--run [FILE]\n' "$0" >&2; exit 2 ;;
esac
