#!/usr/bin/env bash
# ci_run_explicit_test_commands.sh -- run ci/explicit-test-commands.d/ (PMAT-3313).
#
# The workspace-test "Integration tests" step used to be ONE physical ci.yml line,
# `bash -c 'cmd && cmd && ...'` (~4000 chars), and every PR adding a test target
# edited it, so any two such PRs conflicted. A single one-command-per-line file
# only MOVED that lock: two PRs appending at end-of-file still conflict. So the
# commands are FRAGMENTS, the shape of docs/roadmaps/entries/: one file per command,
#
#   ci/explicit-test-commands.d/NNN-<slug>.cmd
#
# NNN is a zero-padded, GAPPED ordinal (010, 020, ...) that fixes execution order;
# a new command takes a free ordinal between two others without renumbering, and
# two PRs adding different files never touch the same path.
#
# Parsing (--list), shared by the CI step and scripts/check_explicit_test_commands.sh:
#   * every entry of the directory must be named ^[0-9]{3}-[a-z0-9-]+\.cmd$
#   * files are read in `LC_ALL=C sort` order
#   * each file holds EXACTLY ONE non-comment, non-blank line (trimmed)
#   * no two files share an ordinal (ambiguous order); no command appears twice
#   * a missing or empty directory is refused -- zero commands is broken wiring
#   Any violation: rc 2, nothing is printed on stdout, nothing runs.
#
# Execution (--run), held equal to the old `&&` chain:
#   * commands in order, each as `bash -c "$cmd"` with stdin from /dev/null (the
#     old docker run had no -i; the list is fully read before anything runs)
#   * a ::group:: header per command
#   * FAIL-FAST: the first non-zero command ends the run with THAT exit status
#
# Usage:
#   scripts/ci_run_explicit_test_commands.sh --list [DIR]
#   scripts/ci_run_explicit_test_commands.sh --run  [DIR]
# DIR defaults to ci/explicit-test-commands.d. The case table for this script
# lives in scripts/check_explicit_test_commands.sh --self-test.
set -euo pipefail

DEFAULT_DIR="ci/explicit-test-commands.d"
NAME_RE='^[0-9]{3}-[a-z0-9-]+\.cmd$'

# one_command FILE -> the file's single command on stdout; rc 1 with a message otherwise.
one_command() {
    local file=$1 raw line n=0 found=""
    while IFS= read -r raw || [ -n "$raw" ]; do
        line="${raw#"${raw%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -n "$line" ] || continue
        case "$line" in \#*) continue ;; esac
        n=$((n + 1)); found=$line
    done < "$file"
    if [ "$n" -ne 1 ]; then
        printf 'REFUSE %s holds %s command line(s); a fragment holds exactly 1\n' "$file" "$n" >&2
        return 1
    fi
    printf '%s\n' "$found"
}

# parse DIR -> the ordered commands on stdout; rc 2 on any violation (stdout then empty).
parse() {
    local dir=$1 base cmd bad=0 ord
    local -a names=() cmds=()
    local -A seen_ord=() seen_cmd=()
    if [ ! -d "$dir" ]; then
        printf 'ENV   %s: no such directory -- refusing to run zero commands\n' "$dir" >&2
        return 2
    fi
    mapfile -t names < <(find "$dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
    if [ "${#names[@]}" -eq 0 ]; then
        printf 'ENV   %s is EMPTY -- zero commands is broken wiring, not a pass\n' "$dir" >&2
        return 2
    fi
    for base in "${names[@]}"; do
        if ! grep -qE "$NAME_RE" <<< "$base" || [ ! -f "$dir/$base" ]; then
            printf 'REFUSE %s/%s: not a regular file named NNN-<slug>.cmd (%s)\n' "$dir" "$base" "$NAME_RE" >&2
            bad=1; continue
        fi
        ord=${base%%-*}
        if [ -n "${seen_ord[$ord]:-}" ]; then
            printf 'REFUSE ordinal %s is shared by %s and %s -- ambiguous order; take a free ordinal\n' "$ord" "${seen_ord[$ord]}" "$base" >&2
            bad=1
        fi
        seen_ord[$ord]=$base
        cmd=$(one_command "$dir/$base") || { bad=1; continue; }
        if [ -n "${seen_cmd[$cmd]:-}" ]; then
            printf 'REFUSE the same command is in %s and %s: %s\n' "${seen_cmd[$cmd]}" "$base" "$cmd" >&2
            bad=1
        fi
        seen_cmd[$cmd]=$base
        cmds+=("$cmd")
    done
    [ "$bad" -eq 0 ] || return 2
    printf '%s\n' "${cmds[@]}"
}

run() {
    local dir=$1 out cmd i=0 total rc
    local -a cmds
    out=$(parse "$dir") || return $?
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
    printf 'PASS  %s/%s explicit test command(s) from %s\n' "$i" "$total" "$dir"
}

case "${1:-}" in
    --list) parse "${2:-$DEFAULT_DIR}" ;;
    --run) run "${2:-$DEFAULT_DIR}" ;;
    -h|--help) sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) printf 'usage: %s --list|--run [DIR]\n' "$0" >&2; exit 2 ;;
esac
