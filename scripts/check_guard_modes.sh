#!/usr/bin/env bash
#
# check_guard_modes.sh - report every scripts/check_*.sh guard that git does not record as
# executable (mode 100755).
#
# WHY THIS EXISTS
# ---------------
# guard_tree.sh runs every guard as `bash <guard>`, so the executable bit never matters
# there: a guard committed as 100644, or one that lost its bit to a chmod -x, stays green in
# CI, while `./scripts/check_X.sh` fails with "Permission denied" for anyone who runs it as
# its own usage line says. Nothing else reads the mode.
#
# REPORT-ONLY
# -----------
# A run prints one FLAG line per guard whose mode is not 100755, then a count, and exits 0
# whatever it finds (30 guards were 100644 when this landed). It exits 2 when the guard list
# could not be read or is empty: that is not_measured, never a pass. The count line's first
# token is SUMMARY because guard_tree.sh surfaces only UNMEASURED and SUMMARY lines under a
# PASS row; any other line of a passing guard never reaches the CI log.
#
# The list is `git ls-files -s 'scripts/check_*.sh'`, the same pathspec as guard_tree.sh's
# guard_universe, so the two cover the same guards.
#
# Usage: check_guard_modes.sh [--self-test | --mutants | --help]
#   --self-test  run the case table (must-flag 100644 and 120000, must-pass 100755, and a
#                throwaway repository whose modes come from chmod); exit 1 on any failed row.
#                It unsets every variable `git rev-parse --local-env-vars` names (GIT_DIR,
#                GIT_INDEX_FILE, GIT_COMMON_DIR, ...; a git hook exports some), which would
#                otherwise point every row at the caller's repo. Each mode has a fixed row
#                count, and a run with fewer rows is UNMEASURED and fails
#   --mutants    plant each mutant in a copy and require its self-test to fail
set -euo pipefail
# this script's absolute path: the bare-run rows cd before they run it
SELF="$(cd "$(dirname "$0")" && pwd)/${0##*/}"

# flag_modes: stdin is `git ls-files -s` output (MODE SHA STAGE<TAB>PATH); one FLAG line per
# entry whose mode is not 100755
flag_modes() {
    awk -F '\t' '{ split($1, m, " "); if (m[1] != "100755") print "FLAG " m[1] " " $2 }'
}

# report DIR: the report for the git tree at DIR; rc 2 when the list is unread or empty
report() {
    local list n f
    if ! list="$(git -C "$1" ls-files -s -- 'scripts/check_*.sh' 2>/dev/null)"; then
        printf 'not_measured: git ls-files failed in %s\n' "$1"
        return 2
    fi
    n="$(printf '%s' "$list" | grep -c . || true)"
    if [ "$n" -eq 0 ]; then
        printf 'not_measured: no scripts/check_*.sh guard is tracked\n'
        return 2
    fi
    f="$(printf '%s\n' "$list" | flag_modes)"
    [ -z "$f" ] || printf '%s\n' "$f"
    printf 'SUMMARY %s of %s guards are not mode 100755 (report-only)\n' "$(printf '%s' "$f" | grep -c . || true)" "$n"
}

# bare: the report for the work tree around the current directory; rc 2 outside one
bare() {
    local top
    if ! top="$(git rev-parse --show-toplevel 2>/dev/null)"; then
        printf 'not_measured: not inside a git work tree\n'
        return 2
    fi
    report "$top"
}

# case NAME WANT INPUT: WANT is flag or pass, INPUT one `git ls-files -s` line
FAILS=0
ROWS=0
case_row() {
    local got
    ROWS=$((ROWS + 1))
    got="$(printf '%s\n' "$3" | flag_modes)"
    if { [ "$2" = flag ] && [ -n "$got" ]; } || { [ "$2" = pass ] && [ -z "$got" ]; }; then
        printf '  ok    %s\n' "$1"
    else
        printf '  FAIL  %s: want %s, got [%s]\n' "$1" "$2" "$got"
        FAILS=$((FAILS + 1))
    fi
}

# repo_row NAME WANT_RC MUST MUST_NOT: run report on the throwaway repository at $TMP_ST/r
repo_row() {
    local out rc=0
    ROWS=$((ROWS + 1))
    out="$(report "$TMP_ST/r")" || rc=$?
    local ok=1
    [ "$rc" -eq "$2" ] || ok=0
    case "$out" in *"$3"*) ;; *) ok=0 ;; esac
    if [ -n "$4" ]; then
        case "$out" in *"$4"*) ok=0 ;; esac
    fi
    if [ "$ok" -eq 1 ]; then
        printf '  ok    %s\n' "$1"
    else
        printf '  FAIL  %s: rc=%s want %s, output [%s]\n' "$1" "$rc" "$2" "$out"
        FAILS=$((FAILS + 1))
    fi
}

self_test() {
    local t last want
    MODE="${1:-}"
    # every variable git itself treats as repository-local (16 on git 2.43, GIT_COMMON_DIR
    # among them); an empty or short list would leave the rows on the caller's repository
    local -a gv
    local v has_dir=0 has_common=0
    mapfile -t gv < <(git rev-parse --local-env-vars)
    for v in "${gv[@]}"; do
        [ "$v" != GIT_DIR ] || has_dir=1
        [ "$v" != GIT_COMMON_DIR ] || has_common=1
    done
    if [ "$has_dir" -ne 1 ] || [ "$has_common" -ne 1 ]; then
        printf 'UNMEASURED git rev-parse --local-env-vars did not list GIT_DIR and GIT_COMMON_DIR\n'
        return 1
    fi
    unset "${gv[@]}"
    t="$(printf '\t')"
    case_row a_644_guard_is_flagged flag "100644 0123456789abcdef0123456789abcdef01234567 0${t}scripts/check_a.sh"
    case_row a_755_guard_passes pass "100755 0123456789abcdef0123456789abcdef01234567 0${t}scripts/check_a.sh"
    case_row a_symlinked_guard_is_flagged flag "120000 0123456789abcdef0123456789abcdef01234567 0${t}scripts/check_a.sh"
    case_row a_path_with_a_space_keeps_its_name flag "100644 0123456789abcdef0123456789abcdef01234567 0${t}scripts/check_a b.sh"
    TMP_ST="$(mktemp -d)"
    trap 'rm -rf "${TMP_ST:?}"' EXIT
    mkdir -p "$TMP_ST/r/scripts"
    git -C "$TMP_ST/r" init -q
    : > "$TMP_ST/r/scripts/other.sh"
    git -C "$TMP_ST/r" add scripts/other.sh
    repo_row a_tree_without_guards_is_not_measured 2 "not_measured: no scripts/check_" ""
    printf '#!/usr/bin/env bash\n' > "$TMP_ST/r/scripts/check_x.sh"
    printf '#!/usr/bin/env bash\n' > "$TMP_ST/r/scripts/check_y.sh"
    chmod 644 "$TMP_ST/r/scripts/check_x.sh"
    chmod 755 "$TMP_ST/r/scripts/check_y.sh"
    git -C "$TMP_ST/r" add scripts/check_x.sh scripts/check_y.sh
    repo_row a_chmod_minus_x_guard_is_flagged 0 "FLAG 100644 scripts/check_x.sh" "check_y.sh"
    repo_row the_count_names_both_numbers 0 "SUMMARY 1 of 2 guards" ""
    repo_row a_non_guard_is_not_listed 0 "of 2 guards" "other.sh"
    # the bare entry (what guard_tree.sh runs), not only report(), from inside the repository
    child_row a_bare_run_flags_the_644_guard 0 "FLAG 100644 scripts/check_x.sh" \
        bash -c 'cd "$1" && bash "$2"' bare-run "$TMP_ST/r/scripts" "$SELF"
    # outside any work tree the bare entry is not_measured (rc 2), never a pass
    mkdir -p "$TMP_ST/norepo"
    child_row a_bare_run_outside_a_repository_is_not_measured 2 "not_measured: not inside a git work tree" \
        env GIT_CEILING_DIRECTORIES="$TMP_ST" bash -c 'cd "$1" && bash "$2"' bare-run "$TMP_ST/norepo" "$SELF"
    git -C "$TMP_ST/r" update-index --chmod=+x scripts/check_x.sh
    repo_row an_executable_tree_flags_nothing 0 "SUMMARY 0 of 2 guards" "FLAG"
    # guard_tree.sh surfaces a passing guard's line only when its FIRST token is SUMMARY
    ROWS=$((ROWS + 1))
    last="$(report "$TMP_ST/r" | tail -n 1)"
    case "$last" in
        "SUMMARY "*) printf '  ok    the_count_line_starts_with_summary\n' ;;
        *) printf '  FAIL  the_count_line_starts_with_summary: got [%s]\n' "$last"; FAILS=$((FAILS + 1)) ;;
    esac
    # The two rows below rerun this self-test as a child. How deep a run is comes ONLY from
    # its argument (MODE), never from the environment, so no exported variable can make a run
    # skip a row: --nested skips both rows, --env-probe skips only the second, which bounds
    # the recursion at two levels.
    # (1) a pre-commit hook exports GIT_DIR, GIT_WORK_TREE and GIT_INDEX_FILE; the rows must
    # still read their own throwaway repository.
    if [ "$MODE" != --nested ]; then
        git init -q "$TMP_ST/decoy"
        child_row an_exported_git_dir_is_ignored 0 "--- 12/12 rows ---" \
            env GIT_DIR="$TMP_ST/decoy/.git" GIT_WORK_TREE="$TMP_ST/decoy" GIT_INDEX_FILE="$TMP_ST/decoy/.git/index" \
            GIT_COMMON_DIR="$TMP_ST/decoy/.git" \
            bash "$SELF" --self-test --nested
    fi
    # (2) a caller that exports the old nesting variable must still get every row
    if [ -z "$MODE" ]; then
        child_row an_exported_nesting_variable_skips_no_row 0 "--- 13/13 rows ---" \
            env GUARD_MODES_NESTED=1 bash "$SELF" --self-test --env-probe
    fi
    printf -- '--- %s/%s rows ---\n' "$((ROWS - FAILS))" "$ROWS"
    # a row that did not run is not a pass: each mode has a fixed row count
    case "$MODE" in
        --nested) want=12 ;;
        --env-probe) want=13 ;;
        *) want=14 ;;
    esac
    if [ "$ROWS" -ne "$want" ]; then
        printf 'UNMEASURED %s of %s rows ran\n' "$ROWS" "$want"
        return 1
    fi
    [ "$FAILS" -eq 0 ]
}

# child_row NAME WANT_RC MUST CMD...: run CMD as a child; it must exit WANT_RC and print MUST
child_row() {
    local name="$1" want_rc="$2" must="$3" out rc=0
    shift 3
    ROWS=$((ROWS + 1))
    "$@" > "$TMP_ST/child.log" 2>&1 || rc=$?
    out="$(cat "$TMP_ST/child.log")"
    if [ "$rc" = "$want_rc" ]; then
        case "$out" in
            *"$must"*) printf '  ok    %s\n' "$name"; return 0 ;;
        esac
    fi
    printf '  FAIL  %s: rc=%s want %s, [%s] not printed\n' "$name" "$rc" "$want_rc" "$must"
    FAILS=$((FAILS + 1))
}

# one mutant per line: NAME<TAB>sed expression
MUTANTS='m01_755_is_flagged_too	s/if (m\[1\] != "100755")/if (m[1] != "100644")/
m02_an_empty_list_passes	s/if \[ "\$n" -eq 0 \]; then/if false; then/
m03_only_the_first_field_is_the_path	s/print "FLAG " m\[1\] " " \$2/print "FLAG " m[1] " " m[2]/
m04_an_exported_git_dir_is_followed	s/^    unset "\${gv\[@\]}"$/    true/
m05_the_count_line_has_another_first_token	s/^\(    printf .\)SUMMARY %s of/\1guards: SUMMARY %s of/
m06_the_git_env_row_is_skipped	s/^    if \[ "\$MODE" != --nested \]; then$/    if false; then/
m07_the_bare_run_drops_flag_lines	s/^\(    report ".top"\)$/\1 | tail -n 1/
m08_a_failed_toplevel_is_not_checked	s/^    if ! top="\(.*\)"; then$/    top="\1"; if false; then/'

mutants() {
    local name expr copy killed=0 total=0 rc
    copy="$(mktemp)"
    while IFS="$(printf '\t')" read -r name expr; do
        total=$((total + 1))
        sed -e "$expr" "$0" > "$copy"
        if cmp -s "$0" "$copy" || ! bash -n "$copy"; then
            printf '  ERROR %s: the mutant did not apply or does not parse\n' "$name"
            continue
        fi
        rc=0
        bash "$copy" --self-test > /dev/null 2>&1 || rc=$?
        if [ "$rc" -ne 0 ]; then
            printf '  ok    %s killed\n' "$name"
            killed=$((killed + 1))
        else
            printf '  SURVIVED %s\n' "$name"
        fi
    done <<<"$MUTANTS"
    rm -f "${copy:?}"
    printf -- '--- %s/%s mutants killed ---\n' "$killed" "$total"
    [ "$killed" -eq "$total" ]
}

case "${1:-}" in
    --self-test)
        case "${2:-}" in
            '' | --nested | --env-probe) self_test "${2:-}" ;;
            *) printf 'usage: %s --self-test\n' "${0##*/}" >&2; exit 2 ;;
        esac
        ;;
    --mutants) mutants ;;
    --help | -h) sed -n '2,33p' "$0" ;;
    '') bare ;;
    *) printf 'usage: %s [--self-test | --mutants | --help]\n' "${0##*/}" >&2; exit 2 ;;
esac
