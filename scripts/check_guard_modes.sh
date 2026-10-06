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
# could not be read or is empty: that is not_measured, never a pass.
#
# The list is `git ls-files -s 'scripts/check_*.sh'`, the same pathspec as guard_tree.sh's
# guard_universe, so the two cover the same guards.
#
# Usage: check_guard_modes.sh [--self-test | --mutants | --help]
#   --self-test  run the case table (must-flag 100644 and 120000, must-pass 100755, and a
#                throwaway repository whose modes come from chmod); exit 1 on any failed row
#   --mutants    plant each mutant in a copy and require its self-test to fail
set -euo pipefail

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
    printf 'REPORT %s of %s guards are not mode 100755 (report-only)\n' "$(printf '%s' "$f" | grep -c . || true)" "$n"
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
    local t
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
    repo_row the_count_names_both_numbers 0 "REPORT 1 of 2 guards" ""
    repo_row a_non_guard_is_not_listed 0 "of 2 guards" "other.sh"
    git -C "$TMP_ST/r" update-index --chmod=+x scripts/check_x.sh
    repo_row an_executable_tree_flags_nothing 0 "REPORT 0 of 2 guards" "FLAG"
    printf -- '--- %s/%s rows ---\n' "$((ROWS - FAILS))" "$ROWS"
    [ "$FAILS" -eq 0 ]
}

# one mutant per line: NAME<TAB>sed expression
MUTANTS='m01_755_is_flagged_too	s/if (m\[1\] != "100755")/if (m[1] != "100644")/
m02_an_empty_list_passes	s/if \[ "\$n" -eq 0 \]; then/if false; then/
m03_only_the_first_field_is_the_path	s/print "FLAG " m\[1\] " " \$2/print "FLAG " m[1] " " m[2]/'

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
    --self-test) self_test ;;
    --mutants) mutants ;;
    --help | -h) sed -n '2,25p' "$0" ;;
    '') report "$(git rev-parse --show-toplevel)" ;;
    *) printf 'usage: %s [--self-test | --mutants | --help]\n' "${0##*/}" >&2; exit 2 ;;
esac
