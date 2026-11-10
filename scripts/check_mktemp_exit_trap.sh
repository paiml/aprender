#!/usr/bin/env bash
# check_mktemp_exit_trap.sh -- a merge-path script that makes scratch files with mktemp
# must remove them on EXIT, including when a step dies mid-function (PMAT-6647, #6647).
#
# WHY. framework16 /tmp reached 0 free inodes (df -i, 2026-11-01 04:16Z) and every
# merge-group run scheduled there failed ENOSPC; the queue ejected #6582 and #6584.
# The hot-path scripts removed their scratch paths on the normal return only. A
# cancelled job, a signal or `set -e` firing between mktemp and rm left them behind.
#
# THE RULE, per file. For every `VAR=$(mktemp ...)` assignment (comments ignored):
#   1. the file installs `trap '... "${SCRATCH[@]}" ...' EXIT` at top level, and
#   2. the file registers that variable: `SCRATCH+=("$VAR")`.
# A file with no mktemp assignment passes vacuously and is reported as such.
#
# usage: check_mktemp_exit_trap.sh [FILE...]   (default: the merge-path hot set)
#        check_mktemp_exit_trap.sh --self-test
# exit:  0 every file obeys the rule; 1 a file breaks it; 2 usage / unreadable file
set -uo pipefail
SCRATCH=(); trap 'rm -rf -- "${SCRATCH[@]}"' EXIT

HOT_PATH=(
    scripts/contracts_gate.sh
    scripts/ci_test_tier.sh
    scripts/ci_mg_reuse.sh
    scripts/ci_self_hosted_preflight.sh
    # per-PR ci.yml helpers that made scratch with no EXIT trap (#6647 slice C)
    scripts/check_ci_fat_secrets_plumbed.sh
    scripts/check_determinism_wait.sh
    scripts/ci/check_sov_advisories_off.sh
    scripts/ci/fetch_p6.sh
    scripts/ci/free_own_target_dir.sh
    scripts/ci/install_pinned_toolchain.sh
)

# check_file FILE -- prints one verdict line per finding; returns 0 clean, 1 dirty, 2 unreadable.
check_file() {
    local f=$1 vars v rc=0
    [ -r "$f" ] || { printf 'ENV   %s: unreadable\n' "$f"; return 2; }
    vars=$(grep -vE '^[[:space:]]*#' "$f" \
        | grep -oE '(^|[[:space:];&|({])[A-Za-z_][A-Za-z0-9_]*="?\$\(mktemp' \
        | sed -E 's/^[[:space:];&|({]//; s/="?\$\(mktemp$//' | sort -u)
    if [ -z "$vars" ]; then printf 'ok    %s: no mktemp assignment\n' "$f"; return 0; fi
    if ! grep -qE '(^|;)[[:space:]]*trap .*"\$\{SCRATCH\[@\]\}".* EXIT([[:space:]]+#.*)?[[:space:]]*$' "$f"; then
        printf 'FAIL  %s: mktemp with no top-level EXIT trap over "${SCRATCH[@]}"\n' "$f"; rc=1
    fi
    for v in $vars; do
        grep -qE "SCRATCH\+=\([^)]*\"\\$\{?$v\}?\"" "$f" \
            || { printf 'FAIL  %s: %s=$(mktemp) is never registered (SCRATCH+=("$%s"))\n' "$f" "$v" "$v"; rc=1; }
    done
    [ "$rc" -ne 0 ] || printf 'ok    %s: %s registered, EXIT trap present\n' "$f" "$(printf '%s' "$vars" | tr '\n' ' ')"
    return "$rc"
}

run() {
    local f rc=0 r
    for f in "$@"; do check_file "$f"; r=$?; [ "$r" -le "$rc" ] || rc=$r; done
    return "$rc"
}

self_test() {
    local T pass=0 fail=0
    T=$(mktemp -d "${TMPDIR:-/tmp}/mktemp-trap-st.XXXXXX") || return 2
    SCRATCH+=("$T")
    good='SCRATCH=(); trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXIT\nf() { d=$(mktemp -d); SCRATCH+=("$d"); }\n'
    # name | want rc | body
    row() {
        printf '%b' "$3" > "$T/$1.sh"
        check_file "$T/$1.sh" >/dev/null; local got=$?
        if [ "$got" -eq "$2" ]; then pass=$((pass + 1)); else fail=$((fail + 1)); printf 'self-test row %s: want %s got %s\n' "$1" "$2" "$got"; fi
    }
    row G1-good 0 "$good"
    row G2-no-mktemp 0 'echo hi\n'
    row G3-quoted-assign 0 'SCRATCH=(); trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXIT\nx="$(mktemp)"; SCRATCH+=("${x}")\n'
    row G5-trailing-comment 0 'SCRATCH=(); trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXIT  # why\nd=$(mktemp); SCRATCH+=("$d")\n'
    row G6-two-in-one-list 0 'SCRATCH=(); trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXIT\na=$(mktemp) && b=$(mktemp); SCRATCH+=("$a" "$b")\n'
    row B7-trap-then-word 1 'trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXITX\nd=$(mktemp); SCRATCH+=("$d")\n'
    row G4-comment-only 0 '  # \x74=$(mktemp) in a comment is not an assignment\n'
    row B1-no-trap 1 'f() { d=$(mktemp -d); SCRATCH+=("$d"); rm -rf "${d:?}"; }\n'
    row B2-unregistered 1 'SCRATCH=(); trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXIT\nd=$(mktemp -d); rm -rf "$d"\n'
    row B3-return-trap-only 1 'f() { td=$(mktemp -d); SCRATCH+=("$td"); trap '\''rm -rf "${SCRATCH[@]}"'\'' RETURN; }\n'
    row B4-one-of-two 1 'SCRATCH=(); trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXIT\na=$(mktemp); SCRATCH+=("$a")\nb=$(mktemp)\n'
    row B5-trap-not-scratch 1 'trap '\''rm -f "$a"'\'' EXIT\na=$(mktemp); SCRATCH+=("$a")\n'
    row B6-prefix-name 1 'SCRATCH=(); trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXIT\nd=$(mktemp); SCRATCH+=("$dd")\n'
    check_file "$T/absent.sh" >/dev/null; [ $? -eq 2 ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo 'self-test row E1-unreadable: want 2'; }
    # Behaviour, not just shape: a GOOD script killed by set -e mid-function leaves no file.
    mkdir "$T/tmp"
    printf '%s\n' 'set -e' 'SCRATCH=(); trap '\''rm -rf -- "${SCRATCH[@]}"'\'' EXIT' \
        'f() { d=$(mktemp -d "$TMPDIR/x.XXXXXX"); SCRATCH+=("$d"); false; rm -rf "${d:?}"; }' 'f' > "$T/kill.sh"
    TMPDIR="$T/tmp" bash "$T/kill.sh" 2>/dev/null
    if [ -z "$(ls -A "$T/tmp")" ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo 'self-test row K1-set-e-mid-function: scratch leaked'; fi
    echo "check_mktemp_exit_trap self-test: $pass passed, $fail failed"
    [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]
}

case "${1-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,18p' "$0"; exit 2 ;;
    "") cd "$(dirname "$0")/.." && run "${HOT_PATH[@]}" ;;
    *) run "$@" ;;
esac
