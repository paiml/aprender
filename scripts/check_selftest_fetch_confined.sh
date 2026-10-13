#!/usr/bin/env bash
# check_selftest_fetch_confined.sh — a self-test that fetches with a depth option is confined to its scratch root (#5025)
#
# WHY. On 2026-10-10 a mutant of a fetch helper dropped `-C "$dir"`; the helper's --self-test ran from the caller's
#   cwd, the shared checkout, so its `--depth=1` fetch wrote `shallow` into the common git dir every worktree reads.
#   A self-test that first `cd`s into its scratch root and exports GIT_CEILING_DIRECTORIES above it turns that same
#   slip into "not a git repository" instead.
#
# THE CHECK. Every tracked scripts/**/*.sh that is a self-test -- it has a `--self-test` mode, or it lives under
#   scripts/tests/ -- and that runs `git ... fetch` with --depth, --deepen or --shallow-* must contain BOTH
#     - an assignment of GIT_CEILING_DIRECTORIES, and
#     - a `cd` command.
#   and the first ceiling assignment must not come before the file's first mktemp (the #5025 slip).
#   It is a file-level check: it cannot prove the `cd` targets the scratch root or runs before the fetch. The
#   runtime half of #5025 (shared_checkout_no_shallow.sh) refuses the write itself whatever a script does.
#
# USAGE
#   bash scripts/check_selftest_fetch_confined.sh [ROOT]   scan ROOT's tracked scripts (default: this checkout)
#   bash scripts/check_selftest_fetch_confined.sh --self-test   the case table, in a fixture repo
#   bash scripts/check_selftest_fetch_confined.sh --mutants     each mutant of this script must turn the table red
# EXIT 0 green · 1 red · 3 a caller error.
set -euo pipefail

SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
DEPTH_FETCH='(^|[^A-Za-z0-9_])fetch([^A-Za-z0-9_][^|;&]*)?--(depth|deepen|shallow-)'
CEILING='GIT_CEILING_DIRECTORIES='
CD_CMD='(^|[;&|({]|then|do|&&)[[:space:]]*cd[[:space:]]'

# verdict FILE PATH -> "skip" | "ok" | "red: <why>"
verdict() {
    local f=$1 p=$2 miss=()
    command grep -Eq -- "$DEPTH_FETCH" "$f" || { echo skip; return; }
    case "$p" in
        scripts/tests/*) : ;;
        *) command grep -Fq -- '--self-test' "$f" || { echo skip; return; } ;;
    esac
    command grep -Fq -- "$CEILING" "$f" || miss+=("no $CEILING")
    command grep -Eq -- "$CD_CMD" "$f" || miss+=("no cd")
    # #5025 slipped past the two greps above: its confinement line sat at line 1, above the shebang and the
    # mktemp that sets its scratch var, where `cd -- ""` is a silent no-op. The ceiling must follow a mktemp.
    local cl ml
    cl="$(awk -v p="$CEILING" 'index($0, p) { print NR; exit }' "$f")"
    ml="$(awk '/mktemp/ { print NR; exit }' "$f")"
    if [ -n "$cl" ] && [ -n "$ml" ] && [ "$cl" -lt "$ml" ]; then miss+=("$CEILING above the first mktemp"); fi
    if [ "${#miss[@]}" -eq 0 ]; then echo ok; else echo "red: ${miss[*]}"; fi
}

scan() {
    local root=$1 n=0 red=0 p v
    [ -d "$root" ] || { echo "check_selftest_fetch_confined: no directory $root" >&2; return 3; }
    while IFS= read -r p; do
        v="$(verdict "$root/$p" "$p")"
        case "$v" in
            skip) continue ;;
            ok) n=$((n + 1)) ;;
            *) n=$((n + 1)); red=$((red + 1)); echo "RED: $p -- a self-test that fetches with a depth option, ${v#red: }" ;;
        esac
    done < <(git -C "$root" ls-files -- 'scripts/*.sh')
    if [ "$red" -gt 0 ]; then
        echo "RED: $red of $n depth-fetching self-tests are not confined (change into the scratch root + GIT_CEILING_DIRECTORIES, #5025)"
        return 1
    fi
    if [ "$n" -eq 0 ]; then echo "RED: no depth-fetching self-test found under $root/scripts -- the scan read nothing"; return 1; fi
    echo "GREEN: $n depth-fetching self-tests, all confined"
}

self_test() {
    local td ok=0 bad=0
    td="$(mktemp -d "${TMPDIR:-/tmp}/fetch-confined.XXXXXX")"
    trap 'rm -rf -- "${td:?}"' RETURN
    cd -- "$td" || return 1
    export GIT_CEILING_DIRECTORIES="${td%/*}" GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
    row() { # row LABEL GOT WANT
        if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; ok=$((ok + 1))
        else printf 'FAIL  %s: got [%s], want [%s]\n' "$1" "$2" "$3"; bad=$((bad + 1)); fi
    }
    put() { mkdir -p -- "$(dirname -- "$td/$1")"; printf '%s\n' "${@:2}" > "$td/$1"; }
    v() { verdict "$td/$1" "$1"; }
    # The planted slip of 2026-10-10: a self-test whose fetch lost its -C, run from the caller's cwd.
    put scripts/a.sh 'if [ "$1" = --self-test ]; then' '  git fetch -q --depth=1 origin main' 'fi'
    row "a self-test that dropped -C and fetches --depth=1 is RED" "$(v scripts/a.sh)" "red: no $CEILING no cd"
    put scripts/b.sh 'if [ "$1" = --self-test ]; then' '  cd -- "$TD" || exit 1' \
        '  export GIT_CEILING_DIRECTORIES="${TD%/*}"' '  git fetch -q --depth=1 origin main' 'fi'
    row "the same self-test with cd + the ceiling is ok" "$(v scripts/b.sh)" ok
    put scripts/c.sh '--self-test' '  export GIT_CEILING_DIRECTORIES="$T"' '  git fetch --deepen=1 origin'
    row "the ceiling without a cd is RED" "$(v scripts/c.sh)" "red: no cd"
    put scripts/d.sh '--self-test' '  cd "$T" && git -C "$w" fetch -q --no-tags --shallow-since=2000-01-01 origin'
    row "a cd without the ceiling is RED (--shallow-since)" "$(v scripts/d.sh)" "red: no $CEILING"
    put scripts/tests/e_test.sh 'git_fx "$C" fetch -q --no-tags --depth=1 "file://$O" main'
    row "a scripts/tests/ harness is a self-test without the flag" "$(v scripts/tests/e_test.sh)" "red: no $CEILING no cd"
    put scripts/ci/f.sh 'git fetch --no-tags --depth=1 origin main   # production, no self-test'
    row "a production script with no self-test is out of scope" "$(v scripts/ci/f.sh)" skip
    put scripts/g.sh '--self-test' 'git fetch origin main' 'git log --depth=1'
    row "a self-test whose fetch has no depth option is out of scope" "$(v scripts/g.sh)" skip
    put scripts/h.sh '--self-test' 'git prefetch_all --depth=1'
    row "a word that only contains 'fetch' is not a fetch" "$(v scripts/h.sh)" skip
    put scripts/i.sh '--self-test' 'echo "cdrom"; export GIT_CEILING_DIRECTORIES=x' 'git fetch --depth=1 o'
    row "a word that only starts with 'cd' is not a cd" "$(v scripts/i.sh)" "red: no cd"
    put scripts/j.sh '    cd -- "$TD" || exit 1; export GIT_CEILING_DIRECTORIES="${TD%/*}"' '#!/usr/bin/env bash' \
        'if [ "$1" = --self-test ]; then' '  TD=$(mktemp -d)' '  git fetch -q --depth=1 origin main' 'fi'
    row "the #5025 slip: cd + ceiling above the shebang and the mktemp is RED" "$(v scripts/j.sh)" \
        "red: $CEILING above the first mktemp"
    put scripts/k.sh '#!/usr/bin/env bash' 'if [ "$1" = --self-test ]; then' '  TD=$(mktemp -d)' \
        '  cd -- "$TD" || exit 1; export GIT_CEILING_DIRECTORIES="${TD%/*}"' '  git fetch -q --depth=1 origin main' 'fi'
    row "the same line moved below its mktemp is ok" "$(v scripts/k.sh)" ok

    git init -q "$td"
    git -C "$td" add -A
    row "scan: the tree with slips is RED" "$(scan "$td" > /dev/null 2>&1; echo $?)" 1
    row "scan names the planted slip" "$(scan "$td" 2>&1 | command grep -c '^RED: scripts/a.sh ')" 1
    git -C "$td" rm -q --cached scripts/a.sh scripts/c.sh scripts/d.sh scripts/i.sh scripts/j.sh scripts/tests/e_test.sh
    row "scan: only confined self-tests left is GREEN" "$(scan "$td" > /dev/null 2>&1; echo $?)" 0
    git -C "$td" rm -q --cached scripts/b.sh scripts/k.sh
    row "scan: no self-test in scope is RED, never a vacuous GREEN" "$(scan "$td" > /dev/null 2>&1; echo $?)" 1

    printf 'check_selftest_fetch_confined self-test: %s ok, %s bad\n' "$ok" "$bad"
    [ "$bad" -eq 0 ] && [ "$ok" -ge 15 ]
}

mutants() {
    local n=0 bad=0 name expr rc td
    td="$(mktemp -d "${TMPDIR:-/tmp}/fetch-confined-mut.XXXXXX")"
    trap 'rm -rf -- "${td:?}"' RETURN
    bash "$SELF" --self-test > /dev/null 2>&1 || { echo "BROKEN: the unmutated self-test is red"; return 1; }
    while IFS=$'\t' read -r name expr; do
        [ -n "$name" ] || continue
        n=$((n + 1))
        sed -E "$expr" "$SELF" > "$td/$name.sh"
        if cmp -s "$SELF" "$td/$name.sh"; then echo "BROKEN: $name changed nothing"; bad=$((bad + 1)); continue; fi
        rc=0; bash "$td/$name.sh" --self-test > /dev/null 2>&1 || rc=$?
        if [ "$rc" != 0 ]; then echo "killed: $name"; else echo "SURVIVED: $name"; bad=$((bad + 1)); fi
    done <<'MUTANTS'
m1_no_ceiling_needed	s/^    command grep -Fq -- "\$CEILING" "\$f" \|\| miss\+=/    true || miss+=/
m2_no_cd_needed	s/^    command grep -Eq -- "\$CD_CMD" "\$f" \|\| miss\+=/    true || miss+=/
m3_tests_dir_not_in_scope	s/^        scripts\/tests\/\*\) : ;;$/        scripts\/tests\/*) echo skip; return ;;/
m4_deepen_not_a_depth	s/--\(depth\|deepen\|shallow-\)/--(depth|shallow-)/
m5_vacuous_green	s/^    if \[ "\$n" -eq 0 \]; then echo "RED/    if false; then echo "RED/
m6_any_fetch_counts	s/^DEPTH_FETCH=.*/DEPTH_FETCH='fetch'/
m7_ceiling_placement_unchecked	s/^    if \[ -n "\$cl" \]/    if false \&\& [ -n "\$cl" ]/
MUTANTS
    echo "--- $((n - bad))/$n mutants killed ---"
    [ "$bad" = 0 ] && [ "$n" -gt 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    --mutants) mutants ;;
    -h | --help) sed -n '2,20p' "$SELF" ;;
    -*) echo "check_selftest_fetch_confined: unknown option $1" >&2; exit 3 ;;
    *) scan "${1:-$(git -C "$(dirname -- "$SELF")" rev-parse --show-toplevel)}" ;;
esac
