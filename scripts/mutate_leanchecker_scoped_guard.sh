#!/usr/bin/env bash
# mutate_leanchecker_scoped_guard.sh - prove check_leanchecker_scoped.sh turns RED.
#
# PR-REVIEW-SKILL-002 v2 S3.D / S6.4: "Mechanically flip each validation branch and drop
# each required-field check. Target: 100% kill." The guard's `--self-test` is a table its
# own author wrote; until each rule it states has been broken on purpose and watched to
# turn a row RED, that table proves only that it agrees with itself.
#
# WHAT A MUTANT IS
#
#   drop  <site>   the rule at that site never fires (the guard goes permissive on ONE rule)
#   flip  <site>   the rule's sense is inverted
#   regex <site>   one alternative of CMD_RE / OPTOUT_RE removed, or the pattern made to
#                  match nothing or too much - guard regexes here were wrong five times and
#                  every one was caught by a case table, none by review
#
# Every mutant is applied to a COPY of the guard in a mktemp tree, never in place, and is
# judged by the BATTERY:
#   1. the guard's own `--self-test` (judge_line's must-match / must-not-match rows), and
#   2. a scan-mode fixture table: the guard run with no arguments at the root of a scratch
#      git repo, one per surface it claims (.yml, .yaml, Makefile, *.mk, *.sh), one with a
#      RED last line that has no trailing newline, one clean, and one with nothing tracked
#      but the guard itself (UNMEASURED, rc 2). `--self-test` calls judge_line() only, so
#      without these rows nothing tests scan() at all.
# A mutant is KILLED when any battery row gives the wrong answer, SURVIVED when none does.
#
# THE PATCH STEP FAILS CLOSED. The old text must occur EXACTLY ONCE in the guard, the
# patched file must differ from the original, and the new text must be present in it.
# Any of those false makes the mutant INVALID - never a kill, never a survivor - and an
# INVALID mutant fails the run.
#
# M0 is the discrimination case: the UNMUTATED copy must pass the whole battery.
#
# THIS FILE IS ON THE GUARD'S OWN SURFACE (*.sh). The tool's name is therefore never
# written literally outside a comment line; it is assembled into $LC, so the real guard
# does not read this file's mutation table as a RED line.
#
# EXCLUDED, ON PURPOSE (equivalent mutants - no input can tell them from the original):
#   the exec|env|nice|lake env|timeout prefix alternatives in CMD_RE. The group is `*`,
#     and every prefix is followed by whitespace, which the leading class already accepts
#     directly in front of the tool name - so removing any one changes no verdict.
#   scan()'s `*<tool>*` pre-filter dropped: judge_line() returns 0 on every line that
#     does not contain the name, so the filter only saves time. (Its FLIP is mutated.)
#   self_test()'s own machinery - it is the harness, not a validation branch.
#
# Exit 0 = baseline GREEN and every mutant KILLED.
# Exit 1 = a mutant survived, a patch did not apply, or the baseline was not GREEN.
# Exit 2 = the environment could not run the guard at all.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 2
REPO="$PWD"

GUARD_REL=scripts/check_leanchecker_scoped.sh
LC=lean
LC="${LC}checker"

for t in python3 git; do
    command -v "$t" >/dev/null 2>&1 || { printf '%s not on PATH: UNMEASURED\n' "$t" >&2; exit 2; }
done
[ -f "$REPO/$GUARD_REL" ] || { printf 'no guard at %s\n' "$GUARD_REL" >&2; exit 2; }

T="$(mktemp -d)"
trap 'rm -rf "${T:?}"' EXIT

killed=0
survived=0
invalid=0

# replace_once <file> <old> <new> - rc 0 applied; 3 old not present exactly once;
# 4 file unchanged; 5 new text absent afterwards. Fails closed on every one of them.
replace_once() {
    python3 -c '
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p, encoding="utf-8").read()
if s.count(old) != 1:
    sys.exit(3)
t = s.replace(old, new, 1)
if t == s:
    sys.exit(4)
if new and new not in t:
    sys.exit(5)
open(p, "w", encoding="utf-8").write(t)
' "$1" "$2" "$3"
}

# fixture <dir> <guard-file> - a scratch git repo with the guard tracked at its real path,
# plus one clean tracked script, so a dropped surface reads as a quiet PASS (rc 0), which
# is how it would fail in the real tree, and not as an UNMEASURED rc 2.
fixture() {
    mkdir -p "$1/scripts" "$1/.github/workflows" "$1/tools/deep"
    cp "$2" "$1/$GUARD_REL"
    printf '#!/bin/sh\npv discharge run lean\n' > "$1/tools/clean.sh"
}

track() {
    git -C "$1" init -q
    git -C "$1" add -A
}

# battery <guard-file> - prints the first row that gave the wrong answer; rc 0 = all right.
battery() {
    local g="$1" b rc want name
    b="$(mktemp -d "$T/battery.XXXXXX")"
    rc=0
    bash "$g" --self-test > "$b/selftest.out" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then printf 'self-test(rc=%s)' "$rc"; return 1; fi

    fixture "$b/clean" "$g"
    printf 'lean:\n\tsystemd-run --user --scope -p MemoryMax=24G lake env %s ProvableContracts\n' "$LC" > "$b/clean/Makefile"
    printf 'pv discharge check --%s lean\n' "$LC" > "$b/clean/tools/ok.sh"
    fixture "$b/yml" "$g"
    printf 'jobs:\n  a:\n    steps:\n      - run: lake env %s ProvableContracts\n' "$LC" > "$b/yml/.github/workflows/a.yml"
    fixture "$b/yaml" "$g"
    printf 'jobs:\n  a:\n    steps:\n      - run: lake env %s ProvableContracts\n' "$LC" > "$b/yaml/.github/workflows/b.yaml"
    fixture "$b/makefile" "$g"
    printf 'lean:\n\tlake env %s ProvableContracts\n' "$LC" > "$b/makefile/Makefile"
    fixture "$b/mk" "$g"
    printf 'lean:\n\tlake env %s ProvableContracts\n' "$LC" > "$b/mk/lean.mk"
    fixture "$b/sh" "$g"
    printf '#!/bin/sh\nlake env %s ProvableContracts\n' "$LC" > "$b/sh/tools/deep/run.sh"
    fixture "$b/lastline" "$g"
    printf 'lake env %s ProvableContracts' "$LC" > "$b/lastline/tools/last.sh"
    mkdir -p "$b/empty/scripts"
    cp "$g" "$b/empty/$GUARD_REL"
    for name in clean yml yaml makefile mk sh lastline empty; do track "$b/$name"; done

    for row in "0 clean" "1 yml" "1 yaml" "1 makefile" "1 mk" "1 sh" "1 lastline" "2 empty"; do
        want=${row%% *}; name=${row#* }
        rc=0
        bash "$b/$name/$GUARD_REL" > "$b/$name.out" 2>&1 || rc=$?
        if [ "$rc" != "$want" ]; then printf 'scan-%s(rc=%s,want=%s)' "$name" "$rc" "$want"; return 1; fi
    done
    return 0
}

# mutant <id> <old> <new>
mutant() {
    local id="$1" old="$2" new="$3" g prc=0 why
    g="$T/$id.sh"
    cp "$REPO/$GUARD_REL" "$g"
    replace_once "$g" "$old" "$new" || prc=$?
    if [ "$prc" -eq 0 ] && cmp -s "$REPO/$GUARD_REL" "$g"; then prc=4; fi
    if [ "$prc" -ne 0 ]; then
        printf '  %-40s INVALID   patch did not apply (rc=%s) - verdict not read\n' "$id" "$prc"
        invalid=$((invalid + 1))
        return 0
    fi
    if why="$(battery "$g")"; then
        printf '  %-40s SURVIVED  <-- no battery row noticed\n' "$id"
        survived=$((survived + 1))
    else
        printf '  %-40s KILLED    by %s\n' "$id" "$why"
        killed=$((killed + 1))
    fi
}

printf 'mutate_leanchecker_scoped_guard.sh - S3.D mutation set for %s\n\n' "$GUARD_REL"

# --- M0: discrimination case -------------------------------------------------
if why="$(battery "$REPO/$GUARD_REL")"; then
    printf '  %-40s GREEN     as required\n' 'M0-baseline-unmutated'
else
    printf '  %-40s RED at %s - every kill below would be a kill of the harness\n' 'M0-baseline-unmutated' "$why"
    exit 1
fi

# --- judge_line(): the rules --------------------------------------------------
mutant drop-comment-skip       '[[ "$l" =~ ^[[:space:]]*# ]] && return 0' '[[ "$l" =~ ^[[:space:]]*# ]] && true'
mutant flip-comment-skip       '[[ "$l" =~ ^[[:space:]]*# ]] && return 0' '[[ "$l" =~ ^[[:space:]]*# ]] || return 0'
mutant drop-optout-reject      "drops the 24G/800%% scope'; return 1" "drops the 24G/800%% scope'; return 0"
mutant drop-unscoped-reject    "check --${LC}'; return 1" "check --${LC}'; return 0"
mutant drop-systemd-run-req    '! { [[ "$l" == *systemd-run* ]] &&' '! { true &&'
mutant drop-memorymax-req      '&& [[ "$l" == *MemoryMax=* ]]; }' '&& true; }'
mutant scope-and-becomes-or    '*systemd-run* ]] && [[' '*systemd-run* ]] || [['
mutant flip-scope-negation     ']] && ! { [[' ']] && { [['

# --- OPTOUT_RE -----------------------------------------------------------------
mutant optout-regex-never      "OPTOUT_RE='--${LC}-unscoped'" "OPTOUT_RE='--${LC}-NEVER'"

# --- CMD_RE: the leading class, the path qualifier, the trailing class ----------
mutant cmd-regex-never         "/)?${LC}(" "/)?${LC}_NEVER("
mutant lead-drop-line-start    "CMD_RE='(^|[;&|(" "CMD_RE='([;&|("
mutant lead-drop-punctuation   '(^|[;&|(`]|[[:space:]])((exec' '(^|[[:space:]])((exec'
mutant lead-drop-whitespace    ']|[[:space:]])((exec' '])((exec'
mutant lead-matches-anything   "CMD_RE='(^|[;&|(" "CMD_RE='(^|.|[;&|("
mutant drop-path-qualifier     '("?[^[:space:]"]*/)?' ''
mutant trail-drop-quote        '("|[[:space:]]|$|[;)&|])'"'" '([[:space:]]|$|[;)&|])'"'"
mutant trail-drop-whitespace   '("|[[:space:]]|$|[;)&|])'"'" '("|$|[;)&|])'"'"
mutant trail-drop-end-of-line  '("|[[:space:]]|$|[;)&|])'"'" '("|[[:space:]]|[;)&|])'"'"
mutant trail-drop-punctuation  '("|[[:space:]]|$|[;)&|])'"'" '("|[[:space:]]|$)'"'"
mutant trail-matches-anything  '("|[[:space:]]|$|[;)&|])'"'" '("|[[:space:]]|$|[;)&|]|.)'"'"

# --- scan(): the surfaces and the verdict --------------------------------------
mutant drop-own-file-skip      '_scoped.sh ] && continue' '_scoped.sh ] && true'
mutant flip-name-prefilter     "[[ \"\$l\" == *${LC}* ]] || continue" "[[ \"\$l\" == *${LC}* ]] && continue"
mutant drop-last-line-read     '|| [ -n "$l" ]; do' '; do'
mutant drop-red-record         '"$why" "$l"; red=1' '"$why" "$l"; red=0'
mutant scan-returns-zero       'return "$red"' 'return 0'
mutant drop-unmeasured-floor   '[ "$files" -gt 0 ] || {' 'true || {'
mutant drop-surface-yml        "'.github/workflows/*.yml' " ''
mutant drop-surface-yaml       "'.github/workflows/*.yaml' " ''
mutant drop-surface-makefile   "'Makefile' " ''
mutant drop-surface-mk         "'*.mk' " ''
mutant drop-surface-sh         " '*.sh')" ')'

total=$((killed + survived + invalid))
printf '\nmutants: %s killed, %s survived, %s invalid (of %s)\n' "$killed" "$survived" "$invalid" "$total"
if [ "$total" -eq 0 ] || [ "$survived" -ne 0 ] || [ "$invalid" -ne 0 ]; then
    printf 'FAIL: %s is not proved to turn RED for every rule it states.\n' "$GUARD_REL"
    exit 1
fi
printf 'guard_mutation_score = 100%% (%s/%s)\n' "$killed" "$total"
printf 'PASS: baseline GREEN, every mutant KILLED, every patch observed to apply.\n'
exit 0
