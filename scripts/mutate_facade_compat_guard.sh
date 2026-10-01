#!/usr/bin/env bash
# mutate_facade_compat_guard.sh - prove check_facade_compat.sh's SCOPED-RELEASE allowlist turns RED.
#
# PR-REVIEW-SKILL-002 v2 S3.D / S6.4: "Mechanically flip each validation branch and drop
# each required-field check. Target: 100% kill." The scoped-release change (#4604) let six
# crates sit off the workspace version, each pinned at exactly 0.69.4, through two
# functions: scoped_ver() (the closed list) and pub_ver() (the version this tree publishes
# for a crate, or rc 1). A widened allowlist is the one change to this guard that turns a
# RED into a GREEN, so every rule in it is mutated here.
#
# SCOPE: ONLY the scoped-allowlist branches that change added - scoped_ver() and pub_ver().
# The guard's structural rows (R1-R7) are covered by its own fixture table and are not
# re-mutated here. NOT COVERED, and said so rather than hidden: the two CALL SITES
# (CURRENCY's `if WANT_PUB=$(pub_ver ...)` and PUBLISH ORDER's `[ -n "$up_pub" ] && ...`).
# They run only on the main path, after `cargo metadata` and the facade builds, so no
# text-only fixture can reach them; they are exercised by the full guard run in CI.
#
# WHAT A MUTANT IS
#
#   drop  <site>   the rule at that site never fires (the guard goes permissive on ONE rule)
#   flip  <site>   the rule's sense is inverted
#   list  <crate>  one crate removed from the allowlist, or the pinned version changed
#
# Every mutant is applied to a COPY of the guard in a mktemp tree, never in place, and is
# judged by the guard's own `--self-test` (the structural fixture table, the classifier
# table, and the scoped-allowlist rows that drive pub_ver() against committed metadata
# fixtures under scripts/lib/facade_cases/). KILLED = the self-test exits non-zero.
#
# THE PATCH STEP FAILS CLOSED. The old text must occur EXACTLY ONCE in the guard, the
# patched file must differ from the original, and the new text must be present in it.
# Any of those false makes the mutant INVALID - never a kill, never a survivor - and an
# INVALID mutant fails the run.
#
# M0 is the discrimination case: the UNMUTATED copy's self-test must be GREEN.
#
# EXCLUDED, ON PURPOSE (equivalent mutants - no input can tell them from the original):
#   `v=$(python3 ... --version-of ...) || return 1` dropped   version_of() prints only when
#       it exits 0, so a failed read leaves v empty and the next line returns 1 anyway.
#   `[ -n "$v" ]` dropped   an empty v can equal neither a non-empty WS_VER (checked on the
#       same line) nor a non-empty scoped version, so every later branch still returns 1.
#
# Exit 0 = baseline GREEN and every mutant KILLED.
# Exit 1 = a mutant survived, a patch did not apply, or the baseline was not GREEN.
# Exit 2 = the environment could not run the guard at all.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 2
REPO="$PWD"

GUARD_REL=scripts/check_facade_compat.sh

command -v python3 >/dev/null 2>&1 || { printf 'python3 not on PATH: UNMEASURED\n' >&2; exit 2; }
for f in "$GUARD_REL" scripts/lib/facade_facts.py scripts/cargo_classify.sh; do
    [ -f "$REPO/$f" ] || { printf 'missing %s\n' "$f" >&2; exit 2; }
done

T="$(mktemp -d)"
trap 'rm -rf "${T:?}"' EXIT

killed=0
survived=0
invalid=0

# stage <name> - a tree with everything --self-test reads, laid out as in the repo.
stage() {
    local dir="$T/$1"
    mkdir -p "$dir/scripts/lib"
    cp "$REPO/$GUARD_REL" "$dir/$GUARD_REL"
    cp "$REPO/scripts/cargo_classify.sh" "$dir/scripts/"
    cp "$REPO/scripts/lib/facade_facts.py" "$dir/scripts/lib/"
    cp -r "$REPO/scripts/lib/facade_cases" "$dir/scripts/lib/"
    # cargo_classify_selftest (sourced) reads its own case table and errexit probe.
    cp -r "$REPO/scripts/lib/cargo_failure_cases" "$dir/scripts/lib/"
    cp "$REPO/scripts/lib/cargo_classify_errexit_probe.sh" "$dir/scripts/lib/"
    printf '%s' "$dir"
}

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

# battery <tree> - prints the first failing self-test row; rc 0 = self-test GREEN.
battery() {
    local d="$1" rc=0 first
    bash "$d/$GUARD_REL" --self-test > "$d/selftest.out" 2>&1 || rc=$?
    [ "$rc" -eq 0 ] && return 0
    first="$(grep -m1 '^FAIL' "$d/selftest.out" || true)"
    printf 'self-test(rc=%s) %s' "$rc" "${first:-<no FAIL row printed>}"
    return 1
}

# mutant <id> <old> <new>
mutant() {
    local id="$1" old="$2" new="$3" d prc=0 why
    d="$(stage "$id")"
    replace_once "$d/$GUARD_REL" "$old" "$new" || prc=$?
    if [ "$prc" -eq 0 ] && cmp -s "$REPO/$GUARD_REL" "$d/$GUARD_REL"; then prc=4; fi
    if [ "$prc" -ne 0 ]; then
        printf '  %-36s INVALID   patch did not apply (rc=%s) - verdict not read\n' "$id" "$prc"
        invalid=$((invalid + 1))
        return 0
    fi
    if why="$(battery "$d")"; then
        printf '  %-36s SURVIVED  <-- no self-test row noticed\n' "$id"
        survived=$((survived + 1))
    else
        printf '  %-36s KILLED    by %s\n' "$id" "$why"
        killed=$((killed + 1))
    fi
}

printf 'mutate_facade_compat_guard.sh - S3.D mutation set for %s (scoped allowlist)\n\n' "$GUARD_REL"

# --- M0: discrimination case -------------------------------------------------
d="$(stage M0)"
if why="$(battery "$d")"; then
    printf '  %-36s GREEN     as required\n' 'M0-baseline-unmutated'
else
    printf '  %-36s RED at %s - every kill below would be a kill of the harness\n' 'M0-baseline-unmutated' "$why"
    exit 1
fi

# --- scoped_ver(): the closed list, one crate at a time -------------------------
mutant list-drop-build-sha           'aprender-build-sha|aprender-update|' 'aprender-update|'
mutant list-drop-update              '|aprender-update|' '|'
mutant list-drop-contracts-macros    '|aprender-contracts-macros|' '|'
mutant list-drop-common              '|aprender-common|' '|'
mutant list-drop-contracts           '|aprender-contracts|' '|'
mutant list-drop-contracts-cli       '|aprender-contracts-cli) echo' ') echo'
mutant list-pin-moved                'echo 0.69.4 ;;' 'echo 0.69.5 ;;'
mutant list-default-admits-any       '*) return 1 ;;' '*) echo 0.69.4 ;;'

# --- pub_ver(): the rules -------------------------------------------------------
mutant flip-nonempty-guard           '[ -n "$v" ] && [ -n "$WS_VER" ] || return 1' '[ -n "$v" ] && [ -n "$WS_VER" ] && return 1'
mutant drop-ws-nonempty              '[ -n "$v" ] && [ -n "$WS_VER" ] ||' '[ -n "$v" ] ||'
mutant flip-ws-equality              '[ "$v" = "$WS_VER" ] &&' '[ "$v" != "$WS_VER" ] &&'
mutant drop-ws-shortcut              '[ "$v" = "$WS_VER" ] && { printf' 'false && { printf'
mutant ws-shortcut-prints-nothing    '{ printf '"'"'%s\n'"'"' "$v"; return 0; }' '{ return 0; }'
mutant flip-scoped-lookup            'sv=$(scoped_ver "$1") || return 1' 'sv=$(scoped_ver "$1") && return 1'
mutant drop-exact-pin                '[ "$v" = "$sv" ] || return 1' 'true || return 1'
mutant flip-exact-pin                '[ "$v" = "$sv" ] || return 1' '[ "$v" != "$sv" ] || return 1'
mutant scoped-path-prints-nothing    '|| return 1
    printf '"'"'%s\n'"'"' "$v"
}' '|| return 1
    true
}'

total=$((killed + survived + invalid))
printf '\nmutants: %s killed, %s survived, %s invalid (of %s)\n' "$killed" "$survived" "$invalid" "$total"
if [ "$total" -eq 0 ] || [ "$survived" -ne 0 ] || [ "$invalid" -ne 0 ]; then
    printf 'FAIL: %s scoped allowlist is not proved to turn RED for every rule it states.\n' "$GUARD_REL"
    exit 1
fi
printf 'guard_mutation_score = 100%% (%s/%s)\n' "$killed" "$total"
printf 'PASS: baseline GREEN, every mutant KILLED, every patch observed to apply.\n'
exit 0
