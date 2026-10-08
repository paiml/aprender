#!/usr/bin/env bash
# mutate_lean_modules_reachable_guard.sh - prove check_lean_modules_reachable.sh turns RED.
#
# PR-REVIEW-SKILL-002 v2 S3.D / S6.4: "Mechanically flip each validation branch and drop
# each required-field check. Target: 100% kill." The guard's `--self-test` is a table its
# own author wrote; until each rule it states has been broken on purpose and watched to
# turn a row RED, that table proves only that it agrees with itself.
#
# WHAT A MUTANT IS
#
#   drop  <site>   the rule at that site never fires (the guard goes permissive on ONE rule)
#   flip  <site>   the rule's sense is inverted (a clean tree reads RED, a broken one GREEN)
#
# Every mutant is applied to a COPY of the guard in a mktemp tree, never in place, and is
# judged by the BATTERY:
#   1. the guard's own `--self-test` (rows over copies of the real Lean tree), and
#   2. a judge-mode fixture table - the guard run with no arguments against three staged
#      roots (clean -> 0, an orphan module -> 1, no root file -> 2). `--self-test` calls
#      reach() directly, so without these rows nothing tests judge()'s exit status.
# A mutant is KILLED when any battery row gives the wrong answer, SURVIVED when none does.
#
# THE PATCH STEP FAILS CLOSED. The old text must occur EXACTLY ONCE in the guard, the
# patched file must differ from the original, and the new text must be present in it.
# Any of those false makes the mutant INVALID - never a kill, never a survivor - and an
# INVALID mutant fails the run. (A mutation that never applied once reported "SURVIVED"
# in mutate_vendored_schemas_guard.sh's first draft, and the mirror image - a kill for a
# mutant that changed nothing - is worse.)
#
# M0 is the discrimination case: the UNMUTATED copy must pass the whole battery. Without
# it, "the battery fails unconditionally" scores a perfect kill rate.
#
# EXCLUDED, ON PURPOSE (equivalent mutants - no input can tell them from the original):
#   `if m in seen: continue` dropped   the import graph is a DAG, so revisiting a module
#                                      re-adds the same names to `seen`; only time differs.
#   self_test()'s own machinery        it is the harness, not a validation branch.
#
# Exit 0 = baseline GREEN and every mutant KILLED.
# Exit 1 = a mutant survived, a patch did not apply, or the baseline was not GREEN.
# Exit 2 = the environment could not run the guard at all.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 2
REPO="$PWD"

GUARD_REL=scripts/check_lean_modules_reachable.sh
LEAN_REL=crates/aprender-contracts-staging/lean

command -v python3 >/dev/null 2>&1 || { printf 'python3 not on PATH: UNMEASURED\n' >&2; exit 2; }
[ -f "$REPO/$GUARD_REL" ] || { printf 'no guard at %s\n' "$GUARD_REL" >&2; exit 2; }
[ -f "$REPO/$LEAN_REL/ProvableContracts.lean" ] || { printf 'no Lean root under %s\n' "$LEAN_REL" >&2; exit 2; }

T="$(mktemp -d)"
trap 'rm -rf "${T:?}"' EXIT

killed=0
survived=0
invalid=0

# stage_root <dir> <guard-file> - one repo-shaped root: the guard plus the Lean inputs it reads.
stage_root() {
    local dir="$1" guard="$2"
    mkdir -p "$dir/scripts" "$dir/$LEAN_REL"
    cp "$guard" "$dir/$GUARD_REL"
    cp -r "$REPO/$LEAN_REL/ProvableContracts" "$REPO/$LEAN_REL/ProvableContracts.lean" \
          "$REPO/$LEAN_REL/orphan-allowlist.yaml" "$dir/$LEAN_REL/"
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

# battery <guard-file> - prints the first row that gave the wrong answer; rc 0 = all right.
battery() {
    local g="$1" b rc want name
    b="$(mktemp -d "$T/battery.XXXXXX")"
    stage_root "$b/selftest" "$g"
    rc=0
    bash "$b/selftest/$GUARD_REL" --self-test > "$b/selftest.out" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then printf 'self-test(rc=%s)' "$rc"; return 1; fi

    stage_root "$b/clean" "$g"
    stage_root "$b/orphan" "$g"
    printf 'theorem t : True := trivial\n' > "$b/orphan/$LEAN_REL/ProvableContracts/MutantOrphan.lean"
    stage_root "$b/noroot" "$g"
    rm -f "${b:?}/noroot/${LEAN_REL:?}/ProvableContracts.lean"
    for row in "0 clean" "1 orphan" "2 noroot"; do
        want=${row%% *}; name=${row#* }
        rc=0
        bash "$b/$name/$GUARD_REL" > "$b/$name.out" 2>&1 || rc=$?
        if [ "$rc" != "$want" ]; then printf 'judge-%s(rc=%s,want=%s)' "$name" "$rc" "$want"; return 1; fi
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

printf 'mutate_lean_modules_reachable_guard.sh - S3.D mutation set for %s\n\n' "$GUARD_REL"

# --- M0: discrimination case -------------------------------------------------
if why="$(battery "$REPO/$GUARD_REL")"; then
    printf '  %-40s GREEN     as required\n' 'M0-baseline-unmutated'
else
    printf '  %-40s RED at %s - every kill below would be a kill of the harness\n' 'M0-baseline-unmutated' "$why"
    exit 1
fi

# --- reach(): the root/tree presence check (rc 2) -----------------------------
mutant drop-root-check        '|| { echo "no root in $1"; return 2; }' '|| true'
mutant drop-root-file-test    '[ -f "$1/ProvableContracts.lean" ] && [ -d' '[ -d'
mutant drop-root-dir-test     '] && [ -d "$1/ProvableContracts" ] ||' '] ||'

# --- reach(): which import lines count --------------------------------------
mutant import-regex-never     "imp = re.compile(r'^\\s*import" "imp = re.compile(r'^\\s*NEVERimport"
mutant drop-import-prefix     'if x.startswith("ProvableContracts.")]' 'if x]'
mutant drop-lean-suffix       'if f.endswith(".lean"):' 'if True:'

# --- reach(): the walk --------------------------------------------------------
mutant flip-dangling-test     'if m not in mods:' 'if m in mods:'
mutant drop-dangling-record   'dangling.add(m)' 'pass'
mutant drop-seen-record       'seen.add(m)' 'pass'
mutant drop-transitive-walk   'todo += imports(mods[m])' 'pass'

# --- reach(): the orphan allowlist (closed list, shrink-only) -----------------
mutant drop-allowlist-read    'if os.path.exists(ap):' 'if False:'
mutant allowlist-regex-never  "r'^\\s*-?\\s*module:" "r'^\\s*-?\\s*NEVERmodule:"
mutant drop-allowlist-excuse  'orphans - allow)]' 'orphans)]'
mutant drop-stale-rule        'bad += [f"stale-allowlist {m}" for m in sorted(allow - orphans)]' 'bad += []'
mutant drop-dangling-verdict  ' + [f"dangling {m}" for m in sorted(dangling)]' ''
mutant exit-always-zero       'sys.exit(1 if bad else 0)' 'sys.exit(0)'

# --- judge(): the exit status the gate reads ---------------------------------
mutant judge-returns-zero     'return "$rc"' 'return 0'
mutant flip-judge-pass-branch 'if [ "$rc" -eq 0 ]; then echo "PASS' 'if [ "$rc" -ne 0 ]; then echo "PASS'

total=$((killed + survived + invalid))
printf '\nmutants: %s killed, %s survived, %s invalid (of %s)\n' "$killed" "$survived" "$invalid" "$total"
if [ "$total" -eq 0 ] || [ "$survived" -ne 0 ] || [ "$invalid" -ne 0 ]; then
    printf 'FAIL: %s is not proved to turn RED for every rule it states.\n' "$GUARD_REL"
    exit 1
fi
printf 'guard_mutation_score = 100%% (%s/%s)\n' "$killed" "$total"
printf 'PASS: baseline GREEN, every mutant KILLED, every patch observed to apply.\n'
exit 0
