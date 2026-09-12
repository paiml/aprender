#!/usr/bin/env bash
# mutate_pr_review_wiring_guard.sh — the mutation set for
# scripts/check_pr_review_wiring.sh.
#
# Same shape and the same reason as scripts/mutate_vendored_schemas_guard.sh
# (PRREV-002) and scripts/mutate-guard.sh (PRREV-004): a guard's case table is
# evidence only for the branches it can turn RED. Every rule the guard STATES
# gets a mutant that removes it, and the guard's own `--self-test` must catch
# each one. A surviving mutant is a rule nothing tests.
#
# THREE THINGS THIS SET ALREADY CAUGHT, none of which review did.
#
#  1. R1's and R3's zero branches were REDUNDANT with their more-than-one
#     branches, because a count of 0 also satisfies `!= 1`. Dropping either
#     `-z` check left the guard still rejecting — on the neighbouring branch,
#     with a message about "more than one job" for a file that had none. The
#     table asserted only the verdict, so both mutants SURVIVED. Asserting the
#     DIAGNOSTIC is what kills them, which is the finding tests/pr-review.bats
#     records for the receipt guard's B1 one level up.
#
#  2. The step-level fixture was the WRONG SHAPE. It wrote `      - if: ...`,
#     where the `- ` means a widened job-level pattern `^ *if:` still does not
#     match — so the mutation that demotes job-level detection to "any
#     indentation" survived. The common spelling is an `if:` at 8 spaces under a
#     `- name:` step, and that one does match. A fixture that cannot be reached
#     by the mutant is not coverage.
#
#  3. The evaluator compared literals WITHOUT word delimiters and nothing
#     noticed, because no row used an event name that is a prefix of another.
#     `pull_request` is a prefix of the real `pull_request_target`.
#
# A MUTATION THAT FAILS TO APPLY IS HARNESS-BROKEN, NEVER A KILL. Every anchor
# is asserted present, every mutant is asserted to differ from the original, and
# rc=127 (the mutant file never executed) is reported as HARNESS rather than as
# a dead mutant. An early run of this very set reported `KILLED ... rc=127` for
# a mutant whose anchor had been mangled — a false kill, and the exact shape
# scripts/mutate-guard.sh names as the first of the three ways it has been
# fooled.
#
#   bash scripts/mutate_pr_review_wiring_guard.sh          # run the set
#   bash scripts/mutate_pr_review_wiring_guard.sh --list   # print the catalogue
#
# EXIT: 0 only when every mutant was killed and no anchor was broken.

set -euo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$REPO_ROOT/scripts/check_pr_review_wiring.sh"

for t in python3 bash; do
    command -v "$t" >/dev/null 2>&1 || {
        echo "$PROG: FAIL (environment) - $t is not on PATH." >&2
        echo "  An unmeasured mutation score is not a score of 1." >&2
        exit 2
    }
done

if [ ! -f "$GUARD" ]; then
    echo "$PROG: FAIL - no guard at $GUARD" >&2
    exit 1
fi

WORK=$(mktemp -d) || exit 1
cleanup() { rm -rf "${WORK:?}"; }
trap cleanup EXIT

LIST_ONLY=0
case "${1:-}" in
    --list) LIST_ONLY=1 ;;
    '')     ;;
    *)      echo "$PROG: unknown argument '$1'" >&2; exit 2 ;;
esac

# The baseline FIRST. A set whose unmutated control is already RED scores
# nothing: every mutant would "die" of the pre-existing failure.
if [ "$LIST_ONLY" -eq 0 ]; then
    if ! bash "$GUARD" --self-test > "$WORK/baseline.log" 2>&1; then
        echo "$PROG: FAIL - the UNMUTATED guard's own self-test is red." >&2
        sed 's/^/    /' "$WORK/baseline.log" >&2
        echo "  Fix that first; against a red baseline every mutant dies for the wrong reason." >&2
        exit 1
    fi
    printf 'baseline: the unmutated guard self-tests GREEN\n\n'
fi

GUARD_PATH="$GUARD" WORK_DIR="$WORK" LIST_ONLY="$LIST_ONLY" python3 - <<'PY'
import os, pathlib, subprocess, sys

guard = pathlib.Path(os.environ["GUARD_PATH"])
work  = pathlib.Path(os.environ["WORK_DIR"])
list_only = os.environ["LIST_ONLY"] == "1"
src = guard.read_text()

# id -> (rule, what it removes, anchor, replacement)
#
# RE-DERIVED 2026-09-10 from the rules the guard now STATES. BSE-15 (#3046) deleted the receipt job, so the
# guard became: R1 = no job invokes the receipt guard (one presence branch), R2 = unchanged, R3 = the guard
# is still tracked (so guard_tree.sh's derived universe runs it). The old set still mutated R1's zero and
# many-job branches, R3's job-level if: and the whole R4 event evaluator -- code the rewrite removed -- so
# 11 of 15 anchors matched nothing and the set reported 4/4 killed over rules half of which no longer exist.
MUTANTS = [
 ("R1-drop", "R1", "the receipt-job-is-back rejection",
  '    if [ -n "$job" ]; then', '    if false; then'),
 ("R1-mention", "R1", "comment stripping, so a MENTION reads as an invocation",
  '            line = $0; sub(/#.*$/, "", line)\n            if (job != "" && line ~ re) { print job }',
  '            line = $0\n            if (job != "" && line ~ re) { print job }'),
 ("R1-jobscope", "R1", "entry to the jobs: block, so no invocation is ever seen",
  '        /^jobs:[[:space:]]*$/            { injobs = 1; next }', '        /^ZZZNEVER/                      { injobs = 1; next }'),
 ("R2-drop", "R2", "the workflow-level path-filter rejection",
  '    if [ -n "$filters" ]; then', '    if false; then'),
 ("R2-onstart", "R2", "entry to the on: block, so no filter is ever seen",
  '        /^on:/            { ino = 1; next }', '        /^ZZZNEVER/       { ino = 1; next }'),
 ("R2-onblock", "R2", "exit from the on: block, so paths: anywhere reads as a filter",
  'ino && /^[^[:space:]#]/ { ino = 0 }', 'ino && /^ZZZNEVER/ { ino = 0 }'),
 ("R3-drop", "R3", "the untracked-guard rejection",
  '    if ! guard_is_tracked; then', '    if false; then'),
 ("R3-lsfiles", "R3", "the git ls-files probe, so an untracked guard reads as tracked",
  '    git -C "$REPO_ROOT" ls-files --error-unmatch "scripts/$GUARD_BASENAME" >/dev/null 2>&1', '    true'),
]

if list_only:
    for mid, rule, what, _o, _n in MUTANTS:
        print(f"{mid:<16} {rule}  removes {what}")
    print(f"\n{len(MUTANTS)} mutants")
    sys.exit(0)

killed = survived = harness = 0
for mid, rule, what, old, new in MUTANTS:
    if old not in src:
        print(f"HARNESS  {mid:<16} {rule}  ANCHOR MISSING - this mutant tested NOTHING")
        harness += 1
        continue
    mutated = src.replace(old, new, 1)
    if mutated == src:
        print(f"HARNESS  {mid:<16} {rule}  NO-OP - the edit changed no bytes")
        harness += 1
        continue
    f = work / f"{mid}.sh"
    f.write_text(mutated)
    r = subprocess.run(["bash", str(f), "--self-test"], capture_output=True, text=True)
    if r.returncode == 127:
        print(f"HARNESS  {mid:<16} {rule}  rc=127 - the mutant never executed")
        harness += 1
    elif r.returncode != 0:
        print(f"killed   {mid:<16} {rule}  {what}")
        killed += 1
    else:
        print(f"SURVIVED {mid:<16} {rule}  {what}")
        print(f"         The case table cannot tell this rule is gone. Add a row that can.")
        survived += 1

total = killed + survived
print()
print(f"{killed}/{total} killed, {harness} harness-broken")
if harness:
    print("A harness failure is NOT a kill: an anchor that no longer matches means the")
    print("mutant was never applied, and reporting it green is how a mutation score lies.")
sys.exit(0 if survived == 0 and harness == 0 else 1)
PY
