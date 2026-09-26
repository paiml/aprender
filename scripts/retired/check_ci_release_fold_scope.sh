#!/usr/bin/env bash
# scripts/check_ci_release_fold_scope.sh -- a fold push to release/** runs the tree guards, and ONLY them (#4112).
#
# Batching folds each author branch into a release/** branch by a direct PUSH, and no PR is opened. ci.yml
# fired only for main/master, so guard-tree and guard-cargo first ran on the release->main integration PR:
# drift was caught at release time, not at the fold that caused it. The operator approved option B
# (2026-09-24): `push: branches` includes 'release/**', and a push there runs guard-tree + guard-cargo and
# nothing else, so a fold never pays for the full workspace-test.
#
# This pins the three things that make that true, each of which a later edit can silently undo:
#   R1  on.push.branches contains 'release/**'.
#   R2  for EVERY job: it runs on a release push iff it is guard-tree or guard-cargo. A job's `if:` is
#       evaluated for three events (pull_request->main, push->main, push->release/x) by a small evaluator
#       that knows only the expression shapes ci.yml uses; any other shape is REFUSED, never guessed.
#   R3  both guard jobs unshallow on a release push: a depth-1 checkout cannot name merge-base(origin/main,
#       HEAD), and scripts/lib/resolve_base.sh refuses a single-parent fold outright without it.
#
# Usage:  check_ci_release_fold_scope.sh [ci.yml]   default .github/workflows/ci.yml
#         check_ci_release_fold_scope.sh --self-test  the case table (planted mutants must go RED)
# Executed, never sourced.
set -euo pipefail

GUARDS='guard-tree guard-cargo'

scan() {
  local wf=$1
  [ -f "$wf" ] || { echo "check_ci_release_fold_scope: $wf not found -- refusing to pass vacuously"; return 2; }
  GUARDS="$GUARDS" python3 - "$wf" <<'PY'
import os, re, sys, yaml

wf = sys.argv[1]
d = yaml.safe_load(open(wf))
on = d.get(True, d.get("on")) or {}
guards = set(os.environ["GUARDS"].split())
bad = []

# R1
branches = ((on.get("push") or {}).get("branches")) or []
if "release/**" not in branches:
    bad.append(f"R1: on.push.branches {branches} lacks 'release/**': a fold push runs no guard (#4112)")

FOLD = "github.event_name == 'push' && startsWith(github.ref, 'refs/heads/release/')"
EVENTS = {
    "pull_request->main": {"event": "pull_request", "ref": "refs/pull/1/merge"},
    "push->main": {"event": "push", "ref": "refs/heads/main"},
    "push->release": {"event": "push", "ref": "refs/heads/release/0.70"},
}

def ev(expr, ctx):
    """Evaluate the expression shapes ci.yml uses. Returns True/False, or None for an unknown shape."""
    e = (expr or "").strip()
    if e.startswith("${{") and e.endswith("}}"):
        e = e[3:-2].strip()
    if e.startswith('"') and e.endswith('"'):
        e = e[1:-1].strip()
    fold = ctx["event"] == "push" and ctx["ref"].startswith("refs/heads/release/")
    table = {
        "": True,
        "always()": True,
        f"!({FOLD})": not fold,
        f"always() && !({FOLD})": not fold,
        "github.event_name == 'pull_request'": ctx["event"] == "pull_request",
    }
    if e in table:
        return table[e]
    # PR-only jobs gated further on a needs output: false outside a PR, unknowable (None) inside one
    if e.startswith("github.event_name == 'pull_request' &&"):
        return False if ctx["event"] != "pull_request" else "pr-conditional"
    return None

runs = {name: {} for name in d["jobs"]}
for name, job in d["jobs"].items():
    for label, ctx in EVENTS.items():
        v = ev(job.get("if"), ctx)
        if v is None:
            bad.append(f"R2: job {name}: `if: {job.get('if')}` is a shape this guard cannot evaluate -- extend ev(), never guess")
        runs[name][label] = v

for name in d["jobs"]:
    rel = runs[name]["push->release"]
    if name in guards and rel is not True:
        bad.append(f"R2: {name} must run on a release push (it is the reason #4112 exists); its if evaluates {rel}")
    if name not in guards and rel not in (False,):
        bad.append(f"R2: {name} would run on a release push (if: {d['jobs'][name].get('if')}); only {sorted(guards)} may")
    # a job's `needs` that is skipped on a release push would skip a guard silently
    if name in guards:
        needs = d["jobs"][name].get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        for n in needs:
            if runs.get(n, {}).get("push->release") is False:
                bad.append(f"R2: guard {name} needs {n}, which a release push skips: the guard would never run")

# R3
for g in sorted(guards):
    steps = d["jobs"].get(g, {}).get("steps") or []
    body = "\n".join(str(s.get("run", "")) for s in steps)
    if not re.search(r"push:refs/heads/release/\*\)\s*git fetch[^\n]*--unshallow", body):
        bad.append(f"R3: {g} does not unshallow on a release push: resolve_base.sh refuses a single-parent fold at depth 1")

if bad:
    print("\n".join(bad))
    print(f"check_ci_release_fold_scope: {len(bad)} violation(s) in {wf} (#4112)")
    sys.exit(1)
rel = sorted(n for n in d["jobs"] if runs[n]["push->release"] is True)
print(f"check_ci_release_fold_scope: OK -- {len(d['jobs'])} jobs evaluated; a release push runs exactly {rel}")
PY
}

self_test() {
  local dir rc fails=0 src=${1:-.github/workflows/ci.yml}
  dir=$(mktemp -d)
  trap 'rm -rf "${dir:?}"' RETURN
  [ -f "$src" ] || { echo "SELF-TEST: $src not found"; return 2; }
  cp "$src" "$dir/ok.yml"
  # mutant 1: release/** dropped from the push trigger
  sed "s#branches: \[main, master, 'release/\*\*'\]#branches: [main, master]#" "$src" > "$dir/m1.yml"
  # mutant 2: the fold exclusion dropped from one heavy job (workspace-test-shard would run on every fold)
  python3 - "$src" "$dir/m2.yml" <<'PY'
import sys
s = open(sys.argv[1]).read()
i = s.index("\n  workspace-test-shard:\n")
j = s.index("    if: ${{ !(", i)
k = s.index("\n", j)
open(sys.argv[2], "w").write(s[:j] + "    # (mutant)" + s[k:])
PY
  # mutant 3: a guard job excluded from folds
  python3 - "$src" "$dir/m3.yml" <<'PY'
import sys
s = open(sys.argv[1]).read()
i = s.index("\n  guard-tree:\n") + len("\n  guard-tree:\n")
open(sys.argv[2], "w").write(s[:i] + "    if: ${{ !(github.event_name == 'push' && startsWith(github.ref, 'refs/heads/release/')) }}\n" + s[i:])
PY
  # mutant 4: the unshallow dropped (both guard jobs)
  grep -v 'push:refs/heads/release/\*) git fetch --no-tags --unshallow' "$src" > "$dir/m4.yml"
  # mutant 5: an if shape the evaluator does not know must be refused, not guessed
  python3 - "$src" "$dir/m5.yml" <<'PY'
import sys
s = open(sys.argv[1]).read()
i = s.index("\n  mac-check:\n") + len("\n  mac-check:\n")
s2 = s[:i] + s[i:].replace("    if: ${{ !(", "    if: ${{ github.ref_name != 'x' && !(", 1)
open(sys.argv[2], "w").write(s2)
PY
  # mutant 6: a guard made to depend on a job a release push skips (the guard would silently never run)
  python3 - "$src" "$dir/m6.yml" <<'PY'
import sys
s = open(sys.argv[1]).read()
i = s.index("\n  guard-cargo:\n") + len("\n  guard-cargo:\n")
open(sys.argv[2], "w").write(s[:i] + "    needs: [ci]\n" + s[i:])
PY
  for c in m1 m2 m3 m4 m5 m6; do
    if cmp -s "$dir/$c.yml" "$dir/ok.yml"; then echo "SELF-TEST FAIL: mutant $c did not change the file"; fails=$((fails + 1)); continue; fi
    rc=0; scan "$dir/$c.yml" > "$dir/$c.out" 2>&1 || rc=$?
    [ "$rc" -eq 1 ] || { echo "SELF-TEST FAIL: mutant $c must be flagged (rc=$rc)"; cat "$dir/$c.out"; fails=$((fails + 1)); }
  done
  rc=0; scan "$dir/does-not-exist.yml" > /dev/null 2>&1 || rc=$?
  [ "$rc" -eq 2 ] || { echo "SELF-TEST FAIL: a missing workflow must refuse (rc=$rc)"; fails=$((fails + 1)); }
  rc=0; scan "$dir/ok.yml" > "$dir/ok.out" 2>&1 || rc=$?
  [ "$rc" -eq 0 ] || { echo "SELF-TEST FAIL: the real ci.yml must pass (rc=$rc)"; cat "$dir/ok.out"; fails=$((fails + 1)); }
  if [ "$fails" -eq 0 ]; then echo "check_ci_release_fold_scope self-test: 8/8 cases OK"; return 0; fi
  return 1
}

case "${1:-}" in
  --help|-h) echo "usage: $0 [ci.yml] | --self-test   (default .github/workflows/ci.yml)" ;;
  --self-test) self_test ;;
  "") scan .github/workflows/ci.yml ;;
  *) scan "$1" ;;
esac
