#!/usr/bin/env bash
# bookkeeping_autofix.sh -- PROPOSE the bookkeeping fixes as commits (#4045 M8, the operator's shift-left order).
#
# A bookkeeping gate never blocks a publish (scripts/release/gate_classes.yaml), and it is never silently waived
# either: its fix is regenerated with the repo's OWN tool, checked against what that fix may change, and committed
# on a branch of its own for review. The working tree is left exactly as it was.
#
#   bash scripts/release/bookkeeping_autofix.sh [--push] <fixer>...     fixers: census readme claims complexity
#
#   census      pv census contracts --format json > contracts/census.json        (contract count bumps)
#   readme      bash scripts/readme_sync.sh --write                              (README's derived counts)
#   claims      bash scripts/check_no_claim_literals.sh --update                 MOVE ONLY: a literal may move or
#               leave the baseline, never arrive (scripts/lib/autofix_invariants.py claim_move_only)
#   complexity  bash scripts/check_complexity_ratchet.sh --update                SHRINK ONLY: no new function, no
#               number up (complexity_shrink_only)
#
# Per fixer, one line: `AUTOFIX <fixer>: proposed <sha> on <branch> (<n> file(s))` | `AUTOFIX <fixer>: nothing to fix`
# | `AUTOFIX <fixer>: REFUSED -- <why>` (the tree restored). --push pushes the branch; opening the PR is the watch's.
# exit 0 every fixer proposed or had nothing to do . 1 a fixer was refused or failed . 2 usage / not a clean tree
#
# Seams for the case table (never set in production): AUTOFIX_CMD_<FIXER> replaces a fixer's regeneration command.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
PROG=bookkeeping_autofix
PUSH=0; FIXERS=()
for a in "$@"; do case "$a" in --push) PUSH=1 ;; census|readme|claims|complexity) FIXERS+=("$a") ;; *) echo "$PROG: unknown '$a'" >&2; exit 2 ;; esac; done
[ "${#FIXERS[@]}" -gt 0 ] || { echo "usage: bookkeeping_autofix.sh [--push] census|readme|claims|complexity..." >&2; exit 2; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "$PROG: the tree has tracked changes -- a proposal must start clean" >&2; exit 2; }
BASE=$(git rev-parse HEAD) || exit 2
START=$(git symbolic-ref -q --short HEAD || echo "$BASE")

cmd_for() { # the regeneration command, or its seam
  local v="AUTOFIX_CMD_${1^^}"
  if [ -n "${!v:-}" ]; then printf '%s' "${!v}"; return; fi
  case "$1" in
    census) printf '%s' '. scripts/pv_bin.sh && "$PV" census contracts --format json > contracts/census.json' ;;
    readme) printf '%s' 'bash scripts/readme_sync.sh --write' ;;
    claims) printf '%s' 'bash scripts/check_no_claim_literals.sh --update' ;;
    complexity) printf '%s' 'bash scripts/check_complexity_ratchet.sh --update' ;;
  esac
}
invariant() { # invariant <fixer> -> prints why it is refused, rc 1; rc 0 admissible
  case "$1" in
    claims) python3 - "$BASE" <<'PY'
import subprocess, sys
sys.path.insert(0, "scripts/lib"); import autofix_invariants as I
base, f = sys.argv[1], "scripts/claim_literal_baseline.txt"
def entries(text):
    return [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#") and ":" in l]
def line_text(rev, path, n):
    src = subprocess.run(["git", "show", "%s:%s" % (rev, path)], capture_output=True, text=True).stdout if rev else open(path).read()
    lines = src.splitlines()
    return lines[int(n) - 1] if n.isdigit() and 0 < int(n) <= len(lines) else "<line %s absent>" % n
# the OLD literals are read where the baseline was last WRITTEN: its line numbers name lines of THAT tree, and
# reading them at HEAD after lines drifted would compare the wrong text
wrote = subprocess.run(["git", "log", "-1", "--format=%H", base, "--", f], capture_output=True, text=True).stdout.strip() or base
old = [(e.rsplit(":", 1)[0], line_text(wrote, *e.rsplit(":", 1))) for e in entries(subprocess.run(["git", "show", "%s:%s" % (base, f)], capture_output=True, text=True).stdout)]
new = [(e.rsplit(":", 1)[0], line_text(None, *e.rsplit(":", 1))) for e in entries(open(f).read())]
why = I.claim_move_only(old, new)
print("\n".join(why)); sys.exit(1 if why else 0)
PY
      ;;
    complexity) python3 - "$BASE" <<'PY'
import subprocess, sys
sys.path.insert(0, "scripts/lib"); import autofix_invariants as I
base, f = sys.argv[1], "scripts/complexity_baseline.txt"
old = I.parse_complexity(subprocess.run(["git", "show", "%s:%s" % (base, f)], capture_output=True, text=True).stdout)
new = I.parse_complexity(open(f).read())
why = I.complexity_shrink_only(old, new)
print("\n".join(why)); sys.exit(1 if why else 0)
PY
      ;;
    *) return 0 ;;   # census/readme are pure derivations of the tree: the generator IS the invariant
  esac
}
bad=0
for fx in "${FIXERS[@]}"; do
  git checkout -q "$BASE" 2> /dev/null || git checkout -q --detach "$BASE"
  if ! bash -c "$(cmd_for "$fx")" > "/tmp/.autofix-$fx.$$.log" 2>&1; then
    echo "AUTOFIX $fx: FAILED -- $(tail -1 "/tmp/.autofix-$fx.$$.log")"; bad=1
    git checkout -q -- . ; rm -f "/tmp/.autofix-$fx.$$.log"; continue
  fi
  rm -f "/tmp/.autofix-$fx.$$.log"
  changed=$(git status --porcelain --untracked-files=no | wc -l)
  if [ "$changed" = 0 ]; then echo "AUTOFIX $fx: nothing to fix"; continue; fi
  if ! why=$(invariant "$fx"); then
    echo "AUTOFIX $fx: REFUSED -- $(head -3 <<< "$why" | tr '\n' ' ')"; bad=1
    git checkout -q -- . ; continue
  fi
  br="autofix/$fx-${BASE:0:9}"
  git checkout -q -b "$br" && git add -u && \
    git -c core.hooksPath=/dev/null commit -q -m "bookkeeping autofix: $fx at ${BASE:0:9} (#4045 M8)

Proposed by scripts/release/bookkeeping_autofix.sh -- the repo's own regenerator, checked against what a
$fx fix may change. A bookkeeping gate never blocks a publish; it is fixed by a reviewed commit, never waived." || {
      echo "AUTOFIX $fx: FAILED to commit"; bad=1; git checkout -q -- . ; continue; }
  echo "AUTOFIX $fx: proposed $(git rev-parse --short=9 HEAD) on $br ($changed file(s))"
  if [ "$PUSH" = 1 ]; then git push -q -u origin "$br" 2>&1 | tail -1; fi
done
git checkout -q "$START" 2> /dev/null || git checkout -q --detach "$BASE"
exit "$bad"
