#!/usr/bin/env bash
# check_review_tier.sh -- proves scripts/review_tier.sh (FLOW-011 MT-13): its own case table passes, and each planted
# mutant of it turns that table RED. A mutant that does not apply, or survives, fails this check.
#   bash scripts/check_review_tier.sh [--self-test]   (no argument runs the same proof: guard_tree runs every guard bare)
set -uo pipefail
[[ -z ${1:-} || ${1:-} == --self-test ]] || { echo "usage: check_review_tier.sh [--self-test]" >&2; exit 2; }
SRC=scripts/review_tier.sh
W=$(mktemp -d); trap 'rm -rf "${W:?}"' EXIT
fail=0

bash "$SRC" --self-test > "$W/real.log" 2>&1 || { echo "FAIL the real case table is red:"; tail -5 "$W/real.log"; exit 1; }
echo "ok    the case table passes on the real script"

mutant() { # NAME SED-EXPR
  cp "$SRC" "$W/m.sh"
  sed -i "$2" "$W/m.sh"
  if cmp -s "$SRC" "$W/m.sh"; then echo "FAIL  mutant $1 did not apply -- it proves nothing"; fail=1; return; fi
  if bash "$W/m.sh" --self-test > "$W/m.log" 2>&1; then echo "FAIL  mutant $1 survived"; fail=1; else echo "ok    mutant $1 turns the table red"; fi
}
mutant highest-wins   's/    (( $(rank "$t") > $(rank "$best") )) \&\& best=$t/    best=$t/'
mutant comment-only   's/  elif comment_only "$1" "$2" "$3"; then echo light/  elif false; then echo light/'
mutant sensitive-path 's/  if matches_any "$3" <<< "$SENS"; then echo sensitive/  if false; then echo sensitive/'
mutant table-from-head 's/git show "$1:$TABLE"/git show "$2:$TABLE"/'
mutant empty-table-ok 's/  \[\[ -n $SENS \&\& -n $LIGHT \]\] ||/  true ||/'
mutant no-paths-ok    's/  \[\[ $n -gt 0 \]\] || die/  true || die/'
exit "$fail"
