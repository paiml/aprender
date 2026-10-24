#!/usr/bin/env bash
# check_review_tier.sh -- proves scripts/review_tier.sh (FLOW-011 MT-13): its own case table passes, and each planted
# mutant of it turns that table RED. A mutant that does not apply, or survives, fails this check.
#   bash scripts/check_review_tier.sh [--self-test]   (no argument runs the same proof: guard_tree runs every guard bare)
set -uo pipefail
[[ -z ${1:-} || ${1:-} == --self-test ]] || { echo "usage: check_review_tier.sh [--self-test]" >&2; exit 2; }
SRC=scripts/review_tier.sh
TBL=.pmat/review-tiers.yaml
W=$(mktemp -d); trap 'rm -rf "${W:?}"' EXIT
fail=0

bash "$SRC" --self-test > "$W/real.log" 2>&1 || { echo "FAIL the real case table is red:"; tail -5 "$W/real.log"; exit 1; }
echo "ok    the case table passes on the real script"

mutant() { # NAME SED-EXPR [SED-EXPR]
  cp "$SRC" "$W/m.sh"
  sed -i -e "$2" ${3:+-e "$3"} "$W/m.sh"
  if cmp -s "$SRC" "$W/m.sh"; then echo "FAIL  mutant $1 did not apply -- it proves nothing"; fail=1; return; fi
  if bash "$W/m.sh" --self-test > "$W/m.log" 2>&1; then echo "FAIL  mutant $1 survived"; fail=1; else echo "ok    mutant $1 turns the table red"; fi
}
mutant highest-wins   's/    (( $(rank "$t") > $(rank "$best") )) \&\& best=$t/    best=$t/'
mutant comment-only   's/  elif comment_only "$1" "$2" "$3"; then echo light/  elif false; then echo light/'
mutant sensitive-path 's/  if matches_any "$3" <<< "$SENS"; then echo sensitive/  if false; then echo sensitive/'
mutant table-from-head 's/git show "$1:$TABLE"/git show "$2:$TABLE"/'
mutant empty-table-ok 's/  \[\[ -n $SENS \&\& -n $LIGHT \]\] ||/  true ||/'
mutant doc-comments-stripped 's#grep -Ev .^\[\[:space:\]\]\*(//(\[^/!\].\*)?)?\$. || true ;;#grep -Ev "^[[:space:]]*(//.*)?$" || true ;;#'
# A non-ASCII path is quoted by git unless told otherwise, and a quoted path cannot be read back. The fail-closed
# check keeps it from passing as comment-only; the original defect needed both parts to be wrong.
Q='s#git -c core.quotePath=false diff --name-only -z --no-renames "$1" "$2" | tr .\\0. .\\n.#git diff --name-only --no-renames "$1" "$2"#'
U='s#  git cat-file -e "$1:$3" 2>/dev/null || git cat-file -e "$2:$3" 2>/dev/null || return 1#  true#'
mutant quoted-paths "$Q"
mutant quoted-paths-and-both-unread-ok "$Q" "$U"
mutant shebang-stripped 's#awk .NR == 1 \&\& /^\#!/ { print; next } !/^\[ \\t\]\*(\#.\*)?$/.#grep -Ev "^[[:space:]]*(\#.*)?$"#'
mutant missing-table-uses-worktree 's#then missing=1; fi#then cp "$TABLE" "$tbl"; fi#'
# Table mutants: the same case table must go red when a protective row is dropped from the path table.
tmutant() { # NAME PATTERN-to-delete
  local T="$W/t.yaml"; grep -Fv -- "$2" "$TBL" > "$T"
  if cmp -s "$TBL" "$T"; then echo "FAIL  table mutant $1 did not apply"; fail=1; return; fi
  if REVIEW_TIER_TABLE="$T" bash "$SRC" --self-test > "$W/m.log" 2>&1; then echo "FAIL  table mutant $1 survived"; fail=1; else echo "ok    table mutant $1 turns the table red"; fi
}
tmutant claude-dir-dropped '".claude/*"'
tmutant agents-dir-dropped '".agents/*"'
tmutant pmat-gates-dropped '".pmat-gates.toml"'
tmutant clippy-dropped '".clippy.toml"'
tmutant roadmap-base-dropped '"docs/roadmaps/roadmap-base.yaml"'
tmutant broad-md-removed-readme '"README.md"'
mutant no-paths-ok    's/  \[\[ $n -gt 0 \]\] || die/  true || die/'
exit "$fail"
