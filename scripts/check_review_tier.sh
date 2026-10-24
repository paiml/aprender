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
tmutant() { # NAME PATTERN-to-delete ROW (the case row that must be the one to fail)
  local T="$W/t.yaml"; grep -Fv -- "$2" "$TBL" > "$T"
  if cmp -s "$TBL" "$T"; then echo "FAIL  table mutant $1 did not apply"; fail=1; return; fi
  if REVIEW_TIER_TABLE="$T" bash "$SRC" --self-test > "$W/m.log" 2>&1; then echo "FAIL  table mutant $1 survived"; fail=1
  elif ! grep -aq "^FAIL $3:" "$W/m.log"; then echo "FAIL  table mutant $1 is red but row $3 did not fail (red for the wrong reason)"; fail=1
  else echo "ok    table mutant $1 turns row $3 red"; fi
}
# Control: an UNMUTATED copy passed the same way must pass, or every table mutant below is red for the wrong reason.
cp "$TBL" "$W/control.yaml"
if REVIEW_TIER_TABLE="$W/control.yaml" bash "$SRC" --self-test > "$W/c.log" 2>&1; then echo "ok    control: an unmutated table copy passes"; else echo "FAIL  control: an unmutated table copy is red:"; tail -3 "$W/c.log"; fail=1; fi
tmutant claude-dir-dropped '".claude/*"' gate-read-claude-md
tmutant agents-dir-dropped '".agents/*"' agents-dir
tmutant pmat-gates-dropped '".pmat-*"' pmat-gates-toml
tmutant clippy-dropped '".clippy.toml"' clippy-toml
tmutant roadmap-base-dropped '"docs/roadmaps/roadmap-base.yaml"' roadmap-base
tmutant broad-md-removed-readme '"README.md"' readme-light
tmutant docs-md-dropped '"docs/*.md"' docs-only
tmutant docs-audit-dropped '"docs/audits/*"' docs-audit-receipt
tmutant codeowners-dropped '"*CODEOWNERS"' docs-codeowners
tmutant scripts-dropped '"scripts/*"' guard-tree
tmutant github-dropped '".github/*"' pr-review-key
tmutant nextest-config-dropped '".config/*"' nextest-config
# Surface scan: every file a CI surface names under the gate directories must be sensitive in the table, so a new
# gate script cannot sit outside it. (Derived from the tree, not a list kept here.)
gaps=$(cat .github/workflows/*.yml ci/sections.yml ci/explicit-test-commands.d/*.cmd Makefile 2>/dev/null \
  | grep -aoE '(scripts|ci|tools|schemas|\.pmat|\.github|\.config|\.cargo)/[A-Za-z0-9_./-]+' | sort -u \
  | while IFS= read -r f; do [[ -f $f ]] && [[ $(bash "$SRC" --classify "$f") != sensitive ]] && echo "$f"; done)
if [[ -n $gaps ]]; then echo "FAIL  CI-named files outside the sensitive list:"; echo "$gaps" | head; fail=1; else echo "ok    every gate file a CI surface names is sensitive"; fi
mutant mode-change-ignored 's#  \[\[ $(git ls-tree "$1" -- "$3" | cut -d. . -f1) == .*|| return 1#  true#'
mutant malformed-line-dropped 's#{ print "review_tier: malformed table line: " $0 > "/dev/stderr"; bad = 1; exit 2 }#{ next }#'
mutant read-by-code-ignored 's#if read_by_code "$1" "$2" "$3"; then#if false; then#'
mutant basename-include-ignored 's#-e "\[\\"/\]\$b\\""#-e "NOPE$b"#'
mutant include-per-line-only 's#--all-match -e .include(_str|_bytes)?!. -e "\[\\"/\]\$b\\""#-e "include(_str|_bytes)?!.*[\\"/]$b\\""#'
mutant no-paths-ok    's/  \[\[ $n -gt 0 \]\] || die/  true || die/'
exit "$fail"
