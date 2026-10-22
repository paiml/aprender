#!/usr/bin/env bash
# LAB (#5879): the coverage scan updates ONE report (epic #5875), it never files a ticket per function.
# RED when an open issue titled "test: cover ..." is not named in the epic body (the epic lists each row's
# original ticket as "was #N" while a live PR keeps it open). Usage: check_coverage_single_report.sh [--self-test]
set -euo pipefail
EPIC="${COVERAGE_EPIC:-5875}"
check() { # $1 = file of "number<TAB>title" (open issues)  $2 = epic body file ; prints offenders, returns 1 on any
  local n t bad=0
  while IFS=$'\t' read -r n t; do
    case "$t" in "test: cover"*) grep -qE "#$n([^0-9]|$)" "$2" || { echo "RED  #$n '$t' is a per-function ticket not rolled into epic #$EPIC"; bad=1; } ;; esac
  done < "$1"
  [ "$bad" = 0 ] && echo "ok    every open 'test: cover' issue is named in epic #$EPIC"
  return "$bad"
}
if [ "${1:-}" = --self-test ]; then
  t="$(mktemp -d)"; trap 'rm -rf -- "${t:?}"' EXIT
  printf -- '- [ ] fn a (was #11; live PR)\n' > "$t/epic"
  printf '11\ttest: cover never-executed fn in a.rs\n12\tfix: other\n' > "$t/ok"
  check "$t/ok" "$t/epic" > /dev/null || { echo "self-test: a rolled-in ticket must pass"; exit 1; }
  printf '11\ttest: cover a\n13\ttest: cover never-executed fn in b.rs\n' > "$t/bad"
  check "$t/bad" "$t/epic" > /dev/null && { echo "self-test: a new per-function ticket must be RED"; exit 1; }
  printf '110\ttest: cover c\n' > "$t/pre"
  check "$t/pre" "$t/epic" > /dev/null && { echo "self-test: #110 must not match #11"; exit 1; }
  echo "PASS  check_coverage_single_report self-test: 3 rows"; exit 0
fi
cd "$(dirname "$0")/.."
t="$(mktemp -d)"; trap 'rm -rf -- "${t:?}"' EXIT
gh issue view "$EPIC" --json body --jq .body > "$t/epic"
# Reads go through GraphQL (GH-2), 100 an page, so a repo past 500 open issues is read whole, not cut off.
repo="${GH_REPO:-${GITHUB_REPOSITORY:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}}"
cursor=""; : > "$t/open"
while :; do
  gh api graphql -F owner="${repo%%/*}" -F name="${repo##*/}" ${cursor:+-F cursor="$cursor"} -f query='
    query($owner:String!,$name:String!,$cursor:String){repository(owner:$owner,name:$name){
      issues(states:OPEN,first:100,after:$cursor){nodes{number title} pageInfo{hasNextPage endCursor}}}}' > "$t/page"
  jq -r '.data.repository.issues.nodes[]|"\(.number)\t\(.title)"' "$t/page" >> "$t/open"
  [ "$(jq -r '.data.repository.issues.pageInfo.hasNextPage' "$t/page")" = true ] || break
  cursor="$(jq -r '.data.repository.issues.pageInfo.endCursor' "$t/page")"
done
check "$t/open" "$t/epic"
