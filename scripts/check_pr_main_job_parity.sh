#!/usr/bin/env bash
# check_pr_main_job_parity.sh - RED MAIN guard: no job may run on a push to main that no
# pull_request and no merge_group run covers (ticket in the PR body; cause class "a": the job
# does not run before the merge, so main is the test bench).
#
# For every workflow with a push trigger that reaches main, each job is evaluated on three
# events: push, pull_request, merge_group. A job is on an event when the workflow triggers
# on it and the job's `if:` does not exclude it. The `if:` is read for `github.event_name`
# literals only: `!= 'X'` removes X; `== 'X'` (no `||` in the expression) keeps only the
# listed events; any `||` mix is read as "runs everywhere" (never a false red).
#
# VIOLATION: job is on push, and on neither pull_request nor merge_group.
# With --live, a job covered only by merge_group is a violation too while no active ruleset
# of the repo carries a `merge_queue` rule (the queue is off, so merge_group never fires).
# EXEMPT: ci/push-only-jobs.txt, one `<workflow-file>:<job> <reason>` per line. An entry
# whose job is not push-only, or does not exist, is stale and fails.
#
#   bash scripts/check_pr_main_job_parity.sh              # scan .github/workflows
#   bash scripts/check_pr_main_job_parity.sh --live       # also read the ruleset (gh, 1 call)
#   bash scripts/check_pr_main_job_parity.sh --self-test  # case table on throwaway workflows
#   env: PARITY_WORKFLOWS (dir), PARITY_EXEMPT (file)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WF="${PARITY_WORKFLOWS:-$ROOT/.github/workflows}"
EX="${PARITY_EXEMPT:-$ROOT/ci/push-only-jobs.txt}"

# rows: "<file>\t<job>\t<push>\t<pr>\t<mg>" with 1/0
rows() {
  local f
  for f in "$WF"/*.yml; do
    [ -e "$f" ] || continue
    yq -o=json '.' "$f" | jq -r --arg f "$(basename "$f")" '
      def lits($e; $op): [ $e | scan("github\\.event_name\\s*" + $op + "\\s*\u0027([a-z_]+)\u0027") | .[0] ];
      (.on // {}) as $on
      | (if ($on|type)=="string" then [$on] elif ($on|type)=="array" then $on else ($on|keys) end) as $ev
      | (if ($on|type)=="object" and ($on.push|type)=="object"
           then (($on.push.branches // null) as $b | ($on.push.tags // null) as $t
                 | if $b==null and $t!=null then false
                   else ($b==null or ($b|map(select(.=="main" or .=="master" or test("\\*")))|length>0)) end)
           else ($ev|index("push")!=null) end) as $pushmain
      | (.jobs // {}) | to_entries[]
      | .key as $j | ((.value.if // "")|tostring) as $e
      | ( ["push","pull_request","merge_group"]
          | map(select(. as $x | ($x=="push" and $pushmain) or ($x!="push" and ($ev|index($x)!=null))))
          | (if ($e|contains("||")) then .
             else ( lits($e;"==") as $pos | lits($e;"!=") as $neg
                    | (if ($pos|length)>0 then map(select(. as $x | $pos|index($x))) else . end)
                    | map(select(. as $x | ($neg|index($x))|not)) ) end) ) as $on
      | select($on|index("push"))
      | [$f,$j,"1",(if $on|index("pull_request") then "1" else "0" end),(if $on|index("merge_group") then "1" else "0" end)] | @tsv'
  done
}

queue_live() { # 0 = an active ruleset carries a merge_queue rule
  local ids id
  ids=$(gh api "repos/${GITHUB_REPOSITORY:-paiml/aprender}/rulesets" --jq '.[]|select(.enforcement=="active")|.id' 2>/dev/null) || return 2
  for id in $ids; do
    gh api "repos/${GITHUB_REPOSITORY:-paiml/aprender}/rulesets/$id" --jq '[.rules[].type]|index("merge_queue")' 2>/dev/null | grep -qE '^[0-9]+$' && return 0
  done
  return 1
}

scan() {
  local live="$1" fail=0 f j p pr mg key reason seen=""
  declare -A exempt used
  if [ -f "$EX" ]; then
    while read -r key reason || [ -n "$key" ]; do
      case "$key" in ''|\#*) continue ;; esac
      [ -n "$reason" ] || { echo "STALE: $key has no reason"; fail=1; }
      exempt="$exempt
$key"
    done < "$EX"
  fi
  local mgok=1
  if [ "$live" = 1 ]; then
    queue_live; case $? in 0) mgok=1 ;; 1) mgok=0 ;; *) echo "not_measured: ruleset read failed"; return 2 ;; esac
  fi
  while IFS=$'\t' read -r f j p pr mg; do
    [ -n "$f" ] || continue
    key="$f:$j"
    if [ "$pr" = 0 ] && { [ "$mg" = 0 ] || [ "$mgok" = 0 ]; }; then
      if printf "%s\n" "$exempt" | grep -qxF -- "$key"; then used="$used
$key"; continue; fi
      if [ "$mg" = 1 ]; then echo "RED: $key runs on push to main and only on merge_group, and no merge_queue rule is active"
      else echo "RED: $key runs on push to main and on no pull_request or merge_group run"; fi
      fail=1
    fi
  done < <(rows)
  for key in $(printf "%s\n" "$exempt" | tr " " "_" | grep -v "^$"); do
    printf "%s\n" "$used" | grep -qxF -- "$key" || { echo "STALE: exemption $key is not a push-only job"; fail=1; }
  done
  [ "$fail" = 0 ] && echo "OK: every job that runs on push to main also runs on a pull_request or merge_group event"
  return "$fail"
}

self_test() {
  local d fails=0 w
  d="$(mktemp -d)"; w="$d/wf"; mkdir -p "$w"
  t() { # name, expected rc, exempt content, workflow body
    printf '%s' "$4" > "$w/ci.yml"; printf '%s' "$3" > "$d/ex"
    PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; local rc=$?
    [ "$rc" = "$2" ] && echo "ok   $1" || { echo "FAIL $1 (rc $rc, want $2)"; fails=1; }
  }
  local H='on: {push: {branches: [main]}, pull_request: {}, merge_group: {}}
jobs:'
  t "push-only job is RED" 1 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
"
  t "job on pull_request is green" 0 "" "$H
  a: {runs-on: x, steps: []}
"
  t "!= pull_request but on merge_group is green" 0 "" "$H
  a: {runs-on: x, if: \"github.event_name != 'pull_request'\", steps: []}
"
  t "== push with || is read as everywhere" 0 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || x\", steps: []}
"
  t "exempt push-only job is green" 0 "ci.yml:a ticket-maker, cannot block" "$H
  a: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
"
  t "stale exemption is RED" 1 "ci.yml:a was push only" "$H
  a: {runs-on: x, steps: []}
"
  t "exemption without reason is RED" 1 "ci.yml:a" "$H
  a: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
"
  t "tag-only push is not main" 0 "" "on: {push: {tags: ['v*']}}
jobs:
  a: {runs-on: x, steps: []}
"
  t "push-only workflow, no if, is RED" 1 "" "on: {push: {branches: [main]}}
jobs:
  a: {runs-on: x, steps: []}
"
  rm -rf "${d:?}"
  return "$fails"
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --live) scan 1; exit $? ;;
  -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) scan 0; exit $? ;;
esac
