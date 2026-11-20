#!/usr/bin/env bash
# check_pr_main_job_parity.sh - RED MAIN guard: no job may run on a push to main that no
# pull_request and no merge_group run covers (ticket in the PR body; cause class "a": the job
# does not run before the merge, so main is the test bench).
#
# For every workflow with a push trigger that reaches main, each job is evaluated on three
# events: push, pull_request, merge_group. A job is on an event when the workflow triggers
# on it and the job's `if:` does not exclude it. The `if:` is read for `github.event_name`
# literals only: `!= 'X'` removes X; `== 'X'` (no `||` in the expression) keeps only the
# listed events; a `||` expression is the UNION of its disjuncts (a disjunct naming no event runs everywhere); `'X' == github.event_name`
# is read like `github.event_name == 'X'`; a negated group `!( .. )` around event_name is unreadable and fails closed (rc 2).
# A missing or empty workflow dir, a workflow that does not parse, or a workflows set with no job also fail closed (rc 2, never OK).
#
# VIOLATION 1: job is on push, and not on pull_request. A merge_group run does not count: the queue
# is a different event and may be off, and a job that first runs after the merge makes main the test bench.
# VIOLATION 2: a job's run script returns success on a pull_request before it has checked the heavy jobs
# (an `if` on the pull_request event whose body reaches `exit 0`, or `[ .. = pull_request ] && exit 0`).
# An aggregate job that passes a PR on a light check alone makes the heavy jobs optional there.
# VIOLATION 3 (draft parity): in a workflow where a heavy job (timeout-minutes >= 100 in any branch of its expression) that runs on
# push to main and on pull_request carries a `github.event.pull_request.draft` clause, every other such heavy job must carry it
# too. A heavy job without it runs on a draft PR while its siblings skip, so a draft burns the pool the others spare.
# --live is kept as an alias of the plain scan (merge_group-only coverage is now RED offline, so no ruleset read is needed).
# EXEMPT: ci/push-only-jobs.txt, one `<workflow-file>:<job> <reason>` per line. An entry
# whose job is not push-only, or does not exist, is stale and fails.
#
#   bash scripts/check_pr_main_job_parity.sh              # scan .github/workflows
#   bash scripts/check_pr_main_job_parity.sh --live       # alias of the plain scan
#   bash scripts/check_pr_main_job_parity.sh --self-test  # case table on throwaway workflows
#   env: PARITY_WORKFLOWS (dir), PARITY_EXEMPT (file)
set -uo pipefail
command -v yq >/dev/null || { echo "ENV: yq missing"; exit 2; }
command -v jq >/dev/null || { echo "ENV: jq missing"; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WF="${PARITY_WORKFLOWS:-$ROOT/.github/workflows}"
EX="${PARITY_EXEMPT:-$ROOT/ci/push-only-jobs.txt}"

# rows: "<file>\t<job>\t<push>\t<pr>\t<mg>" with 1/0
rows() {
  local f
  for f in "$WF"/*.yml "$WF"/*.yaml; do
    [ -e "$f" ] || continue
    yq -o=json '.' "$f" | jq -r --arg f "$(basename "$f")" '
      def lits($e; $op): [ $e | scan("github\\.event_name\\s*" + $op + "\\s*\u0027([a-z_]+)\u0027") | .[0] ]
                         + [ $e | scan("\u0027([a-z_]+)\u0027\\s*" + $op + "\\s*github\\.event_name") | .[0] ];
      def evs($all; $p): lits($p;"==") as $pos | lits($p;"!=") as $neg
                         | $all | (if ($pos|length)>0 then map(select(. as $x | $pos|index($x))) else . end)
                         | map(select(. as $x | ($neg|index($x))|not));
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
          | (if ($e|test("event_name")) and ($e|test("![[:space:]]*\\(")) then error("unreadable if: negated group")
             else . as $base | (($e|split("||")) | if length==0 then [""] else . end) | map(. as $p | $base | evs(.; $p)) | add | unique end) ) as $on
      | select($on|index("push"))
      | [$f,$j,"1",(if $on|index("pull_request") then "1" else "0" end),(if $on|index("merge_group") then "1" else "0" end)] | @tsv' || return 1
  done
}

# VIOLATION 2 rows: "<file>\t<job>" for each job whose run script exits 0 on a pull_request branch.
shortcuts() {
  local f j b js
  for f in "$WF"/*.yml "$WF"/*.yaml; do
    [ -e "$f" ] || continue
    js="$(yq -o=json '.' "$f" | jq -r '(.jobs // {}) | to_entries[] | [.key, ([.value.steps[]?.run // empty] | join("\n") | @base64)] | @tsv')" || return 1
    while IFS=$'\t' read -r j b; do
      [ -n "$j" ] || continue
      if printf '%s' "$b" | base64 -d | awk '
          /^[[:space:]]*#/ { next }
          /^[[:space:]]*\[.*pull_request.*\][[:space:]]*&&.*(exit[[:space:]]+0|exit[[:space:]]*([;}]|$))/ { hit = 1 }
          /^[[:space:]]*\[.*!=[[:space:]]*"?pull_request"?[[:space:]]*\][[:space:]]*\|\|.*(exit[[:space:]]+0|exit[[:space:]]*([;}]|$))/ { hit = 1 }
          /^[[:space:]]*(if|elif)[[:space:]].*pull_request/ { inif = 1; if ($0 ~ /exit[[:space:]]+0|exit[[:space:]]*([;}]|$)/) hit = 1; next }
          inif && /exit[[:space:]]+0|exit[[:space:]]*([;}]|$)/ { hit = 1 }
          inif && /^[[:space:]]*fi([[:space:];]|$)/ { inif = 0 }
          /^[[:space:]]*(\*\|)?pull_request(\|[A-Za-z_*]+)*\)/ { incase = ($0 !~ /;;/); if ($0 ~ /exit[[:space:]]+0|exit[[:space:]]*([;}]|$)/) hit = 1; next }
          incase && /exit[[:space:]]+0|exit[[:space:]]*([;}]|$)/ { hit = 1 }
          incase && /;;/ { incase = 0 }
          END { exit !hit }'; then
        printf '%s\t%s\n' "$(basename "$f")" "$j"
      fi
    done <<<"$js"
  done
}

# VIOLATION 3 rows: "<file>\t<job>" for each heavy push+PR job lacking the draft clause a heavy sibling carries.
drafts() {
  local f pr_jobs
  for f in "$WF"/*.yml "$WF"/*.yaml; do
    [ -e "$f" ] || continue
    pr_jobs="$(printf '%s\n' "$ROWS" | awk -F'\t' -v f="$(basename "$f")" '$1==f && $4=="1" {print $2}')"
    [ -n "$pr_jobs" ] || continue
    yq -o=json '.' "$f" | jq -r --arg pj "$pr_jobs" '
      ($pj | split("\n")) as $pr
      | [ (.jobs // {}) | to_entries[]
          | select(.key as $k | $pr | index($k))
          | select(((.value["timeout-minutes"] // 0) | tostring | [scan("[0-9]+") | tonumber] | max // 0) >= 100)
          | {k: .key, d: (((.value.if // "") | tostring) | contains("github.event.pull_request.draft"))} ] as $h
      | if ($h | map(select(.d)) | length) > 0 then $h[] | select(.d | not) | .k else empty end' \
      | while read -r j; do printf '%s\t%s\n' "$(basename "$f")" "$j"; done || return 1
  done
}

# Fail closed (rc 2, never OK): a missing or empty workflow dir, a workflow that does not parse, a jobs map that yields no row.
preflight() {
  local f n=0 jn=0 c
  [ -d "$WF" ] || { echo "ENV: workflow dir $WF is missing"; return 2; }
  for f in "$WF"/*.yml "$WF"/*.yaml; do
    [ -e "$f" ] || continue
    n=$((n + 1))
    c="$(yq -o=json '.' "$f" 2>/dev/null | jq -e -r 'if type == "object" then ((.jobs // {}) | length) else empty end' 2>/dev/null)" || { echo "ENV: $f does not parse as a YAML mapping"; return 2; }
    jn=$((jn + c))
  done
  [ "$n" -gt 0 ] || { echo "ENV: no workflow file under $WF"; return 2; }
  [ "$jn" -gt 0 ] || { echo "ENV: the workflows under $WF define no job"; return 2; }
}

scan() {
  local fail=0 f j p pr mg key reason SHORTS DRAFTS
  local exempt="" used=""
  preflight || return 2
  ROWS="$(rows)" || { echo "ENV: could not read the jobs and if: conditions of $WF"; return 2; }
  SHORTS="$(shortcuts)" || { echo "ENV: could not read the run scripts of $WF"; return 2; }
  DRAFTS="$(drafts)" || { echo "ENV: could not read the heavy-job draft clauses of $WF"; return 2; }
  if [ -f "$EX" ]; then
    while read -r key reason || [ -n "$key" ]; do
      case "$key" in ''|\#*) continue ;; esac
      [ -n "$reason" ] || { echo "STALE: $key has no reason"; fail=1; }
      exempt="$exempt
$key"
    done < "$EX"
  fi
  while IFS=$'\t' read -r f j p pr mg; do
    [ -n "$f" ] || continue
    key="$f:$j"
    if [ "$pr" = 0 ]; then
      if grep -qxF -- "$key" <<<"$exempt"; then used="$used
$key"; continue; fi
      if [ "$mg" = 1 ]; then echo "RED: $key runs on push to main but is skipped on pull_request by an if: event guard (merge_group does not cover it)"
      else echo "RED: $key runs on push to main and not on any pull_request run"; fi
      fail=1
    fi
  done <<<"$ROWS"
  while IFS=$'\t' read -r f j; do
    [ -n "$f" ] || continue
    echo "RED: $f:$j passes a pull_request (exit 0) before it checks the heavy jobs"
    fail=1
  done <<<"$SHORTS"
  while IFS=$'\t' read -r f j; do
    [ -n "$f" ] || continue
    echo "RED: $f:$j is a heavy job on push and pull_request without the draft clause its heavy siblings carry (a draft PR runs it)"
    fail=1
  done <<<"$DRAFTS"
  for key in $(printf "%s\n" "$exempt" | tr " " "_" | grep -v "^$"); do
    grep -qxF -- "$key" <<<"$used" || { echo "STALE: exemption $key is not a push-only job"; fail=1; }
  done
  [ "$fail" = 0 ] && echo "OK: every job that runs on push to main also runs on a pull_request, and no job passes a PR before its heavy checks"
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
  t "!= pull_request but on merge_group is RED (the queue does not cover it)" 1 "" "$H
  a: {runs-on: x, if: \"github.event_name != 'pull_request'\", steps: []}
"
  t "!= pull_request with a push-only exemption is green" 0 "ci.yml:a ticket-maker, cannot block" "$H
  a: {runs-on: x, if: \"github.event_name != 'pull_request'\", steps: []}
"
  t "draft guard only is green" 0 "" "$H
  a: {runs-on: x, if: \"github.event.pull_request.draft != true\", steps: []}
"
  t "aggregate exits 0 on a PR before the heavy checks is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            echo light passed; exit 0
          fi
          check_heavy
"
  t "aggregate one-liner [ pull_request ] && exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" = pull_request ] && exit 0
          check_heavy
"
  t "aggregate that checks light then falls through is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            [ \"\${PRL:-}\" = success ] || exit 1
            echo light passed
          fi
          check_heavy
"
  t "draft exit 0 and an echo naming pull_request are green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${DRAFT:-}\" != true ] || { echo draft; exit 0; }
          echo \"pull_request: nothing owed\"; exit 0
"
  t "heavy job missing the draft clause its sibling carries is RED" 1 "" "$H
  a: {runs-on: x, timeout-minutes: 240, if: \"github.event.pull_request.draft != true\", steps: []}
  b: {runs-on: x, timeout-minutes: 120, steps: []}
"
  t "heavy jobs both carrying the draft clause are green" 0 "" "$H
  a: {runs-on: x, timeout-minutes: 240, if: \"github.event.pull_request.draft != true\", steps: []}
  b: {runs-on: x, timeout-minutes: 120, if: \"!cancelled() && github.event.pull_request.draft != true\", steps: []}
"
  t "light job without the draft clause beside a heavy one is green" 0 "" "$H
  a: {runs-on: x, timeout-minutes: 240, if: \"github.event.pull_request.draft != true\", steps: []}
  b: {runs-on: x, timeout-minutes: 10, steps: []}
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
  t "|| guard that skips pull_request is RED" 1 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github.event_name == 'merge_group'\", steps: []}
"
  t "|| guard that keeps pull_request is green" 0 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github.event_name == 'pull_request'\", steps: []}
"
  t "reversed operand 'pull_request' != github.event_name is RED" 1 "" "$H
  a: {runs-on: x, if: \"'pull_request' != github.event_name\", steps: []}
"
  t "negated group around event_name is unreadable: rc 2" 2 "" "$H
  a: {runs-on: x, if: \"!(github.event_name == 'pull_request')\", steps: []}
"
  t "unparseable workflow fails closed (rc 2)" 2 "" "on: [push, pull_request]
jobs:
  a: {runs-on: [unclosed
"
  t "empty jobs map fails closed (rc 2)" 2 "" "$H {}
"
  t "[ != pull_request ] || { exit 0; } is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${EVT:-}\" != pull_request ] || { echo ok; exit 0; }
          check_heavy
"
  t "case pull_request) exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            pull_request) exit 0 ;;
          esac
          check_heavy
"
  t "bare exit on a pull_request if is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then exit; fi
          check_heavy
"
  t "a commented-out pull_request shortcut is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          # [ \"\$EVT\" = pull_request ] && exit 0
          check_heavy
"
  t "an exit 0 after the pull_request if block has closed is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            [ \"\${PRL:-}\" = success ] || exit 1
          fi
          [ \"\${DRAFT:-}\" != true ] || exit 0
          check_heavy
"
  t "a comment naming exit 0 inside the pull_request if block is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then
            # never exit 0 here
            [ \"\${PRL:-}\" = success ] || exit 1
          fi
          check_heavy
"
  t "an exit 0 in a later case branch, after pull_request) closed, is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            pull_request) [ \"\${PRL:-}\" = success ] || exit 1 ;;
            *) exit 0 ;;
          esac
          check_heavy
"
  t "a multi-line pull_request) branch closed by ;; before a later exit 0 branch is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            pull_request)
              [ \"\${PRL:-}\" = success ] || exit 1
              ;;
            *)
              exit 0
              ;;
          esac
          check_heavy
"
  # Fail closed on a missing tool: a clean workflow and an empty exemption file would be green,
  # so rc 2 here can only come from the ENV check (without it, no rows and no exemptions print OK).
  local sh yqp jqp rc
  sh="$(command -v bash)"; yqp="$(command -v yq)"; jqp="$(command -v jq)"
  mkdir -p "$d/noyq" "$d/nojq"; ln -sf "$jqp" "$d/noyq/jq"; ln -sf "$yqp" "$d/nojq/yq"
  te() { # name, expected rc, PATH dir
    printf '%s' "$H
  a: {runs-on: x, steps: []}
" > "$w/ci.yml"; : > "$d/ex"
    PATH="$3" PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" "$sh" "$0" >/dev/null 2>&1; local rc=$?
    [ "$rc" = "$2" ] && echo "ok   $1" || { echo "FAIL $1 (rc $rc, want $2)"; fails=1; }
  }
  te "yq missing fails closed (rc 2), clean workflow and empty exemptions" 2 "$d/noyq"
  te "jq missing fails closed (rc 2), clean workflow and empty exemptions" 2 "$d/nojq"
  : > "$d/ex"
  PARITY_WORKFLOWS="$d/missing" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; rc=$?
  [ "$rc" = 2 ] && echo "ok   missing workflow dir fails closed (rc 2)" || { echo "FAIL missing workflow dir (rc $rc, want 2)"; fails=1; }
  mkdir -p "$d/emptydir"
  PARITY_WORKFLOWS="$d/emptydir" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; rc=$?
  [ "$rc" = 2 ] && echo "ok   empty workflow dir fails closed (rc 2)" || { echo "FAIL empty workflow dir (rc $rc, want 2)"; fails=1; }
  rm -rf "${d:?}"
  return "$fails"
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --live) scan; exit $? ;;
  -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) scan; exit $? ;;
esac
