#!/usr/bin/env bash
# check_one_pr_per_ticket.sh -- one PR per PMAT ticket, enforced (PMAT-6102, EMG-001).
#
# Noah 2026-10-23 07:12Z, item c: "one PR per ticket, enforced". Five-whys terminal:
# no check refused a duplicate PR per ticket (#6096 and #6098 both hit it).
#
#   bash scripts/check_one_pr_per_ticket.sh --pr <N> [--repo owner/name]   # live, GraphQL read
#   bash scripts/check_one_pr_per_ticket.sh --json <file>                   # judge a snapshot
#   bash scripts/check_one_pr_per_ticket.sh --self-test   # also what a bare run does
#
# RULES (a PR's tickets are its `PMAT-Ticket: PMAT-<n>` commit trailers):
#   R1  another OPEN PR (draft included) carries the same ticket -> RED, unless this PR's body says
#       "supersedes #M" for that PR AND #M is already closed. A named PR that is still open is RED:
#       close it in the same step, then re-run.
#   R2  more than one commit on this PR carries the same ticket -> RED (EMG-7: one commit per ticket).
#
# EXIT: 0 green, 1 red, 2 not measured (read failed: L23, no decision is made on it; CI fails closed).
# WIRING: none in CI (quorum 3/3 option B, triage-quorum-6107-gate-0741Z.md). arbiter runs it at PR open.
# UNDO: delete this script. Nothing else reads it.
# The snapshot is {pr, body, commits:[msg], open:[{number, isDraft, body, commits:[msg]}]}.
# `open` is every open PR, this one included; "closed" is simply "absent from open".
set -euo pipefail

JQ_JUDGE='
  def tickets: [.[] | capture("(?m)^PMAT-Ticket:[ \t]*(?<t>PMAT-[0-9]+)[ \t]*$").t];
  . as $s
  | ($s.commits | tickets) as $mine
  | ([$s.body | scan("(?i)supersedes[ \t]+#([0-9]+)") | .[0] | tonumber]) as $sup
  | [ ($mine | group_by(.)[] | select(length > 1) | "R2 \(.[0]) is on \(length) commits of this PR"),
      ( $s.open[] | select(.number != $s.pr) as $o
        | ($o.commits | tickets) as $theirs
        | ($mine | unique | .[] | select(. as $t | $theirs | index($t))) as $t
        | if ($sup | index($o.number)) then "R1 \($t): #\($o.number) is named in \"supersedes\" but is still open; close it"
          else "R1 \($t) is also carried by open PR #\($o.number)\(if $o.isDraft then " (draft)" else "" end); add \"supersedes #\($o.number)\" and close it" end ) ] | .[]'

judge() { # <snapshot.json>; prints RED lines; rc 0 green, 1 red
  local out
  out=$(jq -r "$JQ_JUDGE" "$1") || return 2
  if [ -n "$out" ]; then printf '%s\n' "$out"; return 1; fi
  return 0
}

live() { # <repo> <pr>; writes a snapshot to stdout
  local repo=$1 pr=$2 owner name raw
  owner=${repo%%/*}; name=${repo##*/}
  # shellcheck disable=SC2016
  raw=$(gh api graphql -f owner="$owner" -f name="$name" -F pr="$pr" -f query='
    query($owner:String!,$name:String!,$pr:Int!){
      rateLimit{remaining resetAt}
      repository(owner:$owner,name:$name){
        pullRequest(number:$pr){number body commits(first:100){nodes{commit{message}}}}
        pullRequests(states:OPEN,first:100){nodes{number isDraft body commits(first:100){nodes{commit{message}}}}}
      }}') || return 2
  jq -e '.data.repository.pullRequest != null and (.data.repository.pullRequests.nodes | type == "array")' <<<"$raw" >/dev/null || return 2
  jq '.data.repository as $r | {pr: $r.pullRequest.number, body: ($r.pullRequest.body // ""),
      commits: [$r.pullRequest.commits.nodes[].commit.message],
      open: [$r.pullRequests.nodes[] | {number, isDraft, body: (.body // ""), commits: [.commits.nodes[].commit.message]}]}' <<<"$raw"
}

self_test() {
  local td fails=0 rc out
  td=$(mktemp -d)
  local T6=$'fix: x\n\nPMAT-Ticket: PMAT-6000\nCo-Authored-By: a'
  local T7=$'fix: y\n\nPMAT-Ticket: PMAT-7000'
  mk() { # <name> <pr> <body> <this-commits-json> <open-json>
    jq -n --argjson pr "$2" --arg body "$3" --argjson c "$4" --argjson o "$5" '{pr:$pr, body:$body, commits:$c, open:$o}' >"$td/$1.json"
  }
  row() { # <name> <want-rc> <grep-pattern-or-empty> <script>
    local name=$1 want=$2 pat=$3 script=$4
    set +e; out=$(bash "$script" --json "$td/$name.json" 2>&1); rc=$?; set -e
    if [ "$rc" != "$want" ] || { [ -n "$pat" ] && ! grep -Eq -- "$pat" <<<"$out"; }; then
      printf 'FAIL %s (%s): rc=%s want %s: %s\n' "$name" "$script" "$rc" "$want" "$out"; fails=$((fails + 1))
    else printf 'ok   %s\n' "$name"; fi
  }
  local t6=$(jq -n --arg a "$T6" '[$a]') t7=$(jq -n --arg a "$T7" '[$a]')
  mk alone 10 "" "$t6" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6}]"
  mk other-ticket 10 "" "$t6" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6},{\"number\":11,\"isDraft\":false,\"body\":\"\",\"commits\":$t7}]"
  mk dup-open 10 "" "$t6" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6},{\"number\":11,\"isDraft\":false,\"body\":\"\",\"commits\":$t6}]"
  mk dup-draft 10 "" "$t6" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6},{\"number\":11,\"isDraft\":true,\"body\":\"\",\"commits\":$t6}]"
  mk sup-closed 10 "Supersedes #11, closed." "$t6" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6}]"
  mk sup-open 10 "supersedes #11" "$t6" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6},{\"number\":11,\"isDraft\":false,\"body\":\"\",\"commits\":$t6}]"
  mk sup-wrong 10 "supersedes #12" "$t6" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6},{\"number\":11,\"isDraft\":false,\"body\":\"\",\"commits\":$t6}]"
  mk two-commits 10 "" "$(jq -n --arg a "$T6" '[$a,$a]')" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6}]"
  mk two-tickets 10 "" "$(jq -n --arg a "$T6" --arg b "$T7" '[$a,$b]')" "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":$t6}]"
  mk no-trailer 10 "" '["chore: nothing"]' "[{\"number\":10,\"isDraft\":false,\"body\":\"\",\"commits\":[\"chore: nothing\"]},{\"number\":11,\"isDraft\":false,\"body\":\"\",\"commits\":[\"chore: nothing\"]}]"
  local S=${BASH_SOURCE[0]}
  row alone 0 "" "$S"
  row other-ticket 0 "" "$S"
  row dup-open 1 '^R1 PMAT-6000 .*#11' "$S"
  row dup-draft 1 '\(draft\)' "$S"
  row sup-closed 0 "" "$S"
  row sup-open 1 'still open' "$S"
  row sup-wrong 1 '^R1 PMAT-6000 .*#11' "$S"
  row two-commits 1 '^R2 PMAT-6000 is on 2 commits' "$S"
  row two-tickets 0 "" "$S"
  row no-trailer 0 "" "$S"
  # MUTANTS: each rule removed must turn its row green-when-it-should-be-red (so the table above would catch it).
  sed 's/^      ( \$s.open\[\]/      ( empty ) , ( $s.open[]|empty ) , ( $s.open[]|select(false)/' "$S" >"$td/m-r1.sh"
  sed 's/select(length > 1)/select(length > 99)/' "$S" >"$td/m-r2.sh"
  sed 's/(\$sup | index(\$o.number)) then/($sup | index($o.number)) then empty elif false then/' "$S" >"$td/m-open.sh"
  for m in m-r1 m-r2 m-open; do
    cmp -s "$S" "$td/$m.sh" && { printf 'FAIL mutant %s equals the script\n' "$m"; fails=$((fails + 1)); }
  done
  row dup-open 0 "" "$td/m-r1.sh" && true
  row two-commits 0 "" "$td/m-r2.sh" && true
  row sup-open 0 "" "$td/m-open.sh" && true
  # a mutant that stays red would mean the table cannot see the rule: the rows above report FAIL for it.
  rm -rf "${td:?}"
  [ "$fails" = 0 ] && { echo "self-test: all rows pass"; return 0; }
  echo "self-test: $fails FAILED"; return 1
}

main() {
  local mode="" pr="" repo="${GITHUB_REPOSITORY:-paiml/aprender}" json=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --self-test) mode=self ;;
      --pr) pr=$2; mode=live; shift ;;
      --repo) repo=$2; shift ;;
      --json) json=$2; mode=json; shift ;;
      *) echo "usage: $0 --pr N [--repo R] | --json FILE | --self-test" >&2; return 2 ;;
    esac
    shift
  done
  [ -z "$mode" ] && [ $# -eq 0 ] && mode=self # bare run (guard-tree "[run]") = self-test, no network
  case "$mode" in
    self) self_test ;;
    json) judge "$json" ;;
    live)
      local snap rc=0; snap=$(mktemp)
      live "$repo" "$pr" >"$snap" || { rm -f "${snap:?}"; echo "not measured: GraphQL read of $repo#$pr failed (no decision made)" >&2; return 2; }
      judge "$snap" || rc=$?
      rm -f "${snap:?}"
      [ "$rc" = 0 ] && echo "one-PR-per-ticket: green ($repo#$pr)"
      return "$rc" ;;
    *) echo "usage: $0 --pr N | --json FILE | --self-test" >&2; return 2 ;;
  esac
}
main "$@"
