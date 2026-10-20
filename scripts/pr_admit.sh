#!/usr/bin/env bash
# D2 (FLOW-010 runner-wait): PR admission cap. A PR job enters the clean-room pool only
# when the PR's rank, by first-ready time among non-draft open PRs with checks in flight,
# is <= K. A later push keeps the PR's rank (readyAt, not pushedAt). FAIL-OPEN: any read
# or parse failure, an unknown PR, or the max wait running out admits (not_measured).
#
#   pr_admit.sh decide --self N --k K FILE.json   # -> "admit ..." | "wait rank=R"
#   pr_admit.sh wait --self N --k K [--max-min M] [--poll S]   # blocks until admitted; always exits 0
# FILE.json: [{"number":1,"readyAt":"2026-10-20T10:00:00Z","active":true}, ...]
# PR_ADMIT_FETCH overrides the fetch command (tests); default is one GraphQL call (GH-2).
set -euo pipefail

decide() {
  local self="" k="" file=""
  while [ $# -gt 0 ]; do
    case "$1" in --self) self="$2"; shift 2 ;; --k) k="$2"; shift 2 ;; *) file="$1"; shift ;; esac
  done
  case "$k$self" in '' | *[!0-9]*) echo "admit not_measured: bad --self/--k"; return 0 ;; esac
  local rank
  rank=$(jq -r --argjson s "$self" '
      [ .[] | select(.active == true or .number == $s) ] | sort_by([.readyAt, .number])
      | map(.number) | index($s) | if . == null then "none" else (. + 1) end' "$file" 2>/dev/null) || rank=""
  case "$rank" in
    '' | none) echo "admit not_measured: PR #$self not in the ranked list"; return 0 ;;
  esac
  if [ "$rank" -le "$k" ]; then echo "admit rank=$rank k=$k"; else echo "wait rank=$rank k=$k"; fi
}

fetch() {
  if [ -n "${PR_ADMIT_FETCH:-}" ]; then eval "$PR_ADMIT_FETCH"; return; fi
  local owner="${GITHUB_REPOSITORY%/*}" name="${GITHUB_REPOSITORY#*/}"
  gh api graphql -f query='query($o:String!,$n:String!){repository(owner:$o,name:$n){pullRequests(states:OPEN,first:100,baseRefName:"main"){nodes{number isDraft createdAt
      ready:timelineItems(last:1,itemTypes:READY_FOR_REVIEW_EVENT){nodes{... on ReadyForReviewEvent{createdAt}}}
      commits(last:1){nodes{commit{statusCheckRollup{state}}}}}}}}' -f o="$owner" -f n="$name" |
    jq '[.data.repository.pullRequests.nodes[] | select(.isDraft | not)
         | {number, readyAt: (.ready.nodes[0].createdAt // .createdAt),
            active: ((.commits.nodes[0].commit.statusCheckRollup.state // "") | IN("PENDING","EXPECTED"))}]'
}

wait_admit() {
  local self="" k="" max=120 poll=300 t out d
  while [ $# -gt 0 ]; do
    case "$1" in --self) self="$2"; shift 2 ;; --k) k="$2"; shift 2 ;; --max-min) max="$2"; shift 2 ;; --poll) poll="$2"; shift 2 ;; *) shift ;; esac
  done
  t=$(mktemp); trap 'rm -f "$t"' RETURN
  local deadline=$((SECONDS + max * 60))
  while :; do
    if fetch >"$t" 2>/dev/null; then out=$(decide --self "$self" --k "$k" "$t"); else out="admit not_measured: rank read failed"; fi
    echo "$out"
    case "$out" in admit*) return 0 ;; esac
    [ "$SECONDS" -lt "$deadline" ] || { echo "admit not_measured: max wait ${max}m reached"; return 0; }
    sleep "$poll"
  done
}

cmd="${1:-}"; shift || true
case "$cmd" in
  decide) decide "$@" ;;
  wait) wait_admit "$@" ;;
  *) echo "usage: pr_admit.sh decide|wait ..." >&2; exit 2 ;;
esac
