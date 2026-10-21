#!/usr/bin/env bash
# mq_feeder.sh -- PMAT-5843. If the merge queue on main is EMPTY, enqueue the oldest green non-draft PR
# through scripts/arm_pr_automerge.sh and post one line to the cop inbox. Run by a local systemd user timer
# every 5 min (not a workflow). Operator, verbatim (10:29Z): "a job checks MQ every 5 min; if empty, it
# enqueues the oldest green non-draft PR (arm_pr_automerge.sh) and posts to the cop inbox. Reversible, 2 h window."
# Gotcha: on a CLEAN PR `gh pr merge --squash --auto` prints "merge strategy is set by the merge queue" and
# does nothing; arm_pr_automerge.sh uses plain `gh pr merge`. This script then READS mergeQueueEntry back.
# Kill switch: touch "$STATE/disabled" (or `systemctl --user disable --now mq-feeder.timer`).
# Usage: mq_feeder.sh [--dry-run] | --self-test        Exit: 0 ok/noop, 2 ENV
set -uo pipefail
REPO="${MQ_FEEDER_REPO:-paiml/aprender}"
STATE="${MQ_FEEDER_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/mq-feeder}"
INBOX="${MQ_FEEDER_INBOX:-/mnt/nvme-raid0/cop-inbox/inbox.md}"
ARM="${MQ_FEEDER_ARM:-$(dirname "$0")/arm_pr_automerge.sh}"
TRIES="${MQ_FEEDER_TRIES:-3}"
dry=0
Q='query($o:String!,$n:String!){repository(owner:$o,name:$n){mergeQueue(branch:"main"){entries(first:1){totalCount}} pullRequests(states:OPEN,first:100,orderBy:{field:CREATED_AT,direction:ASC}){nodes{number isDraft baseRefName mergeStateStatus autoMergeRequest{enabledAt} mergeQueueEntry{position}}}}}'
# candidates JSON -> "QUEUE_BUSY" | one PR number per line, oldest first (CLEAN, non-draft, base main, not armed, not queued)
pick() {
  jq -r '.data.repository as $r
    | if ($r.mergeQueue.entries.totalCount // 0) > 0 then "QUEUE_BUSY"
      else $r.pullRequests.nodes[] | select(.isDraft==false and .baseRefName=="main" and .mergeStateStatus=="CLEAN" and .autoMergeRequest==null and .mergeQueueEntry==null) | .number end' <<<"$1"
}
inbox() { printf '%s | %s | %s | %s\n' "$(date -u +%H:%MZ)" mq-feeder PMAT-5843 "$1" >> "$INBOX"; }
if [ "${1:-}" = --self-test ]; then
  fail=0
  mk() { printf '{"data":{"repository":{"mergeQueue":{"entries":{"totalCount":%s}},"pullRequests":{"nodes":[%s]}}}}' "$1" "$2"; }
  pr() { printf '{"number":%s,"isDraft":%s,"baseRefName":"%s","mergeStateStatus":"%s","autoMergeRequest":%s,"mergeQueueEntry":null}' "$1" "$2" "$3" "$4" "$5"; }
  eq() { [ "$2" = "$3" ] || { echo "FAIL $1: got '$2' want '$3'"; fail=1; }; }
  eq 'busy queue -> nothing enqueued' "$(pick "$(mk 2 "$(pr 1 false main CLEAN null)")")" QUEUE_BUSY
  eq 'oldest clean first' "$(pick "$(mk 0 "$(pr 7 false main CLEAN null),$(pr 9 false main CLEAN null)")" | head -1)" 7
  eq 'draft skipped' "$(pick "$(mk 0 "$(pr 7 true main CLEAN null),$(pr 9 false main CLEAN null)")" | head -1)" 9
  eq 'BEHIND/BLOCKED/DIRTY skipped' "$(pick "$(mk 0 "$(pr 1 false main BEHIND null),$(pr 2 false main BLOCKED null),$(pr 3 false main DIRTY null)")")" ''
  eq 'non-main base skipped' "$(pick "$(mk 0 "$(pr 4 false feat CLEAN null)")")" ''
  eq 'already armed skipped' "$(pick "$(mk 0 "$(pr 5 false main CLEAN '{"enabledAt":"x"}')")")" ''
  eq 'no PRs' "$(pick "$(mk 0 '')")" ''
  [ "$fail" -eq 0 ] && echo "ok: self-test"; exit "$fail"
fi
[ "${1:-}" = --dry-run ] && dry=1
mkdir -p "$STATE" || exit 2
[ -e "$STATE/disabled" ] && exit 0
rem=$(gh api rate_limit --jq .resources.graphql.remaining 2>/dev/null) || exit 2
[ "${rem:-0}" -ge 500 ] || exit 0
json=$(gh api graphql -f query="$Q" -f o="${REPO%/*}" -f n="${REPO#*/}" 2>/dev/null) || exit 2
cands=$(pick "$json") || exit 2
[ "$cands" = QUEUE_BUSY ] && exit 0
[ -n "$cands" ] || exit 0
n=0
for pr in $cands; do
  grep -qx "$pr" "$STATE/kicked" 2>/dev/null && continue   # kicked by the queue once: the owner must push a fix and clear it
  n=$((n + 1)); [ "$n" -le "$TRIES" ] || break
  if [ "$dry" = 1 ]; then echo "would arm #$pr"; exit 0; fi
  out=$(bash "$ARM" "$pr" 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then
    sleep "${MQ_FEEDER_RECHECK:-30}"
    st=$(gh api graphql -f query='query($o:String!,$n:String!,$p:Int!){repository(owner:$o,name:$n){pullRequest(number:$p){state mergeQueueEntry{position} timelineItems(last:3,itemTypes:[REMOVED_FROM_MERGE_QUEUE_EVENT]){nodes{... on RemovedFromMergeQueueEvent{reason createdAt}}}}}}' -f o="${REPO%/*}" -f n="${REPO#*/}" -F p="$pr" --jq '.data.repository.pullRequest | if .state=="MERGED" then "merged" elif .mergeQueueEntry then "queued pos \(.mergeQueueEntry.position)" else "KICKED: \(.timelineItems.nodes[-1].reason // "unknown")" end' 2>/dev/null || echo unread)
    case "$st" in KICKED*) echo "$pr" >> "$STATE/kicked"; inbox "queue was empty: enqueued #$pr, then $st within ${MQ_FEEDER_RECHECK:-30}s; skipped until removed from $STATE/kicked" ;; *) inbox "queue was empty: enqueued oldest green PR #$pr ($st)" ;; esac
    exit 0
  fi
  [ "$rc" -eq 3 ] && continue   # refused on that PR (guard-tree not green on its head): try the next
  inbox "queue empty, arm of #$pr failed rc=$rc: $(printf '%s' "$out" | tail -n 1 | cut -c1-120)"; exit 0
done
inbox "queue empty, no PR enqueued: $((n > TRIES ? TRIES : n)) CLEAN candidate(s) refused by arm_pr_automerge.sh"
exit 0
