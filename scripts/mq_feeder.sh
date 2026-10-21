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
# guard-tree is a section inside x86-main, not a check run on PR heads (arm_pr_automerge.sh refused every PR as "absent"): judge x86-main
export GUARD_CHECK_NAME="${GUARD_CHECK_NAME:-x86-main}"
TRIES="${MQ_FEEDER_TRIES:-3}"
dry=0
Q='query($o:String!,$n:String!){repository(owner:$o,name:$n){mergeQueue(branch:"main"){entries(first:1){totalCount}} pullRequests(states:OPEN,first:100,orderBy:{field:CREATED_AT,direction:ASC}){nodes{number isDraft baseRefName mergeStateStatus autoMergeRequest{enabledAt} mergeQueueEntry{position}}}}}'
# candidates JSON -> "QUEUE_BUSY" | one PR number per line, oldest first (CLEAN, non-draft, base main, not armed, not queued)
pick() {
  jq -r '.data.repository as $r
    | if ($r.mergeQueue.entries.totalCount // 0) > 0 then "QUEUE_BUSY"
      else $r.pullRequests.nodes[] | select(.isDraft==false and .baseRefName=="main" and .mergeStateStatus=="CLEAN" and .autoMergeRequest==null and .mergeQueueEntry==null) | .number end' <<<"$1"
}
# gate_paths FILES -> 0 when any path is gate content (a gate change needs 2 non-author seats bound to the head).
# Wide on purpose: every path under scripts/, ci/ and .github/ (the explicit-test fragments run arbitrary scripts) plus root build/lint config.
gate_paths() { grep -qE '^(\.github/|ci/|scripts/|Cargo\.toml$|Cargo\.lock$|Makefile$|rust-toolchain|\.config/nextest|\.cargo/|deny\.toml$|clippy\.toml$|\.clippy\.toml$|\.pmat-gates\.toml$)' <<<"$1"; }
# Review format the feeder accepts. Positive anchors AT THE START OF THE BODY; one review = one seat OR one receipt:
#   seat:    "seat <name> APPROVE"         (<name> a word that is not APPROVE/APPROVED)
#   quorum:  "quorum 3/3 [AGREED] PASS"    (or APPROVE/APPROVED); PASS/APPROVE is the verdict direction
# followed by end of body, punctuation, or " @<sha>". A body with a quote mark, a negation/deferral/failure word or a
# carried-over marker anywhere is rejected, so quoted evidence and "APPROVE once X lands" never count.
NEG='\b(NOT|NO|NEED|NEEDS|PENDING|REQUIRED|MISSING|AWAITING|WITHOUT|UNTIL|IF|ONCE|AFTER|WHEN|UNLESS|CANNOT|FAIL|FAILS|FAILED|REJECT|REJECTED|BLOCK|BLOCKS|BLOCKING|CARRIED|PREVIOUS|EARLIER|PRIOR)\b|REQUEST[-_ ]?CHANGES|[0-2]/3|"'"'"'`'
END='([.;:,!]|[[:space:]]+@[0-9a-f]{7,40}|[[:space:]]*$)'
# seat_names REVIEWS -> distinct seat names (one per line); dedupes by NAME, so a repost of one seat counts once.
seat_names() {
  grep -iE "^(COMMENTED|APPROVED)"$'\t'"seat[: ]+[A-Za-z0-9_.@-]+[ :,;-]+APPROVED?${END}" <<<"$1" | grep -viE "$NEG" \
    | sed -nE 's/^[A-Z]+\t[Ss][Ee][Aa][Tt][: ]+([A-Za-z0-9_.@-]+)[ :,;-]+[Aa][Pp][Pp].*/\1/p' \
    | tr 'A-Z' 'a-z' | grep -vxE 'approve|approved' | sort -u
}
# quorum_receipts REVIEWS -> count of reviews that are exactly a "quorum 3/3 [AGREED] PASS|APPROVE(D)" receipt (see above)
quorum_receipts() {
  grep -iE "^(COMMENTED|APPROVED)"$'\t'"quorum[: ]+3/3[ :,;-]+(AGREED[ :,;-]+)?(PASS|APPROVED?)${END}" <<<"$1" | grep -viE "$NEG" | wc -l
}
# needs_seats PR HEAD -> prints the reason and returns 0 unless the PR is non-gate OR holds a verified 3/3 quorum
# receipt plus 2 distinct APPROVE seat reviews bound to HEAD and no REQUEST-CHANGES bound to HEAD. FAILS CLOSED:
# an unreadable file list or review list, or an empty HEAD, means "needs seats".
needs_seats() {
  local f rv a rc q
  [ -n "$2" ] || { echo "needs seats (head unreadable)"; return 0; }
  f=$(gh api "repos/$REPO/pulls/$1/files" --paginate --jq '.[].filename' 2>/dev/null) || { echo "needs seats (files unreadable)"; return 0; }
  [ -n "$f" ] || { echo "needs seats (file list empty)"; return 0; }
  gate_paths "$f" || return 1
  # reviews bound to HEAD (commit_id), COMMENTED/APPROVED only; one body per line, newlines folded
  rv=$(gh api "repos/$REPO/pulls/$1/reviews" --paginate --jq ".[]|select(.commit_id==\"$2\" and (.state==\"COMMENTED\" or .state==\"APPROVED\" or .state==\"CHANGES_REQUESTED\"))|[.state,(.body|gsub(\"[\\n\\r]+\";\" \"))]|@tsv" 2>/dev/null) || { echo "needs seats (reviews unreadable)"; return 0; }
  rc=$(grep -ciE '^CHANGES_REQUESTED|REQUEST[-_ ]?CHANGES|\bFAIL|\bREJECT|\bBLOCKING\b' <<<"$rv")
  a=$(seat_names "$rv" | wc -l)
  q=$(quorum_receipts "$rv")
  { [ "${rc:-0}" -eq 0 ] && [ "${a:-0}" -ge 2 ] && [ "${q:-0}" -ge 1 ]; } && return 1
  echo "needs seats (gate content; bound to ${2:0:10}: $a/2 distinct named APPROVE seats, quorum 3/3 receipt $q/1, $rc request-changes)"
}
inbox() { mkdir -p "$STATE"; printf '%s %s\n' "$(date +%s)" "$1" >> "$STATE/acts.log"; printf '%s | %s | %s | %s\n' "$(date -u +%H:%MZ)" mq-feeder PMAT-5843 "$1" >> "$INBOX"; }
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
  gate_paths $'docs/x.md\nscripts/check_foo.sh' || { echo "FAIL gate_paths check_"; fail=1; }
  gate_paths $'docs/x.md\ncrates/a/src/lib.rs' && { echo "FAIL gate_paths non-gate"; fail=1; }
  for p in ci/explicit-test-commands.d/1.cmd scripts/guard_tree.sh scripts/ci_mg_reuse.sh scripts/lib/x.sh scripts/ci/x.sh scripts/mq_feeder.sh scripts/systemd/mq-feeder.service Cargo.toml Makefile ci/vendor/sovereign-ci.yml; do gate_paths "$p" || { echo "FAIL gate_paths $p"; fail=1; }; done
  # gate_paths: scripts the explicit-test fragments run are gate content (triage 38b290a66f)
  for p in scripts/contracts_gate.sh scripts/mutate_foo_guard.sh scripts/lib_baseline_ratchet.sh scripts/pv_bin.sh scripts/pmat_bin.sh scripts/tests/x.sh; do gate_paths "$p" || { echo "FAIL gate_paths $p"; fail=1; }; done
  # needs_seats with a stubbed gh: REVIEWS is "STATE<TAB>body" lines, FILES the PR file list
  gh() { case "$2" in */files) [ "${CASE:-}" = filefail ] && return 1; printf '%s\n' "${FILES:-scripts/ci_x.sh}";; */reviews) printf '%b' "${REVIEWS:-}";; esac; }
  ns() { REPO=r needs_seats 1 H >/dev/null; echo $?; }   # 0 = needs seats (not armable), 1 = armable
  Q3='COMMENTED\tquorum 3/3 AGREED PASS\n'
  eq 'seats: two named + 3/3 -> armable' "$(REVIEWS="COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w2 APPROVE\n$Q3" ns)" 1
  eq 'seats: repost of one seat (same name, different text) -> needs' "$(REVIEWS="COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w1 APPROVE again\n$Q3" ns)" 0
  eq 'seats: negated quorum "NOT 3/3" + repost -> needs' "$(REVIEWS='COMMENTED\tSeat (a) APPROVE; quorum NOT 3/3\nCOMMENTED\tSeat (a) APPROVE; quorum NOT 3/3 \nCOMMENTED\tSeat (b) APPROVE\n' ns)" 0
  eq 'seats: two seats + only a negated quorum receipt -> needs' "$(REVIEWS='COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w2 APPROVE\nCOMMENTED\tquorum NOT 3/3\n' ns)" 0
  eq 'seats: quorum 2/3 -> needs' "$(REVIEWS='COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w2 APPROVE\nCOMMENTED\tquorum 2/3 AGREED\n' ns)" 0
  eq 'seats: request-changes bound to head -> needs' "$(REVIEWS="COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w2 APPROVE\n${Q3}CHANGES_REQUESTED\tseat w3 RC\n" ns)" 0
  eq 'seats: no 3/3 receipt -> needs' "$(REVIEWS='COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w2 APPROVE\n' ns)" 0
  eq 'seats: files unreadable -> needs (fail closed)' "$(CASE=filefail ns)" 0
  eq 'seats: non-gate PR -> armable' "$(FILES=docs/x.md ns)" 1
  # round 3 (a1b0efc624): positive anchors; each row fails under the matching mutation
  S2='COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w2 APPROVE\n'
  for q in 'quorum 3/3 pending' 'quorum 3/3 required' 'quorum 3/3 missing' 'need quorum 3/3' 'AGREED 3/3' 'quorum 3/3' 'quorum NOT 3/3'; do
    eq "quorum row '$q' is not a receipt" "$(REVIEWS="${S2}COMMENTED\t$q\n" ns)" 0
  done
  eq 'quorum row "quorum 3/3 APPROVED" is a receipt' "$(REVIEWS="${S2}COMMENTED\tquorum 3/3 APPROVED\n" ns)" 1
  eq 'seat "Seat APPROVE" (no name) x2 -> needs' "$(REVIEWS="COMMENTED\tSeat APPROVE\nCOMMENTED\tSeat APPROVE\n$Q3" ns)" 0
  eq 'seat "Seat (a) APPROVE" x2 (no name) -> needs' "$(REVIEWS="COMMENTED\tSeat (a) APPROVE\nCOMMENTED\tSeat (b) APPROVE\n$Q3" ns)" 0
  eq 'seat "seat w1 APPROVE" twice with NOT 3/3 repost -> needs (named seat, not vacuous)' "$(REVIEWS='COMMENTED\tseat w1 APPROVE; quorum NOT 3/3\nCOMMENTED\tseat w1 APPROVE; quorum NOT 3/3 \n' ns)" 0
  eq 'seat names: two named' "$(seat_names "$(printf 'COMMENTED\tseat w1 APPROVE\nCOMMENTED\tSeat: W2 APPROVE\nCOMMENTED\tseat w1 APPROVE again')" | tr '\n' ,)" w1,w2,
  eq 'seat "seat APPROVE APPROVE" yields no name' "$(seat_names "$(printf 'COMMENTED\tseat APPROVE APPROVE')")" ''
  eq 'seat "seat w1 reviewing APPROVE" (no direct anchor) yields no name' "$(seat_names "$(printf 'COMMENTED\tseat w1 reviewing APPROVE')")" ''
  # round 4 (1539bf88d6): verdict direction (B1) and quoted/carried-over text (B2) must not count. Each row: two real
  # seats plus ONLY the bad quorum line -> needs; or ONE real seat plus the bad seat line + a real receipt -> needs.
  for q in 'quorum 3/3 AGREED: FAIL (B1 blocks)' 'quorum 3/3 AGREED to reject' 'quorum 3/3 AGREED on 2 BLOCKING findings' 'quorum 3/3 AGREED?' 'quorum 3/3 AGREED cannot arm yet' 'quorum 3/3 REJECT' 'quorum 3/3 AGREED' 'quorum 3/3 AGREED PASS once B1 lands' 'on "quorum 3/3 AGREED PASS" quoted' 'carried from round 3: quorum 3/3 AGREED PASS'; do
    eq "B1/B2 receipt row '$q' is not a receipt" "$(REVIEWS="${S2}COMMENTED\t$q\n" ns)" 0
  done
  eq 'receipt row with IF is not a receipt' "$(REVIEWS="${S2}COMMENTED\tquorum 3/3 PASS, if seats\n" ns)" 0
  eq 'receipt row with CANNOT is not a receipt' "$(REVIEWS="${S2}COMMENTED\tquorum 3/3 PASS; cannot arm\n" ns)" 0
  eq 'receipt row with trailing prose (END anchor) is not a receipt' "$(REVIEWS="${S2}COMMENTED\tquorum 3/3 PASS the following\n" ns)" 0
  eq 'seat row with trailing prose (END anchor) is not a seat' "$(REVIEWS="COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w9 APPROVE the following\n$Q3" ns)" 0
  for s in 'carried from round 3: seat w9 APPROVE' 'seat w9 APPROVE once B1 lands' 'seat w9 approved round 3, round 4 BLOCKING' 'on "seat w9 APPROVE" pasted evidence' 'see seat w9 APPROVE above' 'seat w9 APPROVE but FAIL'; do
    eq "B2 seat row '$s' is not a seat" "$(REVIEWS="COMMENTED\tseat w1 APPROVE\nCOMMENTED\t$s\n$Q3" ns)" 0
  done
  eq 'positive: seat with trailing @sha and receipt with PASS -> armable' "$(REVIEWS='COMMENTED\tseat w1 APPROVE @abcdef1234\nCOMMENTED\tseat w2 APPROVE.\nCOMMENTED\tquorum 3/3 PASS\n' ns)" 1
  unset -f gh
  [ "$fail" -eq 0 ] && echo "ok: self-test"; exit "$fail"
fi
[ "${1:-}" = --dry-run ] && dry=1
if [ "${1:-}" = --summary ]; then
  mkdir -p "$STATE"; since=$(($(date +%s) - 3600))  # bashrs disable-line=DET002 (runtime window, not a build stamp)
  t=$(awk -v s="$since" '$1>=s' "$STATE/ticks.log" 2>/dev/null | wc -l); e=$(awk -v s="$since" '$1>=s && /enqueued oldest/' "$STATE/acts.log" 2>/dev/null | wc -l); k=$(awk -v s="$since" '$1>=s && /KICKED/' "$STATE/acts.log" 2>/dev/null | wc -l)
  inbox "summary 1h: $t ticks, $e enqueued, $k kicked"; exit 0
fi
mkdir -p "$STATE" || exit 2
[ -x "$ARM" ] || { inbox "mq-feeder: arm script $ARM missing or not executable (install both scripts with scripts/mq_feeder_install.sh)"; exit 2; }
[ -e "$STATE/disabled" ] && exit 0
date +%s >> "$STATE/ticks.log"
rem=$(gh api rate_limit --jq .resources.graphql.remaining 2>/dev/null) || exit 2
[ "${rem:-0}" -ge 500 ] || exit 0
json=$(gh api graphql -f query="$Q" -f o="${REPO%/*}" -f n="${REPO#*/}" 2>/dev/null) || exit 2
cands=$(pick "$json") || exit 2
[ "$cands" = QUEUE_BUSY ] && exit 0
[ -n "$cands" ] || exit 0
n=0 why=""
for pr in $cands; do
  head=$(gh pr view "$pr" --repo "$REPO" --json headRefOid --jq .headRefOid 2>/dev/null || true)
  grep -qx "$pr $head" "$STATE/kicked" 2>/dev/null && continue   # kicked at THIS head: a pushed fix is a new head and clears it
  n=$((n + 1)); [ "$n" -le "$TRIES" ] || break
  if r=$(needs_seats "$pr" "$head"); then why="$why #$pr: $r;"; printf '%s skip #%s %s\n' "$(date +%s)" "$pr" "$r" >> "$STATE/acts.log"; continue; fi
  mc=$(gh api "repos/$REPO/pulls/$pr/commits" --paginate --jq '[.[]|select((.parents|length)>1)]|length' 2>/dev/null | paste -sd+ | bc 2>/dev/null || true)
  if [ "${mc:-0}" -gt 0 ] 2>/dev/null; then r="has $mc merge commit(s); REBASE needs clean commits (operator 07:57Z)"; why="$why #$pr: $r;"; printf '%s skip #%s %s\n' "$(date +%s)" "$pr" "$r" >> "$STATE/acts.log"; continue; fi
  if [ "$dry" = 1 ]; then echo "would arm #$pr"; exit 0; fi
  out=$(bash "$ARM" "$pr" 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then
    sleep "${MQ_FEEDER_RECHECK:-30}"
    st=$(gh api graphql -f query='query($o:String!,$n:String!,$p:Int!){repository(owner:$o,name:$n){pullRequest(number:$p){state mergeQueueEntry{position} timelineItems(last:3,itemTypes:[REMOVED_FROM_MERGE_QUEUE_EVENT]){nodes{... on RemovedFromMergeQueueEvent{reason createdAt}}}}}}' -f o="${REPO%/*}" -f n="${REPO#*/}" -F p="$pr" --jq '.data.repository.pullRequest | if .state=="MERGED" then "merged" elif .mergeQueueEntry then "queued pos \(.mergeQueueEntry.position)" else "KICKED: \(.timelineItems.nodes[-1].reason // "unknown")" end' 2>/dev/null || echo unread)
    case "$st" in KICKED*) echo "$pr $head" >> "$STATE/kicked"; inbox "queue was empty: enqueued #$pr, then $st within ${MQ_FEEDER_RECHECK:-30}s; skipped at this head until a new push" ;; *) inbox "queue was empty: enqueued oldest green PR #$pr ($st)" ;; esac
    exit 0
  fi
  if [ "$rc" -eq 3 ]; then why="$why #$pr: $(printf '%s' "$out" | grep -m1 -E "^refuse" | cut -c1-110);"; printf '%s skip #%s %s\n' "$(date +%s)" "$pr" "$(printf '%s' "$out" | grep -m1 -E "^refuse" | cut -c1-110)" >> "$STATE/acts.log"; continue; fi
  inbox "queue empty, arm of #$pr failed rc=$rc: $(printf '%s' "$out" | tail -n 1 | cut -c1-120)"; exit 0
done
inbox "queue empty, no PR enqueued: $((n > TRIES ? TRIES : n)) CLEAN candidate(s) refused:$why"
exit 0
