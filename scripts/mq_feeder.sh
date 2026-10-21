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
gate_paths() { grep -qE '(^|/)(Cargo\.toml$|build\.rs$|\.?clippy\.toml$|deny\.toml$|rust-toolchain|\.cargo/|\.?rustfmt\.toml$|\.gitignore$|\.?codecov\.yml$)|^(\.github/|ci/|scripts/|Cargo\.toml$|Cargo\.lock$|Makefile$|rust-toolchain|\.config/nextest|\.cargo/|deny\.toml$|clippy\.toml$|\.clippy\.toml$|\.pmat-gates\.toml$)' <<<"$1"; }
# Review format the feeder accepts (a quorum receipt only; seats are not read). Positive anchors AT THE START OF THE BODY:
#   quorum:  "quorum 3/3 [AGREED] PASS"    (or APPROVE/APPROVED); PASS/APPROVE is the verdict direction
# followed ONLY by optional punctuation and an optional " @<sha>" up to the end of the body. A body with a quote mark, a negation/deferral/failure word or a
# carried-over marker anywhere is rejected, so quoted evidence and "APPROVE once X lands" never count.
NEG='\b(NOT|NO|NEED|NEEDS|PENDING|REQUIRED|MISSING|AWAITING|WITHOUT|UNTIL|IF|ONCE|AFTER|WHEN|UNLESS|CANNOT|FAIL|FAILS|FAILED|REJECT|REJECTED|BLOCK|BLOCKS|BLOCKING|CARRIED|PREVIOUS|EARLIER|PRIOR)\b|REQUEST[-_ ]?CHANGES|[0-2]/3|"'"'"'`'
END='[.;:,!]?([[:space:]]+@[0-9a-f]{7,40})?[.;:,!]?[[:space:]]*$'
# quorum_receipts REVIEWS -> count of reviews that are exactly a "quorum 3/3 [AGREED] PASS|APPROVE(D)" receipt (see above)
quorum_receipts() {
  grep -iE "^(COMMENTED|APPROVED)"$'\t'"quorum[: ]+3/3[ :,;-]+(AGREED[ :,;-]+)?(PASS|APPROVED?)${END}" <<<"$1" | grep -viE "$NEG" | wc -l
}
# needs_seats PR HEAD -> prints the skip reason and returns 0 (skip) unless the PR is NON-gate content AND holds a
# "quorum 3/3 [AGREED] PASS" receipt review bound to HEAD with no request-changes/FAIL/BLOCKING bound to HEAD.
# Gate-content PRs are never enqueued by the feeder (operator/arbiter arms them; 16:18Z #5849 was enqueued
# without a receipt). FAILS CLOSED: unreadable file list or review list, or an empty HEAD, means skip.
needs_seats() {
  local f rv rc q
  [ -n "$2" ] || { echo "skip (head unreadable)"; return 0; }
  f=$(gh api "repos/$REPO/pulls/$1/files" --paginate --jq '.[]|.filename, (.previous_filename // empty)' 2>/dev/null) || { echo "skip (files unreadable)"; return 0; }
  [ -n "$f" ] || { echo "skip (file list empty)"; return 0; }
  gate_paths "$f" && { echo "skip (gate content: the feeder never enqueues a gate change)"; return 0; }
  # reviews bound to HEAD (commit_id); one body per line, newlines folded
  rv=$(gh api "repos/$REPO/pulls/$1/reviews" --paginate --jq ".[]|select(.commit_id==\"$2\" and (.state==\"COMMENTED\" or .state==\"APPROVED\" or .state==\"CHANGES_REQUESTED\"))|[.state,(.body|gsub(\"[\\n\\r]+\";\" \"))]|@tsv" 2>/dev/null) || { echo "needs seats (reviews unreadable)"; return 0; }
  rc=$(grep -ciE '^CHANGES_REQUESTED|REQUEST[-_ ]?CHANGES|\bFAIL|\bREJECT|\bBLOCKING\b|\bBLOCKER|\bBLOCKS\b|NO[- ]?GO|DO[- ]?N(OT|.?T)[- ]MERGE|CHANGES REQUESTED' <<<"$rv")
  q=$(quorum_receipts "$rv")
  { [ "${rc:-0}" -eq 0 ] && [ "${q:-0}" -ge 1 ]; } && return 1
  echo "skip (no quorum receipt at ${2:0:10}: quorum 3/3 PASS receipts $q/1, $rc request-changes)"
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
  # FILES lines are "name" or "name<TAB>previous_name"; the stub feeds them through the REAL --jq filter, so the filter is under test
  gh() { case "$2" in */files) [ "${CASE:-}" = filefail ] && return 1; local flt=; while [ $# -gt 0 ]; do [ "$1" = --jq ] && flt=$2; shift; done
      printf '%s\n' "${FILES:-docs/x.md}" | jq -Rn '[inputs|split("\t")|{filename:.[0]}+(if length>1 then {previous_filename:.[1]} else {} end)]' | jq -r "$flt";;
    */reviews) printf '%b' "${REVIEWS:-}";; esac; }
  ns() { REPO=r needs_seats 1 H >/dev/null; echo $?; }   # 0 = skip (not armable), 1 = armable
  Q3='COMMENTED\tquorum 3/3 AGREED PASS\n'
  eq 'receipt on a non-gate PR -> armable' "$(REVIEWS="$Q3" ns)" 1
  eq 'no receipt (non-gate PR) -> skip' "$(REVIEWS='' ns)" 0
  eq 'two seats but no receipt -> skip' "$(REVIEWS='COMMENTED\tseat w1 APPROVE\nCOMMENTED\tseat w2 APPROVE\n' ns)" 0
  eq 'gate content + receipt -> skip (never enqueued by the feeder)' "$(FILES=scripts/x.sh REVIEWS="$Q3" ns)" 0
  eq 'gate content among docs + receipt -> skip' "$(FILES=$'docs/x.md\n.github/workflows/ci.yml' REVIEWS="$Q3" ns)" 0
  eq 'files unreadable -> skip (fail closed)' "$(CASE=filefail REVIEWS="$Q3" ns)" 0
  eq 'receipt + request-changes at head -> skip' "$(REVIEWS="${Q3}CHANGES_REQUESTED\tseat w3 RC\n" ns)" 0
  eq 'receipt + FAIL review at head -> skip' "$(REVIEWS="${Q3}COMMENTED\tseat w3 FAIL\n" ns)" 0
  eq 'rename OUT of a gate path + receipt -> skip (previous_filename)' "$(FILES=$'docs/old_guard.sh\tscripts/check_guard.sh' REVIEWS="$Q3" ns)" 0
  eq 'plain non-gate rename + receipt -> armable' "$(FILES=$'docs/b.md\tdocs/a.md' REVIEWS="$Q3" ns)" 1
  for p in rustfmt.toml .rustfmt.toml .gitignore codecov.yml build.rs crates/foo/Cargo.toml crates/foo/build.rs; do
    eq "gate path $p + receipt -> skip" "$(FILES=$p REVIEWS="$Q3" ns)" 0
  done
  for r in 'seat w3 BLOCKER' 'NO-GO' 'seat w3 BLOCKS' 'DO NOT MERGE'; do
    eq "review '$r' at head -> skip" "$(REVIEWS="${Q3}COMMENTED\t$r\n" ns)" 0
  done
  for p in crates/foo/.clippy.toml crates/foo/clippy.toml crates/foo/deny.toml crates/foo/rust-toolchain.toml crates/foo/.cargo/config.toml; do
    eq "gate path $p + receipt -> skip" "$(FILES=$p REVIEWS="$Q3" ns)" 0
  done
  for r in 'do-not-merge' 'NO GO' "don't merge" 'changes requested' 'do not merge'; do
    eq "review '$r' at head -> skip" "$(REVIEWS="${Q3}COMMENTED\t$r\n" ns)" 0
  done
  eq 'receipt "quorum 3/3 APPROVED" -> armable' "$(REVIEWS='COMMENTED\tquorum 3/3 APPROVED\n' ns)" 1
  eq 'receipt with trailing @sha -> armable' "$(REVIEWS='COMMENTED\tquorum 3/3 AGREED PASS @abcdef1234\n' ns)" 1
  eq 'receipt with punctuation -> armable' "$(REVIEWS='COMMENTED\tquorum 3/3 AGREED PASS.\n' ns)" 1
  eq 'receipt with punctuation then @sha -> armable' "$(REVIEWS='COMMENTED\tquorum 3/3 AGREED PASS. @abcdef1234\n' ns)" 1
  for q in 'quorum 3/3 pending' 'quorum 3/3 required' 'quorum 3/3 missing' 'need quorum 3/3' 'AGREED 3/3' 'quorum 3/3' 'quorum NOT 3/3' 'quorum 2/3 AGREED PASS' \
      'quorum 3/3 AGREED: FAIL (B1 blocks)' 'quorum 3/3 AGREED to reject' 'quorum 3/3 AGREED on 2 BLOCKING findings' 'quorum 3/3 AGREED?' 'quorum 3/3 AGREED cannot arm yet' \
      'quorum 3/3 REJECT' 'quorum 3/3 PASS, if seats' 'quorum 3/3 PASS; cannot arm' 'quorum 3/3 PASS the following' 'quorum 3/3 PASS. Nothing was verified by me' \
      'quorum 3/3 PASS @abcdef1234 (RC on a later head)' 'carried from round 3: quorum 3/3 PASS' 'quorum 3/3 PASS once B1 lands' 'see quorum 3/3 PASS above' 'on "quorum 3/3 PASS" pasted evidence' 'quorum 3/3 PASS but FAIL'; do
    eq "row '$q' is not a receipt" "$(REVIEWS="COMMENTED\t$q\n" ns)" 0
  done
  unset -f gh
  # subprocess: the whole feeder against a stub gh and a stub ARM. REFUSE (rc 3), a missing receipt and gate content must NEVER
  # call ARM or post "enqueued"; an ARM rc 0 control with a receipt does enqueue (proves the harness can see an enqueue).
  d=$(mktemp -d) || exit 2
  cat > "$d/gh" <<'GHEOF'
#!/usr/bin/env bash
case "$*" in
  *rate_limit*) echo 5000;;
  *"mergeQueue(branch"*) printf '{"data":{"repository":{"mergeQueue":{"entries":{"totalCount":0}},"pullRequests":{"nodes":[{"number":7,"isDraft":false,"baseRefName":"main","mergeStateStatus":"CLEAN","autoMergeRequest":null,"mergeQueueEntry":null}]}}}}';;
  *graphql*) echo QUEUED;;
  "pr view"*) echo deadbeefdeadbeef;;
  */files*) printf '%s\n' "$T_FILES";;
  */reviews*) printf '%b' "$T_REVIEWS";;
  */commits*) echo 0;;
esac
GHEOF
  printf '#!/usr/bin/env bash\necho "$1" >> "$T_MARK"\n[ "$T_ARMRC" = 3 ] && echo "refuse: guard check not green"\nexit "$T_ARMRC"\n' > "$d/arm"
  chmod +x "$d/gh" "$d/arm"
  run_feeder() { rm -f "${d:?}/mark" "${d:?}/inbox" "${d:?}/st/kicked"; : > "$d/inbox"
    T_FILES=$1 T_REVIEWS=$2 T_ARMRC=$3 T_MARK="$d/mark" PATH="$d:$PATH" MQ_FEEDER_STATE="$d/st" MQ_FEEDER_INBOX="$d/inbox" MQ_FEEDER_ARM="$d/arm" MQ_FEEDER_RECHECK=0 bash "$0" >/dev/null 2>&1
    echo "armed=$([ -e "$d/mark" ] && echo yes || echo no) enq=$(grep -c 'enqueued oldest' "$d/inbox")"; }
  eq 'feeder: REFUSE (arm rc 3) with a receipt never enqueues' "$(run_feeder docs/x.md "$Q3" 3)" 'armed=yes enq=0'
  eq 'feeder: no receipt never calls ARM' "$(run_feeder docs/x.md '' 0)" 'armed=no enq=0'
  eq 'feeder: gate content with a receipt never calls ARM' "$(run_feeder scripts/x.sh "$Q3" 0)" 'armed=no enq=0'
  eq 'feeder: request-changes at head never calls ARM' "$(run_feeder docs/x.md "${Q3}CHANGES_REQUESTED\tx\n" 0)" 'armed=no enq=0'
  eq 'feeder control: receipt + ARM rc 0 enqueues' "$(run_feeder docs/x.md "$Q3" 0)" 'armed=yes enq=1'
  rm -rf "${d:?}"
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
