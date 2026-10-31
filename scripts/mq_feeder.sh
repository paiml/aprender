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
# guard-tree is a section, not a check run on PR heads (arm_pr_automerge.sh refused every PR as "absent"). A PR no longer runs
# x86-main (Noah 15:09Z, #5955): pr-light runs guard-tree and the other PR-only sections, so judge pr-light
export GUARD_CHECK_NAME="${GUARD_CHECK_NAME:-pr-light}"
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
# (The request-changes word list is best-effort; the hard vetoes are the CHANGES_REQUESTED state and not arming.)
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
# bump_patch PATCH -> 0 when every changed line of a unified-diff PATCH is a `version = "X"` -> `version = "Y"` edit
# (one X, one Y, X != Y, same lines on both sides once X/Y are masked; a dependency line like
# `name = { path = "..", version = "X" }` qualifies, `rust-version`, external crate versions and checksums do not).
# Fails closed on an empty patch (GitHub omits it for large/binary diffs) and on anything else.
# mask_ver V < LINES -> each line minus its leading +/- with every literal "V" replaced by "V" (the word V), no regex
mask_ver() { awk -v v="\"$1\"" '{ $0 = substr($0, 2); o = ""; while ((i = index($0, v)) > 0) { o = o substr($0, 1, i - 1) "\"V\""; $0 = substr($0, i + length(v)) } print o $0 }'; }
bump_patch() {
  [ -n "$1" ] || return 1
  local old new xs ys x y mo mn
  old=$(grep -E '^-' <<<"$1" | grep -vE '^---' ); new=$(grep -E '^\+' <<<"$1" | grep -vE '^\+\+\+')
  [ -n "$old" ] && [ -n "$new" ] || return 1
  [ "$(wc -l <<<"$old")" -eq "$(wc -l <<<"$new")" ] || return 1
  # every changed line must be a workspace `version` edit: a bare `version = "X"` key, or a dependency line that names a
  # `path =` key (a workspace member). `serde = { version = "1" }` has no path, so an external dep bump never qualifies.
  local l
  while IFS= read -r l; do
    [[ $l =~ ^[-+]version\ *=\ *\"[0-9]+\.[0-9]+\.[0-9]+[^\"]*\" ]] && continue
    [[ $l =~ ^[-+][^#]*\{.*[{,\ ]path\ *=\ *\"[^\"]*\" ]] && [[ $l =~ [{,\ ]version\ *=\ *\"[0-9]+\.[0-9]+\.[0-9]+[^\"]*\" ]] && continue
    return 1
  done <<<"$old"$'\n'"$new"
  # a registry package (Cargo.lock `source =`/`checksum =` context) or a [*dependencies*] table is not a workspace version
  grep -qE '^ (source|checksum) *=|^[ +-]?\[[^]]*dependencies' <<<"$1" && return 1
  xs=$(grep -oE '(^[-+]|[^-_[:alnum:]])version *= *"[0-9]+\.[0-9]+\.[0-9]+[^"]*"' <<<"$old" | grep -oE '"[^"]*"' | sort -u)
  ys=$(grep -oE '(^[-+]|[^-_[:alnum:]])version *= *"[0-9]+\.[0-9]+\.[0-9]+[^"]*"' <<<"$new" | grep -oE '"[^"]*"' | sort -u)
  [ "$(wc -l <<<"$xs")" -eq 1 ] && [ "$(wc -l <<<"$ys")" -eq 1 ] && [ "$xs" != "$ys" ] || return 1
  x=${xs//\"/}; y=${ys//\"/}
  # every line must carry its own side's version and nothing else may differ. Literal compare (awk index), never a sed pattern:
  # a version holding / [ . * would otherwise break or loosen the pattern. Any failure fails closed.
  mo=$(mask_ver "$x" <<<"$old") && mn=$(mask_ver "$y" <<<"$new") || return 1
  [ -n "$mo" ] && [ "$(sort <<<"$mo")" = "$(sort <<<"$mn")" ] || return 1
  grep -qF "\"$x\"" <<<"$new" && return 1
  return 0
}
# gate_changed FILES_JSONL -> 0 when the PR changes gate content. A gate-path file is tolerated ONLY when it is a modified
# (not renamed/added/deleted) Cargo.toml / Cargo.lock whose whole patch is a version bump (bump_patch). Ruling N35 item 2.
gate_changed() {
  local line fn st pv pt
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    fn=$(jq -r '.filename' <<<"$line") || return 0
    pv=$(jq -r '.previous_filename // empty' <<<"$line") || return 0
    st=$(jq -r '.status' <<<"$line") || return 0
    if [ -n "$pv" ] && gate_paths "$pv"; then return 0; fi
    gate_paths "$fn" || continue
    [ "$st" = modified ] && [ -z "$pv" ] || return 0
    grep -qE '^(.*/)?Cargo\.(toml|lock)$' <<<"$fn" || return 0
    pt=$(jq -r '.patch // empty' <<<"$line") || return 0
    bump_patch "$pt" || return 0
  done <<<"$1"
  return 1
}
# needs_seats PR HEAD -> prints the skip reason and returns 0 (skip) unless the PR is NON-gate content AND holds a
# "quorum 3/3 [AGREED] PASS" receipt review bound to HEAD with no request-changes/FAIL/BLOCKING bound to HEAD.
# Gate-content PRs are never enqueued by the feeder (operator/arbiter arms them; 16:18Z #5849 was enqueued
# without a receipt). FAILS CLOSED: unreadable file list or review list, or an empty HEAD, means skip.
needs_seats() {
  local f fj rv rc q
  [ -n "$2" ] || { echo "skip (head unreadable)"; return 0; }
  fj=$(gh api "repos/$REPO/pulls/$1/files" --paginate --jq '.[]|{filename,status,previous_filename,patch}|tojson' 2>/dev/null) || { echo "skip (files unreadable)"; return 0; }
  f=$(jq -r '.filename, (.previous_filename // empty)' <<<"$fj" 2>/dev/null) || { echo "skip (files unparsable)"; return 0; }
  [ -n "$f" ] || { echo "skip (file list empty)"; return 0; }
  gate_changed "$fj" && { echo "skip (gate content: the feeder never enqueues a gate change)"; return 0; }
  # reviews bound to HEAD (commit_id); one body per line, newlines folded
  rv=$(gh api "repos/$REPO/pulls/$1/reviews" --paginate --jq ".[]|select(.commit_id==\"$2\" and (.state==\"COMMENTED\" or .state==\"APPROVED\" or .state==\"CHANGES_REQUESTED\"))|[.state,(.body|gsub(\"[\\n\\r]+\";\" \"))]|@tsv" 2>/dev/null) || { echo "needs seats (reviews unreadable)"; return 0; }
  rc=$(grep -ciE '^CHANGES_REQUESTED|REQUEST[-_ ]?CHANGES|\bFAIL|\bREJECT|\bBLOCKING\b|\bBLOCKER|\bBLOCKS\b|NO[-_ ]?GO|DO[-_ ]?N(OT|.?T)[-_ ]MERGE|CHANGES[-_ ]REQUESTED' <<<"$rv")
  q=$(quorum_receipts "$rv")
  { [ "${rc:-0}" -eq 0 ] && [ "${q:-0}" -ge 1 ]; } && return 1
  echo "skip (no quorum receipt at ${2:0:10}: quorum 3/3 PASS receipts $q/1, $rc request-changes)"
}
inbox() { mkdir -p "$STATE"; printf '%s %s\n' "$(date +%s)" "$1" >> "$STATE/acts.log"; printf '%s | %s | %s | %s\n' "$(date -u +%H:%MZ)" mq-feeder PMAT-5843 "$1" >> "$INBOX"; }
# ci_jobs MODE WORKFLOW -> job ids and display names under `jobs:` (MODE all), or only those a pull_request event can never run
# (MODE mq): the workflow has no pull_request trigger, or every `||` branch of the job's `if:` needs `!= 'pull_request'`,
# `== 'merge_group'` or `== 'push'` without `== 'pull_request'`, or the job `needs:` such a job (a skipped need skips it). An
# `if:` branch that names no event is PR-runnable (conservative). A job that is not under `jobs:` (e.g. `on:` keys) is never a job.
# PMAT-5843: the feeder required x86-main, a merge-queue-only job, and refused every CLEAN PR.
ci_jobs() {
  awk -v mode="$1" '
    function mqbranch(d) { return d ~ /(!= .pull_request.|== .merge_group.|== .push.)/ && d !~ /== .pull_request./ }
    function condmq(c,   n, i, parts) { if (c == "") return 0; n = split(c, parts, "[|][|]"); for (i = 1; i <= n; i++) if (!mqbranch(parts[i])) return 0; return 1 }
    /^on:/ { sec = "on"; if ($0 ~ /pull_request/) haspr = 1; next }
    /^jobs:/ { sec = "jobs"; next }
    /^[A-Za-z_]/ { sec = "other"; next }
    sec == "on" && /^  pull_request(_target)?:/ { haspr = 1 }
    sec != "jobs" { next }
    /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { id = $1; sub(/:$/, "", id); order[++nj] = id; key = ""; next }
    id == "" { next }
    /^    name:/ { v = $0; sub(/^    name:[[:space:]]*/, "", v); gsub(/["\047]/, "", v); nm[id] = v; key = ""; next }
    /^    if:/ { cond[id] = $0; key = "if"; next }
    /^    needs:/ { v = $0; sub(/^    needs:[[:space:]]*/, "", v); needs[id] = v; key = "needs"; next }
    /^    [A-Za-z0-9_-]+:/ { key = ""; next }
    key == "if" && /^      / { cond[id] = cond[id] " " $0 }
    key == "needs" && /^      / { needs[id] = needs[id] " " $0 }
    END {
      for (i = 1; i <= nj; i++) { j = order[i]; mq[j] = (!haspr) || condmq(cond[j]) }
      for (pass = 0; pass < nj; pass++) for (i = 1; i <= nj; i++) { j = order[i]; if (mq[j]) continue
        if (cond[j] ~ /always\(\)|cancelled\(\)|failure\(\)/) continue
        n = split(needs[j], dep, /[][, ]+/); for (k = 1; k <= n; k++) if (dep[k] != "" && (dep[k] in mq) && mq[dep[k]]) { mq[j] = 1; break } }
      for (i = 1; i <= nj; i++) { j = order[i]; if (mode == "all" || mq[j]) { print j; if (nm[j] != "") print nm[j] } }
    }
  ' "$2"
}
mq_only_jobs() { ci_jobs mq "$1"; }
# required_is_mq_only NAME WORKFLOW -> 0 (and the name) when the feeder's required check is a merge-queue-only job
required_is_mq_only() { local jobs; jobs="$(ci_jobs mq "$2")"; grep -qxF -e "$1" <<< "$jobs" && echo "$1"; }
# check_required WORKFLOW -> 0 when GUARD_CHECK_NAME is a job of WORKFLOW that a PR head runs; 1 (with the reason) otherwise.
check_required() {
  local wf="$1"
  [ -f "$wf" ] || { echo "check-required: workflow $wf not found"; return 2; }
  local all_jobs; all_jobs="$(ci_jobs all "$wf")"
  grep -qxF -e "$GUARD_CHECK_NAME" <<< "$all_jobs" || { echo "check-required: FAIL the required check ${GUARD_CHECK_NAME} is not a job under jobs: in $wf"; return 1; }
  required_is_mq_only "$GUARD_CHECK_NAME" "$wf" > /dev/null && { echo "check-required: FAIL the required check ${GUARD_CHECK_NAME} runs only in merge_group/push (a PR head never has it)"; return 1; }
  echo "check-required: ok ${GUARD_CHECK_NAME} runs on a PR head"; return 0
}
# fetch_main_ci -> path of main's ci.yml (MQ_FEEDER_CI_FILE overrides, for tests). The deployed copy sits in ~/.local/libexec, so the
# workflow to judge is main's, read from GitHub, never a file beside the script and never a pinned tree's own copy.
# One cached copy in $STATE with If-None-Match (FLEET RULE GH-1): a 304 costs no REST call and nothing is left in /tmp.
fetch_main_ci() {
  if [ -n "${MQ_FEEDER_CI_FILE:-}" ]; then printf '%s' "$MQ_FEEDER_CI_FILE"; return 0; fi
  mkdir -p "$STATE" || return 1
  local c="$STATE/ci.yml" e="$STATE/ci.etag" h="$STATE/ci.resp" tag="" st newtag
  [ -s "$c" ] && [ -s "$e" ] && tag="$(cat "$e")"
  gh api -i -H 'Accept: application/vnd.github.raw' ${tag:+-H "If-None-Match: $tag"} "repos/$REPO/contents/.github/workflows/ci.yml?ref=main" > "$h" 2>/dev/null
  st="$(head -n 1 "$h" | awk '{print $2}')"
  case "$st" in
    304) [ -s "$c" ] && { printf '%s' "$c"; return 0; } ;;
    200) newtag="$(awk '/^[Ee][Tt]ag:/ { sub(/^[^:]*:[ ]*/, ""); sub(/\r$/, ""); print; exit }' "$h")"
         awk 'b { print; next } /^\r?$/ { b = 1 }' "$h" > "$c.new"
         if [ -s "$c.new" ] && [ -n "$newtag" ]; then mv "$c.new" "$c" && printf '%s' "$newtag" > "$e" && { printf '%s' "$c"; return 0; }; fi ;;
  esac
  rm -f "${c:?}.new"; return 1
}
# inbox_once KEY MSG -> inbox line at most once an hour per KEY (a stuck guard ticks every 5 min; the cop reads the inbox hourly)
inbox_once() {
  mkdir -p "$STATE"
  if [ -n "$(find "$STATE/once.$1" -mmin -60 2>/dev/null)" ]; then return 0; fi
  : > "$STATE/once.$1"; inbox "$2"
}
if [ "${1:-}" = --check-required ]; then
  wf="${2:-}"; [ -n "$wf" ] || wf="$(fetch_main_ci)" || { echo "check-required: FAIL could not read main's ci.yml (not measured)"; exit 2; }
  check_required "$wf"; exit $?
fi
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
  # FILES lines: name[<TAB>previous_name[<TAB>patch_file]]; the stub emits the JSON lines the real --jq filter would
  # FILES lines: name[<TAB>previous_name|-[<TAB>patch_file|-[<TAB>status]]]; the stub emits the GitHub files array and pipes it through
  # the --jq filter needs_seats really passes, so a change to the real filter changes what the self-test sees
  gh() { local ep=$2 jf=; while [ $# -gt 0 ]; do [ "$1" = --jq ] && jf=$2; shift; done
    case "$ep" in */files) [ "${CASE:-}" = filefail ] && return 1
      printf '%s\n' "${FILES:-docs/x.md}" | while IFS=$'\t' read -r n pv pf st; do
        jq -nc --arg n "$n" --arg pv "$([ "$pv" = - ] || printf %s "$pv")" --arg p "$( [ -n "$pf" ] && [ "$pf" != - ] && cat "$pf" )" --arg st "$st" '{sha:"s",filename:$n,status:(if $st!="" then $st elif $pv!="" then "renamed" else "modified" end),additions:1,previous_filename:(if $pv!="" then $pv else null end),patch:(if $p!="" then $p else null end)}'; done | jq -sc '.' | jq -r "$jf";;
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
  # N35 item 2: a version-bump-only diff is enqueueable; a bump plus one real gate line, or a gate-only diff, is not
  FX=$(mktemp -d); trap 'rm -rf "${FX:?}"' EXIT
  printf '%s\n' '@@ -1,3 +1,3 @@' ' [workspace.package]' '-version = "0.72.2"' '+version = "0.72.3"' '@@ -9,2 +9,2 @@' '-aprender-core = { path = "crates/aprender-core", version = "0.72.2" }' '+aprender-core = { path = "crates/aprender-core", version = "0.72.3" }' > "$FX/toml.patch"
  printf '%s\n' '@@ -5,3 +5,3 @@' ' name = "aprender-core"' '-version = "0.72.2"' '+version = "0.72.3"' > "$FX/lock.patch"
  printf '%s\n' '@@ -1,2 +1,2 @@' '-version = "0.72.2"' '+version = "0.72.3"' '-lto = "thin"' '+lto = "fat"' > "$FX/toml-extra.patch"
  printf '%s\n' '@@ -1,2 +1,2 @@' '-rust-version = "1.80.0"' '+rust-version = "1.81.0"' > "$FX/rustver.patch"
  printf '%s\n' '@@ -1,2 +1,2 @@' '-serde = { version = "1.0.1" }' '+serde = { version = "1.0.2" }' '-tokio = { version = "1.0.1" }' '+tokio = { version = "1.0.3" }' > "$FX/ext.patch"
  printf '%s\n' '@@ -1,2 +1,2 @@' '-echo old' '+echo new' > "$FX/gate.patch"
  BUMP=$'Cargo.toml\t-\t'"$FX/toml.patch"$'\nCargo.lock\t-\t'"$FX/lock.patch"$'\ncrates/facades/Cargo.lock\t-\t'"$FX/lock.patch"$'\ncrates/foo/Cargo.toml\t-\t'"$FX/toml.patch"$'\nevidence/crux/0.72.3/x.json'
  eq 'bump-only diff + receipt -> armable' "$(FILES="$BUMP" REVIEWS="$Q3" ns)" 1
  eq 'bump-only diff, no receipt -> skip' "$(FILES="$BUMP" REVIEWS='' ns)" 0
  eq 'bump + one gate-file line -> skip' "$(FILES="$BUMP"$'\nscripts/release_lanes.sh\t-\t'"$FX/gate.patch" REVIEWS="$Q3" ns)" 0
  eq 'bump + gate file without patch -> skip' "$(FILES="$BUMP"$'\nscripts/release_lanes.sh' REVIEWS="$Q3" ns)" 0
  eq 'gate-only diff -> skip' "$(FILES=$'scripts/release_lanes.sh\t-\t'"$FX/gate.patch" REVIEWS="$Q3" ns)" 0
  eq 'Cargo.toml with an extra non-version line -> skip' "$(FILES=$'Cargo.toml\t-\t'"$FX/toml-extra.patch" REVIEWS="$Q3" ns)" 0
  eq 'rust-version change -> skip' "$(FILES=$'Cargo.toml\t-\t'"$FX/rustver.patch" REVIEWS="$Q3" ns)" 0
  eq 'external dependency versions -> skip' "$(FILES=$'Cargo.toml\t-\t'"$FX/ext.patch" REVIEWS="$Q3" ns)" 0
  # review 6531: single external bump, registry lock entry, non-Cargo gate file, non-modified status, regex-special versions
  printf '%s\n' '@@ -1,2 +1,2 @@' '-serde = { version = "1.0.1" }' '+serde = { version = "1.0.2" }' > "$FX/ext1.patch"
  printf '%s\n' '@@ -1,2 +1,2 @@' ' [dependencies.serde]' '-version = "1.0.1"' '+version = "1.0.2"' > "$FX/exttable.patch"
  printf '%s\n' '@@ -5,4 +5,4 @@' ' name = "serde"' '-version = "1.0.1"' '+version = "1.0.2"' ' source = "registry+https://github.com/rust-lang/crates.io-index"' > "$FX/extlock.patch"
  printf '%s\n' '@@ -1,2 +1,2 @@' '-version = "0.72.2"' '+version = "0.72.3"' > "$FX/ver.patch"
  printf '%s\n' '@@ -1,2 +1,2 @@' '-version = "0.72.2-[a/b.*"' '+version = "0.72.3-[a/b.*"' > "$FX/special.patch"
  printf '%s\n' '@@ -1,2 +1,2 @@' '-aprender-core = { path = "crates/aprender-core", version = "0.72.2" }' '+aprender-core = { path = "crates/aprender-core", version = "0.72.3" }' > "$FX/pathdep.patch"
  eq 'single external dep bump -> skip' "$(FILES=$'Cargo.toml\t-\t'"$FX/ext1.patch" REVIEWS="$Q3" ns)" 0
  eq 'external [dependencies.x] table bump -> skip' "$(FILES=$'Cargo.toml\t-\t'"$FX/exttable.patch" REVIEWS="$Q3" ns)" 0
  eq 'Cargo.lock registry package bump -> skip' "$(FILES=$'Cargo.lock\t-\t'"$FX/extlock.patch" REVIEWS="$Q3" ns)" 0
  eq 'version-only patch on a non-Cargo gate file -> skip' "$(FILES=$'scripts/x.sh\t-\t'"$FX/ver.patch" REVIEWS="$Q3" ns)" 0
  for st in added copied changed removed; do
    eq "Cargo.toml version bump with status $st -> skip" "$(FILES=$'Cargo.toml\t-\t'"$FX/ver.patch"$'\t'"$st" REVIEWS="$Q3" ns)" 0
  done
  eq 'version with / [ . * (regex-special) -> armable' "$(FILES=$'Cargo.toml\t-\t'"$FX/special.patch" REVIEWS="$Q3" ns)" 1
  eq 'workspace path dep version bump -> armable' "$(FILES=$'Cargo.toml\t-\t'"$FX/pathdep.patch" REVIEWS="$Q3" ns)" 1
  eq 'Cargo.toml with no patch (large/binary) -> skip' "$(FILES=Cargo.toml REVIEWS="$Q3" ns)" 0
  eq 'Cargo.toml renamed -> skip' "$(FILES=$'Cargo.toml\tother/Cargo.toml\t'"$FX/toml.patch" REVIEWS="$Q3" ns)" 0
  for r in 'seat w3 BLOCKER' 'NO-GO' 'seat w3 BLOCKS' 'DO NOT MERGE'; do
    eq "review '$r' at head -> skip" "$(REVIEWS="${Q3}COMMENTED\t$r\n" ns)" 0
  done
  for p in crates/foo/.clippy.toml crates/foo/clippy.toml crates/foo/deny.toml crates/foo/rust-toolchain.toml crates/foo/.cargo/config.toml; do
    eq "gate path $p + receipt -> skip" "$(FILES=$p REVIEWS="$Q3" ns)" 0
  done
  for r in 'changes-requested' 'changes_requested' 'do_not_merge' 'NO_GO' 'do-not-merge' 'NO GO' "don't merge" 'changes requested' 'do not merge'; do
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
  */files*) printf '%s\n' "$T_FILES" | jq -Rc '{filename:.,status:"modified"}';;
  */reviews*) printf '%b' "$T_REVIEWS";;
  */commits*) echo 0;;
esac
GHEOF
  printf '#!/usr/bin/env bash\necho "$1" >> "$T_MARK"\n[ "$T_ARMRC" = 3 ] && echo "refuse: guard check not green"\nexit "$T_ARMRC"\n' > "$d/arm"
  chmod +x "$d/gh" "$d/arm"
  run_feeder() { rm -f "${d:?}/mark" "${d:?}/inbox" "${d:?}/st/kicked"; : > "$d/inbox"
    T_FILES=$1 T_REVIEWS=$2 T_ARMRC=$3 T_MARK="$d/mark" PATH="$d:$PATH" MQ_FEEDER_STATE="$d/st" MQ_FEEDER_INBOX="$d/inbox" MQ_FEEDER_ARM="$d/arm" MQ_FEEDER_RECHECK=0 MQ_FEEDER_CI_FILE="${T_CI:-$(dirname "$0")/../.github/workflows/ci.yml}" bash "$0" >/dev/null 2>&1
    echo "armed=$([ -e "$d/mark" ] && echo yes || echo no) enq=$(grep -c 'enqueued oldest' "$d/inbox")"; }
  eq 'feeder: REFUSE (arm rc 3) with a receipt never enqueues' "$(run_feeder docs/x.md "$Q3" 3)" 'armed=yes enq=0'
  eq 'feeder: no receipt never calls ARM' "$(run_feeder docs/x.md '' 0)" 'armed=no enq=0'
  eq 'feeder: gate content with a receipt never calls ARM' "$(run_feeder scripts/x.sh "$Q3" 0)" 'armed=no enq=0'
  eq 'feeder: request-changes at head never calls ARM' "$(run_feeder docs/x.md "${Q3}CHANGES_REQUESTED\tx\n" 0)" 'armed=no enq=0'
  eq 'feeder control: receipt + ARM rc 0 enqueues' "$(run_feeder docs/x.md "$Q3" 0)" 'armed=yes enq=1'
  # runtime guard: etag-cached fetch (304 serves the cache), no /tmp leak, one inbox line an hour, nothing read before QUEUE_BUSY
  d2=$(mktemp -d) || exit 2
  cat > "$d2/gh" <<'GH2EOF'
#!/usr/bin/env bash
echo "$*" >> "$T_CALLS"
case "$*" in
  *If-None-Match*) printf 'HTTP/2.0 304 Not Modified\r\n\r\n';;
  *contents*) printf 'HTTP/2.0 200 OK\r\nEtag: "abc"\r\n\r\njobs:\n  pr-light:\n    runs-on: x\n';;
esac
GH2EOF
  chmod +x "$d2/gh"
  # a private TMPDIR, so the leak row counts only this fetch's own tmp.* (the shared one raced other sessions)
  mkdir "$d2/tmp" || exit 2
  fm() { TMPDIR="$d2/tmp" STATE="$d2/st" REPO=x/y T_CALLS="$d2/calls" PATH="$d2:$PATH" fetch_main_ci; }
  before=$(find "$d2/tmp" -maxdepth 1 -type f -name 'tmp.*' | wc -l)
  f1=$(fm); f2=$(fm)
  eq 'fetch: second read carries If-None-Match and serves the cache' "$(grep -c 'jobs:' "$f2")|$(wc -l < "$d2/calls")|$f1|$(grep -c 'If-None-Match: "abc"' "$d2/calls")" "1|2|$d2/st/ci.yml|1"
  eq 'fetch: no temp file is left per read' "$(find "$d2/tmp" -maxdepth 1 -type f -name 'tmp.*' | wc -l)" "$before"
  n=$(STATE="$d2/st2" INBOX="$d2/in"; for _ in 1 2 3; do inbox_once k "line"; done; wc -l < "$d2/in")
  eq 'inbox_once: three ticks, one line' "$n" 1
  # a busy queue acts on nothing, so it must not read ci.yml (GH-1: no REST call before the idle checks)
  printf '#!/usr/bin/env bash\necho "$*" >> "$T_CALLS"\ncase "$*" in *rate_limit*) echo 5000;; *"mergeQueue(branch"*) printf %%s %s;; esac\n' "'"'{"data":{"repository":{"mergeQueue":{"entries":{"totalCount":2}},"pullRequests":{"nodes":[]}}}}'"'"'' > "$d2/gh"
  : > "$d2/calls"; rm -rf "${d2:?}/st"
  T_CALLS="$d2/calls" PATH="$d2:$PATH" MQ_FEEDER_STATE="$d2/st" MQ_FEEDER_INBOX="$d2/in" MQ_FEEDER_RECHECK=0 bash "$0" > /dev/null 2>&1
  eq 'busy queue: ci.yml is not read' "$(grep -c contents "$d2/calls")" 0
  rm -rf "${d2:?}"
  # PMAT-5843: the required check must exist on a PR head. required set ∩ merge_group-only jobs = ∅, read from the real ci.yml.
  wf="$(dirname "$0")/../.github/workflows/ci.yml"
  eq 'static: the parser finds the known merge-queue-only job x86-main' "$(mq_only_jobs "$wf" | grep -cxF x86-main)" 1
  eq 'static: the parser does not call pr-light merge-queue-only' "$(mq_only_jobs "$wf" | grep -cxF pr-light)" 0
  eq 'static: the feeder requires a job that exists in ci.yml' "$(ci_jobs all "$wf" | grep -cxF -e "$GUARD_CHECK_NAME")" 1
  eq 'static: the feeder required set is disjoint from the merge-queue-only jobs' "$(required_is_mq_only "$GUARD_CHECK_NAME" "$wf")" ''
  fx="$(mktemp)"; Q="'"
  printf 'on:\n  pull_request:\n  merge_group:\njobs:\n  planted:\n    if: ${{ github.event_name != %spull_request%s }}\n  both:\n    if: ${{ github.event_name != %spull_request%s || x }}\n' "$Q" "$Q" "$Q" "$Q" > "$fx"
  printf '  allmq:\n    if: github.event_name == %smerge_group%s || github.event_name == %spush%s\n' "$Q" "$Q" "$Q" "$Q" >> "$fx"
  printf '  prpush:\n    if: github.event_name == %spull_request%s && github.event_name != %spull_request_target%s\n' "$Q" "$Q" "$Q" "$Q" >> "$fx"
  printf '  child:\n    needs: [planted]\n  grand:\n    needs:\n      - child\n  ok:\n    needs: both\n' >> "$fx"
  eq 'static: a planted merge-queue-only required check is caught' "$(required_is_mq_only planted "$fx")" planted
  eq 'static: an if with || that can run on a PR is not merge-queue-only' "$(required_is_mq_only both "$fx")" ''
  eq 'static: an if whose every || branch is merge-queue-only is caught' "$(required_is_mq_only allmq "$fx")" allmq
  eq 'static: an if that names pull_request alongside a != pull_request_target token can run on a PR' "$(required_is_mq_only prpush "$fx")" ''
  eq 'static: a job that needs a merge-queue-only job is caught' "$(required_is_mq_only child "$fx")" child
  eq 'static: needs reaches through a chain (block list)' "$(required_is_mq_only grand "$fx")" grand
  printf '  x-main:\n    if: ${{ github.event_name != %spull_request%s }}\n  y-child:\n    needs: [x-main, pr-light]\n  z-always:\n    needs: [x-main]\n    if: always()\n' "$Q" "$Q" >> "$fx"
  eq 'static: needs reaches a hyphenated job id (the real ci.yml shape)' "$(required_is_mq_only y-child "$fx")" y-child
  eq 'static: an if: always() job runs when its need is skipped, so it is not merge-queue-only' "$(required_is_mq_only z-always "$fx")" ''
  eq 'static: a job that needs a PR-runnable job is not merge-queue-only' "$(required_is_mq_only ok "$fx")" ''
  printf 'on:\n  push:\n  merge_group:\njobs:\n  pr-light:\n    runs-on: x\n' > "$fx"
  eq 'static: a workflow with no pull_request trigger is merge-queue-only' "$(required_is_mq_only pr-light "$fx")" pr-light
  printf 'on:\n  pull_request:\njobs:\n  a:\n    runs-on: x\nenv:\n  pr-light:\n    x: 1\n' > "$fx"
  eq 'static: a key outside jobs: is not a job (grep over the whole file was wrong)' "$(ci_jobs all "$fx" | grep -cxF pr-light)" 0
  eq 'static: a regex metacharacter in the name is matched literally' "$(GUARD_CHECK_NAME='pr.light' check_required "$fx" >/dev/null; echo $?)" 1
  printf 'on:\n  pull_request:\njobs:\n  prXlight:\n    runs-on: x\n' > "$fx"
  eq 'static: a dot in the name is not a wildcard' "$(GUARD_CHECK_NAME='pr.light' check_required "$fx" >/dev/null; echo $?)" 1
  rm -f "${fx:?}"
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
# Runtime guard (#6175, mandatory), only when about to act: judge main's ci.yml, not a file beside this script.
# Unreadable = not measured = do not act. Inbox lines are deduped to one an hour.
if ! cf=$(fetch_main_ci); then inbox_once ci-unread "mq-feeder: cannot read main's ci.yml; not acting (required check ${GUARD_CHECK_NAME} unverified)"; exit 2; fi
if ! cr=$(check_required "$cf"); then inbox_once ci-required "mq-feeder: $cr"; exit 2; fi
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
