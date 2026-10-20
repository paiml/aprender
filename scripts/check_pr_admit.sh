#!/usr/bin/env bash
# D2 RED rows for scripts/pr_admit.sh (ranked by first-ready, K cap, fail-open).
set -uo pipefail
A=scripts/pr_admit.sh
t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT
fail=0
row() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: want '$2' got '$3'"; fail=1; fi; }
# 12 active PRs, number n readyAt hour n (PR 1 oldest); PR 7 pushed again last (pushedAt ignored)
jq -n '[range(1;13) | {number:., readyAt:("2026-10-20T\(if .<10 then "0\(.)" else "\(.)" end):00:00Z"), active:true, pushedAt:"2026-10-20T23:00:00Z"}]' >"$t/l.json"
row "oldest PR is admitted"                 "admit rank=1 k=8"  "$($A decide --self 1 --k 8 "$t/l.json")"
row "rank == K is admitted"                 "admit rank=8 k=8"  "$($A decide --self 8 --k 8 "$t/l.json")"
row "rank K+1 waits"                        "wait rank=9 k=8"   "$($A decide --self 9 --k 8 "$t/l.json")"
row "a re-push keeps first-ready rank"      "admit rank=7 k=8"  "$($A decide --self 7 --k 8 "$t/l.json")"
jq 'map(if .number <= 4 then .active=false else . end)' "$t/l.json" >"$t/i.json"
row "idle PRs ahead do not count"           "admit rank=8 k=8"  "$($A decide --self 12 --k 8 "$t/i.json")"
echo 'not json' >"$t/bad.json"
row "unreadable list -> fail-open"          "admit not_measured: PR #3 not in the ranked list" "$($A decide --self 3 --k 8 "$t/bad.json")"
row "unknown PR -> fail-open"               "admit not_measured: PR #99 not in the ranked list" "$($A decide --self 99 --k 8 "$t/l.json")"
row "bad K -> fail-open"                    "admit not_measured: bad --self/--k" "$($A decide --self 3 --k x "$t/l.json")"
# wait loop: fetch failure admits; a starved wait admits at max; a later rank change admits
out=$(PR_ADMIT_FETCH=false $A wait --self 9 --k 8 --poll 0 | tail -1)
row "fetch failure -> admitted"             "admit not_measured: rank read failed" "$out"
out=$(PR_ADMIT_FETCH="cat $t/l.json" $A wait --self 9 --k 8 --max-min 0 --poll 0 | tail -1)
row "max wait reached -> admitted"          "admit not_measured: max wait 0m reached" "$out"
printf '%s' 0 >"$t/n"
f="n=\$(cat $t/n); echo \$((n+1)) >$t/n; if [ \$n -lt 2 ]; then cat $t/l.json; else cat $t/i.json; fi"
out=$(PR_ADMIT_FETCH="$f" $A wait --self 9 --k 8 --poll 0 | tr '\n' ' ')
row "waits twice then admitted when ahead PRs go idle" "wait rank=9 k=8 wait rank=9 k=8 admit rank=5 k=8 " "$out"
# workflow wiring: admit is PR-only, both heavy jobs need it, and a skipped admit (push/MG) lets them run
yq -e '.jobs.admit.if | test("pull_request")' .github/workflows/ci.yml >/dev/null 2>&1; row "admit runs on pull_request only" 0 $?
for j in x86-main workspace-test-shard; do
  yq -e ".jobs.\"$j\".needs | contains([\"admit\"])" .github/workflows/ci.yml >/dev/null 2>&1; row "$j needs admit" 0 $?
  yq -e ".jobs.\"$j\".if | test(\"needs.admit.result == 'skipped'\")" .github/workflows/ci.yml >/dev/null 2>&1; row "$j runs when admit is skipped (push/merge_group)" 0 $?
done
exit "$fail"
