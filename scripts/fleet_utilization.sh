#!/usr/bin/env bash
# fleet_utilization.sh — the packing table the operator asked for on every iteration (2026-09-11):
# per box (intel / gx10 / yoga): runners, busy, busy with APRENDER jobs, aprender share vs the 80 / 80 / 50 targets,
# plus the aprender jobs still waiting for a runner and the label set they ask for. Measured from the org runner
# list and the jobs of aprender's queued + in-progress runs; nothing here writes to GitHub.
#
#   bash scripts/fleet_utilization.sh [--repo O/R] [--org ORG] [--selftest]
set -euo pipefail

REPO="paiml/aprender"; ORG="paiml"; SELFTEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --org) ORG="$2"; shift 2 ;;
    --selftest) SELFTEST=1; shift ;;
    *) echo "fleet_utilization: unknown argument $1" >&2; exit 2 ;;
  esac
done

# render RUNNERS_JSON JOBS_JSON -> the table on stdout. RUNNERS_JSON = {"runners":[{name,busy}]},
# JOBS_JSON = {"jobs":[{name,status,runner_name,labels}]} (every job of every queued + in-progress run).
render() {
  jq -r -n --argjson r "$1" --argjson j "$2" '
    def box(n): if n == null then "none" elif (n|startswith("intel")) then "intel"
      elif (n|startswith("gx10")) then "gx10" elif (n|startswith("yoga")) then "yoga" else "other" end;
    def target(b): {intel:80, gx10:80, yoga:50}[b];
    ["box","runners","busy","aprender","share","target"] as $h |
    ($h | join("\t")),
    ( ["intel","gx10","yoga"][] as $b |
      ([$r.runners[] | select(.name|startswith($b))] | length) as $cap |
      ([$r.runners[] | select(.name|startswith($b)) | select(.busy)] | length) as $busy |
      ([$j.jobs[] | select(.status=="in_progress") | select(box(.runner_name)==$b)] | length) as $apr |
      [$b, $cap, $busy, $apr, (if $cap>0 then (($apr*100/$cap)|floor|tostring)+"%" else "n/a" end), (target($b)|tostring)+"%"] | join("\t") ),
    ("queued\t" + ([$j.jobs[] | select(.status=="queued")] | length | tostring)),
    ( [$j.jobs[] | select(.status=="queued") | (.labels // []) | join(",")] | group_by(.) | map("  \(length)\t[\(.[0])]") | .[] )'
}

if [ "$SELFTEST" = 1 ]; then
  runners=$(cat <<'JSON'
{"runners":[{"name":"intel-clean-room","busy":true},{"name":"intel-clean-room-2","busy":true},{"name":"intel-clean-room-3","busy":false},
{"name":"gx10-pool1","busy":false},{"name":"gx10-pool2","busy":true},{"name":"yoga-build","busy":true},{"name":"yoga-build2","busy":false}]}
JSON
)
  jobs=$(cat <<'JSON'
{"jobs":[{"name":"workspace-test","status":"in_progress","runner_name":"intel-clean-room","labels":["self-hosted","X64","clean-room"]},
{"name":"guard-tree","status":"in_progress","runner_name":"yoga-build","labels":["self-hosted","X64","clean-room"]},
{"name":"ci / test","status":"in_progress","runner_name":"gx10-pool2","labels":["self-hosted","clean-room"]},
{"name":"gate","status":"queued","runner_name":null,"labels":["self-hosted","X64","clean-room"]},
{"name":"other-repo-looking","status":"completed","runner_name":"intel-clean-room-2","labels":[]}]}
JSON
)
  out=$(render "$runners" "$jobs"); err=0
  chk() { if printf '%s\n' "$out" | grep -qF "$2"; then echo "ok    $1"; else echo "FAIL  $1: wanted [$2]"; err=1; fi; }
  chk "intel: 3 runners, 2 busy, 1 aprender = 33%"   "$(printf 'intel\t3\t2\t1\t33%%\t80%%')"
  chk "gx10: 2 runners, 1 busy, 1 aprender = 50%"    "$(printf 'gx10\t2\t1\t1\t50%%\t80%%')"
  chk "yoga: 2 runners, 1 busy, 1 aprender = 50%"    "$(printf 'yoga\t2\t1\t1\t50%%\t50%%')"
  chk "one job queued"                               "$(printf 'queued\t1')"
  chk "queued job's label set listed"                "[self-hosted,X64,clean-room]"
  chk "completed job is not counted as aprender-busy" "$(printf 'intel\t3\t2\t1\t')"
  exit "$err"
fi

command -v gh >/dev/null || { echo "fleet_utilization: gh missing" >&2; exit 2; }
command -v jq >/dev/null || { echo "fleet_utilization: jq missing" >&2; exit 2; }
runners=$(gh api "orgs/$ORG/actions/runners" --paginate --jq '{runners:[.runners[]|{name,busy}]}' | jq -s '{runners:[.[].runners[]]}')
ids=$( { gh run list -R "$REPO" --status in_progress --limit 40 --json databaseId --jq '.[].databaseId'; gh run list -R "$REPO" --status queued --limit 40 --json databaseId --jq '.[].databaseId'; } | sort -u)
jobs='{"jobs":[]}'
for id in $ids; do
  part=$(gh api "repos/$REPO/actions/runs/$id/jobs?per_page=60" --jq '{jobs:[.jobs[]|{name,status,runner_name,labels}]}' 2>/dev/null || printf '{"jobs":[]}')
  jobs=$(jq -c -n --argjson a "$jobs" --argjson b "$part" '{jobs: ($a.jobs + $b.jobs)}')
done
printf 'fleet_utilization %s\n' "$(date -u +%Y-%m-%dT%H:%MZ)"
render "$runners" "$jobs"
