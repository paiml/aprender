#!/usr/bin/env bash
# fleet_history.sh — the measured history behind the build-time objective (PMAT-1105, operator 2026-09-11):
# one JSON row per completed CI run of paiml/aprender — event, branch, tier-relevant wall times, and per-job
# box placement with queue wait and duration — appended idempotently to evidence/fleet/history.jsonl.
#
#   bash scripts/fleet_history.sh [--repo O/R] [--limit N] [--out FILE] [--selftest]
#
# Idempotent: a run id already present in FILE is skipped. Needs gh + jq. Box = runner name prefix
# (intel | gx10 | yoga | hosted | none). Times are seconds. Nothing here writes to GitHub.
set -euo pipefail

REPO="paiml/aprender"; LIMIT=50; OUT="evidence/fleet/history.jsonl"; SELFTEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --selftest) SELFTEST=1; shift ;;
    *) echo "fleet_history: unknown argument $1" >&2; exit 2 ;;
  esac
done

# row_from_run RUN_JSON JOBS_JSON -> one JSON line
row_from_run() {
  jq -c -n --argjson run "$1" --argjson jobs "$2" '
    def box(n): if n == null or n == "" then "none"
      elif (n|startswith("intel")) then "intel" elif (n|startswith("gx10")) then "gx10"
      elif (n|startswith("yoga")) then "yoga" elif (n|startswith("GitHub Actions")) then "hosted" else "other" end;
    def secs(a; b): if a == null or b == null then null else ((b|fromdateiso8601) - (a|fromdateiso8601)) end;
    {
      run_id: $run.databaseId, event: $run.event, workflow: $run.workflowName, branch: $run.headBranch,
      sha: ($run.headSha // "")[0:9], conclusion: $run.conclusion,
      created_at: $run.createdAt, wall_s: secs($run.createdAt; $run.updatedAt),
      jobs: [ $jobs.jobs[] | select(.conclusion != "skipped") | {
        name: .name, box: box(.runner_name), runner: (.runner_name // ""),
        conclusion: .conclusion, labels: (.labels // []),
        queue_wait_s: secs(.created_at; .started_at), duration_s: secs(.started_at; .completed_at)
      } ],
      boxes: ( [ $jobs.jobs[] | select(.conclusion != "skipped") | box(.runner_name) ] | group_by(.) | map({(.[0]): length}) | add // {} )
    }'
}

if [ "$SELFTEST" = 1 ]; then
  run_json=$(cat <<'JSON'
{"databaseId":1,"event":"pull_request","workflowName":"CI","headBranch":"b","headSha":"abcdef0123","conclusion":"success","createdAt":"2026-09-11T00:00:00Z","updatedAt":"2026-09-11T00:10:00Z"}
JSON
)
  jobs_json=$(cat <<'JSON'
{"jobs":[{"name":"workspace-test","runner_name":"yoga-build2","conclusion":"success","labels":["self-hosted","X64"],"created_at":"2026-09-11T00:00:00Z","started_at":"2026-09-11T00:02:00Z","completed_at":"2026-09-11T00:09:00Z"},{"name":"skipped-one","runner_name":null,"conclusion":"skipped"},{"name":"gate","runner_name":"intel-clean-room-3","conclusion":"success","labels":[],"created_at":"2026-09-11T00:00:00Z","started_at":"2026-09-11T00:09:00Z","completed_at":"2026-09-11T00:10:00Z"}]}
JSON
)
  row=$(row_from_run "$run_json" "$jobs_json"); err=0
  chk() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1: got $2 want $3"; err=1; fi; }
  chk "wall_s from created/updated"        "$(printf '%s' "$row" | jq .wall_s)" 600
  chk "job queue wait measured"            "$(printf '%s' "$row" | jq '.jobs[0].queue_wait_s')" 120
  chk "job duration measured"              "$(printf '%s' "$row" | jq '.jobs[0].duration_s')" 420
  chk "box from runner prefix"             "$(printf '%s' "$row" | jq -r '.jobs[0].box')" yoga
  chk "skipped jobs excluded"              "$(printf '%s' "$row" | jq '.jobs|length')" 2
  chk "boxes histogram"                    "$(printf '%s' "$row" | jq -c .boxes)" '{"intel":1,"yoga":1}'
  # idempotence: the same run id appended twice must land once
  tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT; printf '%s\n' "$row" > "$tmp"
  if grep -q '"run_id":1,' "$tmp" && [ "$(grep -c '"run_id":1,' "$tmp")" = 1 ]; then echo "ok    seed row present once"; else echo "FAIL  seed row"; err=1; fi
  rm -f "$tmp"; exit "$err"
fi

command -v gh >/dev/null || { echo "fleet_history: gh missing" >&2; exit 2; }
command -v jq >/dev/null || { echo "fleet_history: jq missing" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"; touch "$OUT"
added=0; skipped=0
while IFS= read -r run_json; do
  id=$(printf '%s' "$run_json" | jq -r .databaseId)
  if grep -q "\"run_id\":$id," "$OUT"; then skipped=$((skipped+1)); continue; fi
  jobs_json=$(gh api "repos/$REPO/actions/runs/$id/jobs?per_page=100" 2>/dev/null || printf '{"jobs":[]}')
  row_from_run "$run_json" "$jobs_json" >> "$OUT"; added=$((added+1))
done < <(gh run list -R "$REPO" --status completed --limit "$LIMIT" --json databaseId,event,workflowName,headBranch,headSha,conclusion,createdAt,updatedAt --jq '.[]' | jq -c '.')
echo "fleet_history: out=$OUT added=$added skipped=$skipped rows=$(wc -l < "$OUT")"
