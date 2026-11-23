#!/usr/bin/env bash
# rerun_causes.sh -- name the step cause of every attempt-1 failure of a CI run that was re-run (#7290, #7276).
# One JSONL row per failed attempt-1 job: {run,job,step,conclusion,first_error,branch}. A job whose log
# cannot be read gets first_error "not_measured" (never a guess). LAB tool: run on demand or nightly,
# never inside a PR, merge-queue or release job.
#   bash scripts/ci/rerun_causes.sh [--limit N] [--workflow ci.yml]   # scan recent completed runs
#   bash scripts/ci/rerun_causes.sh --run <run_id>                    # one run
#   bash scripts/ci/rerun_causes.sh --self-test
set -uo pipefail
REPO=${GITHUB_REPOSITORY:-paiml/aprender}
GH_BIN=${GH:-gh}

# stdin = attempt-1 jobs JSON; stdout = "job_id<TAB>job<TAB>step<TAB>conclusion" per failed job
failed_rows() {
  jq -r '.jobs[]? | select(.conclusion=="failure" or .conclusion=="timed_out")
    | [.id, .name, ((.steps // [] | map(select(.conclusion=="failure" or .conclusion=="timed_out")) | .[0].name) // "(job-level, no failed step)"), .conclusion] | @tsv'
}

# stdin = job log text; stdout = first error line, ANSI/CR/timestamp stripped, or "not_measured" when empty
first_error() {
  local t l
  t=$(tr -d '\000\r' | sed -E 's/\x1b\[[0-9;]*[mK]//g; s/^[0-9T:.Z-]+ //')
  l=$(printf '%s\n' "$t" | grep -m1 -E '^(##\[error\]|error(\[[A-Z0-9]+\])?:|Error:|fatal:|FAILED|thread .* panicked)' | cut -c1-240)
  printf '%s\n' "${l:-not_measured}"
}

emit_run() { # $1 run id, $2 branch
  local jobs rows id job step concl err
  jobs=$("$GH_BIN" api "repos/$REPO/actions/runs/$1/attempts/1/jobs?per_page=100") || {
    echo "NOT_MEASURED: jobs read failed run $1" >&2; return 2
  }
  rows=$(printf '%s\n' "$jobs" | failed_rows) || {
    echo "NOT_MEASURED: jobs parse failed run $1" >&2; return 2
  }
  while IFS=$'\t' read -r id job step concl; do
    [ -n "$id" ] || continue
    err=$("$GH_BIN" api "repos/$REPO/actions/jobs/$id/logs" 2>/dev/null | first_error) || err=not_measured
    [ -n "$err" ] || err=not_measured
    # shellcheck disable=SC2016
    jq -nc --arg run "$1" --arg job "$job" --arg step "$step" --arg c "$concl" --arg e "$err" --arg b "$2" \
      '{run:($run|tonumber),job:$job,step:$step,conclusion:$c,first_error:$e,branch:$b}'
  done <<< "$rows"
}

self_test() {
  local j r e d stub rc out
  j='{"jobs":[{"id":7,"name":"shard 3/4","conclusion":"failure","steps":[{"name":"Build","conclusion":"success"},{"name":"cargo audit","conclusion":"failure"}]},{"id":8,"name":"ok","conclusion":"success","steps":[]},{"id":9,"name":"x86-main","conclusion":"timed_out","steps":[]}]}'
  r=$(printf '%s' "$j" | failed_rows)
  if [ "$(printf '%s\n' "$r" | wc -l)" != 2 ] \
    || ! grep -q $'^7\tshard 3/4\tcargo audit\tfailure$' <<<"$r" \
    || ! grep -q $'^9\tx86-main\t(job-level, no failed step)\ttimed_out$' <<<"$r"; then
    echo "FAIL: failed_rows"; return 1
  fi

  e=$(printf '2026-11-01T10:00:00.1234567Z note\n2026-11-01T10:00:01.1Z \x1b[31merror: failed to build addr2line index\x1b[0m\r\n2026-11-01T10:00:02Z error: later\n' | first_error)
  [ "$e" = "error: failed to build addr2line index" ] || { echo "FAIL: first_error got '$e'"; return 1; }
  [ "$(printf 'all fine\n' | first_error)" = not_measured ] || { echo "FAIL: clean log must be not_measured"; return 1; }

  d=$(mktemp -d) || { echo "FAIL: mktemp"; return 1; }
  stub="$d/gh"
  cat >"$stub" <<'EOF'
#!/usr/bin/env bash
case "$2" in
  repos/*/actions/runs/100/attempts/1/jobs*)
    printf '%s\n' '{"jobs":[{"id":7,"name":"shard 3/4","conclusion":"failure","steps":[{"name":"cargo audit","conclusion":"failure"}]},{"id":9,"name":"x86-main","conclusion":"failure","steps":[{"name":"Build","conclusion":"failure"}]}]}' ;;
  repos/*/actions/jobs/7/logs)
    printf '2026-11-01T10:00:01Z error: addr2line missing\n' ;;
  repos/*/actions/jobs/9/logs)
    exit 1 ;;
  repos/*/actions/runs/999/attempts/1/jobs*)
    exit 1 ;;
  *) exit 1 ;;
esac
EOF
  chmod +x "$stub"

  out=$(GH="$stub" bash "$0" --run 100) || { rm -rf "${d:?}"; echo "FAIL: --run 100"; return 1; }
  grep -q '"step":"cargo audit".*"first_error":"error: addr2line missing"' <<<"$out" \
    || { rm -rf "${d:?}"; echo "FAIL: measured row: $out"; return 1; }
  grep -q '"job":"x86-main","step":"Build","conclusion":"failure","first_error":"not_measured"' <<<"$out" \
    || { rm -rf "${d:?}"; echo "FAIL: unreadable log must yield not_measured: $out"; return 1; }

  rc=0; out=$(GH="$stub" bash "$0" --run 999 2>&1) || rc=$?
  if [ "$rc" -eq 0 ] || ! grep -q 'NOT_MEASURED' <<<"$out"; then
    rm -rf "${d:?}"; echo "FAIL: failed jobs read must exit non-zero with NOT_MEASURED (rc=$rc out=$out)"; return 1
  fi

  rm -rf "${d:?}"
  echo "ok: rerun_causes case table"
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --run)
    [ -n "${2:-}" ] || { echo "usage: rerun_causes.sh --run <run_id>" >&2; exit 64; }
    emit_run "$2" "?"; exit $? ;;
esac

limit=100; wf=ci.yml
while [ $# -gt 0 ]; do
  case "$1" in
    --limit) [ -n "${2:-}" ] || { echo "usage: see header" >&2; exit 64; }; limit=$2; shift 2 ;;
    --workflow) [ -n "${2:-}" ] || { echo "usage: see header" >&2; exit 64; }; wf=$2; shift 2 ;;
    *) echo "usage: see header" >&2; exit 64 ;;
  esac
done

runs=$("$GH_BIN" api "repos/$REPO/actions/workflows/$wf/runs?per_page=$limit&status=completed" \
  --jq '.workflow_runs[]|select(.run_attempt>1)|[.id,.head_branch]|@tsv') \
  || { echo "NOT_MEASURED: runs list read failed" >&2; exit 2; }
rc=0
while IFS=$'\t' read -r id br; do
  [ -n "$id" ] || continue
  emit_run "$id" "$br" || rc=$?
done <<< "$runs"
exit "$rc"
