#!/usr/bin/env bash
# check_mutants_pool.sh -- the full mutation job runs on the `mutants` pool, never the intel pool (C100).
#
# A whole-workspace mutation run saturates the clean-room (intel) runners that every PR's CI needs.
# The job's runs-on must name `mutants` and must not name clean-room or intel. A missing job or a
# missing runs-on is ENV rc=2, never a pass.
#
#   check_mutants_pool.sh [workflow]   the case table, then the real workflow (default mutants-nightly.yml)
#   check_mutants_pool.sh --self-test  planted workflows: intel must go RED, the mutants pool GREEN
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
JOB="${MUTANTS_POOL_JOB:-mutants}"

job_runs_on() { # job_runs_on <workflow> -> the runs-on text of the job
    awk -v job="$JOB" '$0 == "  " job ":"{f=1; next} f && /^  [A-Za-z0-9_-]+:/{f=0} f && /^    runs-on:/{sub(/^    runs-on:[ ]*/, ""); sub(/[ ]+#.*$/, ""); print; exit}' "$1"
}
pool_ok() { # 0 iff it names `mutants` and neither clean-room nor intel
    case "$1" in *clean-room*|*intel*) return 1 ;; esac
    case "$1" in *mutants*) return 0 ;; *) return 1 ;; esac
}
table() { # table <workflow>
    local want text got bad=0
    while IFS='|' read -r want text; do
        got=0; pool_ok "$text" || got=1
        if [ "$got" = "$want" ]; then printf 'ok    pool %s -> %s\n' "${text:-<empty>}" "$want"
        else printf 'FAIL  pool %s wanted %s, got %s\n' "${text:-<empty>}" "$want" "$got" >&2; bad=1; fi
    done <<'POOLS'
0|[self-hosted, Linux, X64, mutants]
1|[self-hosted, Linux, clean-room]
1|[self-hosted, Linux, X64]
1|[self-hosted, Linux, X64, mutants, clean-room]
1|[self-hosted, Linux, intel, mutants]
1|
POOLS
    text=$(job_runs_on "$1")
    [ -n "$text" ] || { printf 'ENV   no %s runs-on in %s -- cannot judge, not a pass\n' "$JOB" "$1" >&2; return 2; }
    if pool_ok "$text"; then printf 'ok    %s runs-on %s\n' "$JOB" "$text"
    else printf 'FAIL  %s runs-on %s is not the mutants pool\n' "$JOB" "$text" >&2; bad=1; fi
    return "$bad"
}

if [ "${1:-}" = "--self-test" ]; then
    d=$(mktemp -d "${TMPDIR:-/tmp}/mutants-pool.XXXXXX") || exit 2
    trap 'rm -f "${d:?}"/*.yml; rmdir "${d:?}"' EXIT
    bad=0
    mk() { printf 'jobs:\n  mutants:\n    runs-on: %s\n' "$2" > "$d/$1.yml"; }
    mk intel '[self-hosted, Linux, clean-room]'; mk ok '[self-hosted, Linux, X64, mutants] # C100'
    if table "$d/intel.yml" > /dev/null 2>&1; then echo "FAIL  a job on the intel pool passed"; bad=1; else echo "ok    intel pool is RED"; fi
    if table "$d/ok.yml" > /dev/null 2>&1; then echo "ok    mutants pool is GREEN"; else echo "FAIL  mutants pool refused"; bad=1; fi
    printf 'jobs: {}\n' > "$d/none.yml"; rc=0; table "$d/none.yml" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 2 ] && echo "ok    no job is ENV rc=2" || { echo "FAIL  no job gave rc=$rc"; bad=1; }
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi
table "${1:-$ROOT/.github/workflows/mutants-nightly.yml}"
