#!/usr/bin/env bash
# check_mutants_pool.sh -- the full mutation job runs on the yoga X64 clean-room pool, never the intel clean-room
# pool (C100) and never a gpu runner (#5228: a PR + perf host).
#
# A whole-workspace mutation run saturates the clean-room (intel) runners that every PR's CI needs. The X64
# `mutants` pool of infra#1468 was never built (infra#1468 closed; no runner carries the label), so the job sits
# where the other LAB nightlies sit: [self-hosted, Linux, X64, clean-room, yoga]. Its runs-on must name both
# `clean-room` and `yoga` as whole labels, and must not name intel, perf-solo, gpu or arm64. A missing job or a
# missing runs-on is ENV rc=2, never a pass.
#
#   check_mutants_pool.sh [workflow]   the case table, then the real workflow (default mutants-nightly.yml)
#   check_mutants_pool.sh --self-test  planted workflows: intel, the gpu pools and the dead `mutants` label go RED, yoga GREEN
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
JOB="${MUTANTS_POOL_JOB:-mutants}"

job_runs_on() { # job_runs_on <workflow> -> the runs-on text of the job
    awk -v job="$JOB" '$0 == "  " job ":"{f=1; next} f && /^  [A-Za-z0-9_-]+:/{f=0} f && /^    runs-on:/{sub(/^    runs-on:[ ]*/, ""); sub(/[ ]+#.*$/, ""); print; exit}' "$1"
}
has() { # has <runs-on text> <label> -> 0 iff the label is one of the request's labels (GitHub matches labels case-insensitively)
    local t l
    t=" $(printf '%s' "$1" | tr '[],"' '    ' | tr '[:upper:]' '[:lower:]') "; l=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
    case "$t" in *" $l "*) return 0 ;; *) return 1 ;; esac
}
# The one accepted pool: the yoga X64 clean-room runners, named as both whole labels `clean-room` and `yoga`. Never a
# request that can land on intel: intel or perf-solo (its runner carries intel), and never a bare `clean-room`
# (intel carries it too). Never a gpu runner (#5228: gpu or ARM64, a PR + perf host), and never the `mutants` label
# of infra#1468, which no runner carries (infra#1468 closed): a request naming it matches nothing and never runs.
pool_ok() { # 0 iff it names the yoga clean-room pool and nothing that reaches intel, a gpu box or a dead label
    # Deny by case-insensitive substring (intel-x64 and perf-solo-2 are refused); accept only by whole label.
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in *intel*|*perf-solo*) return 1 ;; esac
    has "$1" gpu && return 1; has "$1" arm64 && return 1; has "$1" cuda && return 1; has "$1" mutants && return 1
    has "$1" clean-room && has "$1" yoga && return 0
    return 1
}
table() { # table <workflow>
    local want text got bad=0
    while IFS='|' read -r want text; do
        got=0; pool_ok "$text" || got=1
        if [ "$got" = "$want" ]; then printf 'ok    pool %s -> %s\n' "${text:-<empty>}" "$want"
        else printf 'FAIL  pool %s wanted %s, got %s\n' "${text:-<empty>}" "$want" "$got" >&2; bad=1; fi
    done <<'POOLS'
0|[self-hosted, Linux, X64, clean-room, yoga]
0|[self-hosted, Linux, X64, CLEAN-ROOM, Yoga]
1|[self-hosted, Linux, X64, mutants]
1|[self-hosted, Linux, X64, clean-room, yoga, mutants]
1|[self-hosted, Linux, clean-room]
1|[self-hosted, Linux, X64, clean-room]
1|[self-hosted, Linux, X64]
1|[self-hosted, Linux, X64, yoga]
1|[self-hosted, Linux, X64, clean-room, intel]
1|[self-hosted, Linux, X64, clean-room, yoga, intel]
1|[self-hosted, Linux, X64, clean-room, yoga, Intel]
1|[self-hosted, Linux, X64, clean-room, yoga, intel-x64]
1|[self-hosted, Linux, X64, clean-room, yoga, perf-solo]
1|[self-hosted, Linux, X64, clean-room, yoga, Perf-Solo-2]
1|[self-hosted, Linux, X64, gpu, yoga]
1|[self-hosted, Linux, X64, gpu, cuda, yoga, ada]
1|[self-hosted, Linux, X64, cuda, yoga, ephemeral, docker]
1|[self-hosted, Linux, X64, clean-room, yoga, GPU]
1|[self-hosted, Linux, ARM64, clean-room, gx10]
1|[self-hosted, Linux, ARM64, gpu, docker]
1|[self-hosted, Linux, X64, clean-room-2, yoga]
1|[self-hosted, Linux, X64, clean-room, yoga-2]
1|[self-hosted, Linux, X64, not-mutants]
1|[self-hosted, Linux, X64, mutants-intel]
1|
POOLS
    text=$(job_runs_on "$1")
    [ -n "$text" ] || { printf 'ENV   no %s runs-on in %s -- cannot judge, not a pass\n' "$JOB" "$1" >&2; return 2; }
    if pool_ok "$text"; then printf 'ok    %s runs-on %s\n' "$JOB" "$text"
    else printf 'FAIL  %s runs-on %s is not the yoga clean-room pool\n' "$JOB" "$text" >&2; bad=1; fi
    return "$bad"
}

if [ "${1:-}" = "--self-test" ]; then
    d=$(mktemp -d "${TMPDIR:-/tmp}/mutants-pool.XXXXXX") || exit 2
    trap 'rm -f "${d:?}"/*.yml; rmdir "${d:?}"' EXIT
    bad=0
    mk() { printf 'jobs:\n  mutants:\n    runs-on: %s\n' "$2" > "$d/$1.yml"; }
    mk intel '[self-hosted, Linux, clean-room]'; mk ok '[self-hosted, Linux, X64, clean-room, yoga] # C100'
    if table "$d/intel.yml" > /dev/null 2>&1; then echo "FAIL  a job on the intel pool passed"; bad=1; else echo "ok    intel pool is RED"; fi
    if table "$d/ok.yml" > /dev/null 2>&1; then echo "ok    yoga clean-room pool is GREEN"; else echo "FAIL  yoga clean-room pool refused"; bad=1; fi
    mk gx10 "[self-hosted, Linux, ARM64, gpu, docker] # #5228"
    if table "$d/gx10.yml" > /dev/null 2>&1; then echo "FAIL  a job on the ARM64 gpu pool passed"; bad=1; else echo "ok    ARM64 gpu pool is RED (#5228)"; fi
    mk dead "[self-hosted, Linux, X64, mutants] # infra#1468 closed"
    if table "$d/dead.yml" > /dev/null 2>&1; then echo "FAIL  the dead mutants label passed"; bad=1; else echo "ok    dead mutants label is RED (no runner carries it)"; fi
    printf 'jobs:\n  mutants:\n    runs-on:\n' > "$d/empty.yml"; rc=0; table "$d/empty.yml" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 2 ] && echo "ok    empty runs-on is ENV rc=2" || { echo "FAIL  empty runs-on gave rc=$rc"; bad=1; }
    printf 'jobs: {}\n' > "$d/none.yml"; rc=0; table "$d/none.yml" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 2 ] && echo "ok    no job is ENV rc=2" || { echo "FAIL  no job gave rc=$rc"; bad=1; }
    # The same workflow's C324 stop decision (#5228): its own case table, run here so the existing CI step covers it.
    if bash "$ROOT/scripts/mutants_weekly_stop.sh" --self-test > /dev/null 2>&1; then echo "ok    weekly stop decision table passes"
    else echo "FAIL  scripts/mutants_weekly_stop.sh --self-test"; bad=1; fi
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi
table "${1:-$ROOT/.github/workflows/mutants-nightly.yml}"
