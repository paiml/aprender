#!/usr/bin/env bash
# collect_build_ledger.sh — APR-RELEASE-001 §3.6 / §5 P0: one append-only ledger
# record per gate job, `docs/build-ledger/<YYYY-MM-DD>/<sha>-<host>-<job>.json`.
#
# WHY THIS IS IN THE REPOSITORY AND NOT A SESSION SCRIPT
# -----------------------------------------------------
# The spec says "the ledger IS the project memory" (§3.6.6) and refuses to set a
# build target below 20 records (§5 P0). For the whole 0.67.0 train the only thing
# writing it was a script in an agent scratchpad: 1260 records existed, 1091 of them
# committed, and nothing in the tree could produce record 1262. A memory that dies
# with the session that wrote it is not memory, and an instrument that ships no
# instrument is the same defect this spec spends §5 measuring in other people's jobs.
#
# WHAT IS MEASURED, AND WHAT IS NOT
#
#   queue_wait_s  created_at -> started_at   (how long the job waited for a runner)
#   exec_s        started_at -> completed_at (how long it ran)
#   total_s       the sum
#   host          runner_name as GitHub reports it
#   host_class    the runner name's FIRST token: intel-clean-room-3 -> intel,
#                 gx10-pool2 -> gx10, yoga-build2 -> yoga, mini-m4 -> mini.
#                 This is what the occupancy report groups on, so a new box is
#                 counted the day it is named, with no table to update.
#
#   peak_rss_mb and free_disk_gb are NOT exposed by the Actions REST API. They are
#   written as `null` and named in `unmeasured[]` with a literal "[U]", so no reader
#   can mistake an absent measurement for a zero one. Writing 0 here would be the
#   `0 violations over 0 files` signature the spec exists to prevent.
#
# COMPLETED JOBS INSIDE AN IN-PROGRESS RUN COUNT. Waiting for the whole run makes
# occupancy lag by a run: measured 2026-09-13, the packer reported 0.0% with four
# runs in flight. Records are keyed sha-host-job, so re-reading a run rewrites the
# same bytes and the ledger stays append-only in effect.
#
# usage:
#   collect_build_ledger.sh --out DIR [--limit N] [--repo OWNER/REPO]
#   collect_build_ledger.sh --self-test        # hermetic: no network, no gh
#
# exit: 0 ok · 1 a record could not be written · 2 usage, or the box cannot answer
#       (no gh, no jq) — an unmeasured ledger is never a written one.

set -uo pipefail

PROG=${0##*/}
REPO_DEFAULT=paiml/aprender

# WHICH JOBS ARE RECORDED: all of them. WHICH ARE GATE JOBS: this regex.
#
# §5 P0 asks for a record per GATE job, and an earlier draft of this file wrote only
# those. Measured over four hours of every event: the gate set is 33.0 of 44.4
# runner-hours — **74.3%**. The other 11.4 h across 285 jobs (ci / coverage,
# vendored-schemas, cuda-unit, ci / security, pr-review-shadow, mutants, gpu-touched…)
# was invisible, so every occupancy figure derived from the ledger was low by a
# quarter and no box could ever read "80% full" no matter how full it was.
#
# A denominator fix without this is half a fix: CAP counted runners that cannot take
# the work (1.5x on gx10), and the numerator skipped a quarter of the work that did.
# So every completed job gets a record and carries `gate_job`, and a reader that wants
# §5 P0's gate set filters on that field instead of losing the rest.
#
# `ci / gate` carries a space and a slash, which is why this is an anchored alternation
# over the WHOLE name and never a substring test: unanchored, `gate` also matches
# `gpu-quick … gate` and `guard-tree`.
#
# macos-arm64 is here because mini is a full-time build host (operator 2026-09-13).
LEDGER_GATE_JOBS_RE='^(workspace-test|ci / gate|gate|guard-tree|guard-cargo|ci / test|ci / lint|macos-arm64)$'

usage() { printf 'usage: %s --out DIR [--limit N] [--repo OWNER/REPO] | --self-test\n' "$PROG" >&2; exit 2; }

# host_class HOST -> the box. The first token, lowercased.
host_class() { printf '%s' "${1%%-*}" | tr 'A-Z' 'a-z'; }

# clamp_delta LATER EARLIER -> seconds, never negative.
# Runner clock skew produced started_at < created_at on a real record; a negative
# queue wait would then subtract from the occupancy total instead of adding nothing.
clamp_delta() {
    # Deterministic: both instants come from the ARGUMENTS — created_at/started_at/
    # completed_at as the Actions API reported them — never from this box's clock.
    _a=$(date -u -d "$1" +%s 2>/dev/null) || return 1  # bashrs disable-line=DET002
    _b=$(date -u -d "$2" +%s 2>/dev/null) || return 1  # bashrs disable-line=DET002
    _d=$((_a - _b)); [ "$_d" -lt 0 ] && _d=0
    printf '%s' "$_d"
}

# job_slug NAME -> a filename-safe slug ("ci / gate" -> "ci-gate").
job_slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9' '-' | sed 's/-\{1,\}/-/g; s/^-//; s/-$//'; }

# record OUT SHA HOST JOB EVENT WF RUN JID CONCL CREATED STARTED COMPLETED
# Writes one record and prints its path. The whole measurement lives here so the
# self-test can exercise it without a network.
record() {
    _out=$1; _sha=$2; _host=$3; _job=$4
    _event=$5; _wf=$6; _run=$7; _jid=$8
    _concl=$9
    _created=${10}
    _started=${11}
    _completed=${12}
    [ -n "$_started" ] && [ -n "$_completed" ] || return 1
    _qw=$(clamp_delta "$_started" "$_created") || return 1
    _ex=$(clamp_delta "$_completed" "$_started") || return 1
    _box=$(host_class "$_host")
    # 0 only on success. Computed here, not in the jq program: jq's `$concl=="success"`
    # parses to bashrs as a shell assignment with `$` on the left (SC1066) — a false
    # positive, but the shell knows the answer already, so the jq stays data-only.
    _exit=1; [ "$_concl" = "success" ] && _exit=0
    _gate=false; printf '%s' "$_job" | grep -qE "$LEDGER_GATE_JOBS_RE" && _gate=true
    _f="$_out/${_sha:0:9}-${_host}-$(job_slug "$_job").json"
    jq -n --arg sha "$_sha" --arg host "$_host" --arg box "$_box" --arg job "$_job" \
          --arg event "$_event" --arg wf "$_wf" --arg run "$_run" --arg jid "$_jid" \
          --arg concl "$_concl" --arg created "$_created" --arg started "$_started" \
          --arg completed "$_completed" \
          --argjson qw "$_qw" --argjson ex "$_ex" --argjson tot "$((_qw + _ex))" \
          --argjson rc "$_exit" --argjson gate "$_gate" \
      '{spec:"APR-RELEASE-001", section:"3.6", sha:$sha, host:$host, host_class:$box,
        job:$job, gate_job:$gate, workflow:$wf, event:$event, run_id:$run, job_id:$jid,
        queue_wait_s:$qw, exec_s:$ex, total_s:$tot,
        peak_rss_mb:null, free_disk_gb:null,
        exit:$rc, conclusion:$concl,
        created_at:$created, started_at:$started, completed_at:$completed,
        unmeasured:["peak_rss_mb [U] — not exposed by the Actions REST API",
                    "free_disk_gb [U] — not exposed by the Actions REST API"],
        source:"gh api repos/<repo>/actions/runs/<run>/jobs"}' > "$_f" || return 1
    printf '%s\n' "$_f"
}

# ingest_jobs_tsv OUT SHA EVENT WF RUN  < jobs.tsv
# One record per FILTERED job row on stdin; prints the count. The network lives in
# main(); this is the part that decides, so this is the part --self-test drives.
# Splitting them is the point: the first version of this file had a case table that
# proved `record` and a main() loop nothing executed, and it collected 0 records on
# its first live run while the self-test read 26/26 PASS.
ingest_jobs_tsv() {
    _i_out=$1; _i_sha=$2; _i_event=$3; _i_wf=$4; _i_run=$5
    _i_n=0
    while IFS=$'\t' read -r _job _host _created _started _completed _concl _jid; do
        [ -n "$_job" ] || continue
        _p=$(record "$_i_out" "$_i_sha" "$_host" "$_job" "$_i_event" "$_i_wf" "$_i_run" "$_jid" \
                    "$_concl" "$_created" "$_started" "$_completed") || continue
        printf '%s\n' "$_p" >&2
        _i_n=$((_i_n + 1))
    done
    printf '%s' "$_i_n"
}

# ── the case table ───────────────────────────────────────────────────────────
# Hermetic: every row exercises `record`/`host_class`/`clamp_delta` directly, so
# --self-test needs neither gh nor a network. Each row is here because the thing it
# asserts was, or could silently become, wrong.
self_test() {
    command -v jq > /dev/null 2>&1 || { echo "$PROG: self-test needs jq" >&2; return 2; }
    _td=$(mktemp -d) || return 2
    trap 'rm -rf -- "${_td:?}"' EXIT
    _fail=0
    _row() { # _row NAME EXPECTED ACTUAL
        if [ "$2" = "$3" ]; then printf '  OK   %s\n' "$1"
        else printf '  FAIL %s: expected %s, got %s\n' "$1" "$2" "$3"; _fail=$((_fail + 1)); fi
    }

    # host_class: every box in the fleet, and the API's own fallback.
    _row "host_class intel"   "intel"   "$(host_class intel-clean-room-3)"
    _row "host_class gx10"    "gx10"    "$(host_class gx10-pool2)"
    _row "host_class yoga"    "yoga"    "$(host_class yoga-build2)"
    _row "host_class mini"    "mini"    "$(host_class mini-m4)"
    _row "host_class unknown" "unknown" "$(host_class unknown)"

    # clamp_delta: a normal span, and the clock-skew case that must floor at 0
    # rather than subtract from the occupancy total.
    _row "clamp 90s" "90" "$(clamp_delta 2026-09-13T10:01:30Z 2026-09-13T10:00:00Z)"
    _row "clamp skew -> 0" "0" "$(clamp_delta 2026-09-13T10:00:00Z 2026-09-13T10:01:30Z)"

    # job_slug: the name with a space and a slash is the one that breaks a filename.
    _row "slug 'ci / gate'" "ci-gate" "$(job_slug 'ci / gate')"

    # The gate-job CLASSIFIER. Both polarities: an anchored alternation, so a name that
    # merely CONTAINS a gate word is not misread as a gate job. It decides a FIELD now,
    # never whether a record exists — that distinction is the point of this file.
    _match() { printf '%s' "$1" | grep -qE "$LEDGER_GATE_JOBS_RE" && echo yes || echo no; }
    _row "gate_job workspace-test"    "yes" "$(_match 'workspace-test')"
    _row "gate_job 'ci / gate'"       "yes" "$(_match 'ci / gate')"
    _row "gate_job macos-arm64"       "yes" "$(_match 'macos-arm64')"
    _row "not a gate job: ci / bench" "no"  "$(_match 'ci / bench')"
    _row "not a gate job: gpu-quick gate" "no" "$(_match 'gpu-quick gate')"
    _row "not a gate job: present"    "no"  "$(_match 'present')"

    # A written record: the fields the packer's occupancy actually reads.
    _f=$(record "$_td" abcdef1234567890 mini-m4 macos-arm64 pull_request CI 42 99 \
                success 2026-09-13T10:00:00Z 2026-09-13T10:00:20Z 2026-09-13T10:10:20Z) || {
        echo "  FAIL record returned non-zero"; _fail=$((_fail + 1)); }
    _row "record filename" "abcdef123-mini-m4-macos-arm64.json" "$(basename "${_f:-none}")"
    _row "record host_class"   "mini" "$(jq -r .host_class    "$_f" 2>/dev/null)"
    _row "record queue_wait_s" "20"   "$(jq -r .queue_wait_s  "$_f" 2>/dev/null)"
    _row "record exec_s"       "600"  "$(jq -r .exec_s        "$_f" 2>/dev/null)"
    _row "record total_s"      "620"  "$(jq -r .total_s       "$_f" 2>/dev/null)"
    _row "record exit"         "0"    "$(jq -r .exit          "$_f" 2>/dev/null)"
    _row "record gate_job"     "true" "$(jq -r .gate_job      "$_f" 2>/dev/null)"

    # The unmeasured fields are null AND named. A reader that sees 0 here would
    # compute a peak-RSS average over machines that never reported one.
    _row "peak_rss_mb null"  "null" "$(jq -r '.peak_rss_mb'   "$_f" 2>/dev/null)"
    _row "free_disk_gb null" "null" "$(jq -r '.free_disk_gb'  "$_f" 2>/dev/null)"
    _row "unmeasured names both" "2" "$(jq -r '[.unmeasured[]|select(test("\\[U\\]"))]|length' "$_f" 2>/dev/null)"

    # A failed job records exit 1 — the ledger must carry reds, or a p99 derived
    # from it describes only the runs that happened to pass.
    _g=$(record "$_td" abcdef1234567890 intel-clean-room-8 workspace-test merge_group CI 43 100 \
                failure 2026-09-13T10:00:00Z 2026-09-13T10:00:00Z 2026-09-13T10:45:00Z)
    _row "failed job exit 1" "1" "$(jq -r .exit "$_g" 2>/dev/null)"

    # Idempotence: the same job re-read rewrites ONE file, never a second.
    record "$_td" abcdef1234567890 mini-m4 macos-arm64 pull_request CI 42 99 \
           success 2026-09-13T10:00:00Z 2026-09-13T10:00:20Z 2026-09-13T10:10:20Z > /dev/null
    _row "re-read is idempotent" "2" "$(find "$_td" -name '*.json' | wc -l | tr -d ' ')"

    # A job with no completed_at is skipped, not written as a zero-length run.
    if record "$_td" abcdef1234567890 mini-m4 macos-arm64 pull_request CI 42 99 \
              "" 2026-09-13T10:00:00Z 2026-09-13T10:00:20Z "" > /dev/null 2>&1
    then _row "incomplete job refused" "refused" "written"
    else _row "incomplete job refused" "refused" "refused"; fi

    # THE PLUMBING, not just the parts. A real jobs TSV as the API emits it — tab
    # separated, one row per completed job, gate jobs mixed with jobs that are not
    # instrumented — driven through the same function main() calls.
    _td2=$(mktemp -d -p "$_td") || return 2
    _tsv=$(printf '%b\n' \
"workspace-test\tintel-clean-room-8\t2026-09-13T09:51:00Z\t2026-09-13T09:51:38Z\t2026-09-13T11:07:38Z\tsuccess\t101" \
"ci / bench\tintel-clean-room-8\t2026-09-13T09:51:00Z\t2026-09-13T09:51:00Z\t2026-09-13T09:52:00Z\tskipped\t102" \
"macos-arm64\tmini-m4\t2026-09-13T09:50:00Z\t2026-09-13T09:50:05Z\t2026-09-13T10:02:05Z\tsuccess\t103" \
"present\tintel-clean-room-2\t2026-09-13T09:50:00Z\t2026-09-13T09:50:02Z\t2026-09-13T09:51:08Z\tfailure\t104")
    _got=$(printf '%s\n' "$_tsv" | ingest_jobs_tsv "$_td2" deadbeefcafe0000 pull_request CI 7 2>/dev/null)
    # EVERY completed job is recorded — 4 rows in, 4 records out. The earlier draft
    # wrote 2 and lost a quarter of the fleet's runner-hours to the occupancy figure.
    _row "ingest records every job" "4" "$_got"
    _row "ingest wrote 4 files" "4" "$(find "$_td2" -name '*.json' | wc -l | tr -d ' ')"
    _row "ingest records mini"  "1" "$(grep -l '\"host_class\": \"mini\"' "$_td2"/*.json 2>/dev/null | wc -l | tr -d ' ')"
    # `present` is NOT a gate job and IS recorded: that is the whole correction.
    _row "present recorded" "1" "$(find "$_td2" -name '*present*' | wc -l | tr -d ' ')"
    _row "present not a gate job" "false" "$(jq -r .gate_job "$_td2"/deadbeefc-intel-clean-room-2-present.json 2>/dev/null)"
    _row "workspace-test is a gate job" "true" "$(jq -r .gate_job "$_td2"/deadbeefc-intel-clean-room-8-workspace-test.json 2>/dev/null)"
    _row "gate jobs are 2 of 4" "2" "$(grep -l '\"gate_job\": true' "$_td2"/*.json 2>/dev/null | wc -l | tr -d ' ')"
    _row "ingest queue wait measured" "38" "$(jq -r .queue_wait_s "$_td2"/deadbeefc-intel-clean-room-8-workspace-test.json 2>/dev/null)"
    # An empty TSV (a run whose jobs are all still going) is 0 records, not an error.
    _row "empty tsv -> 0" "0" "$(printf '' | ingest_jobs_tsv "$_td2" deadbeefcafe0000 pull_request CI 7 2>/dev/null)"

    if [ "$_fail" -eq 0 ]; then echo "SELF-TEST PASS"; return 0; fi
    echo "SELF-TEST FAIL: $_fail row(s)"; return 1
}

main() {
    [ $# -gt 0 ] || usage
    _out="" _limit=30 _repo=$REPO_DEFAULT
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return $? ;;
            --out)   _out=${2:-}; shift 2 || usage ;;
            --limit) _limit=${2:-}; shift 2 || usage ;;
            --repo)  _repo=${2:-}; shift 2 || usage ;;
            -h|--help) usage ;;
            *) printf '%s: unknown argument %s\n' "$PROG" "$1" >&2; usage ;;
        esac
    done
    [ -n "$_out" ] || usage
    case "$_out" in /*) ;; *) echo "$PROG: refused, --out must be absolute" >&2; return 2 ;; esac
    case "$_out" in *..*) echo "$PROG: refused, --out must not contain '..'" >&2; return 2 ;; esac
    for _t in gh jq; do
        command -v "$_t" > /dev/null 2>&1 || { echo "$PROG: $_t is required" >&2; return 2; }
    done
    mkdir -p "$_out" || return 2

    _n=0
    while IFS=$'\t' read -r _run _sha _event _wf; do
        [ -n "$_run" ] || continue
        _n=$((_n + $(ingest_jobs_tsv "$_out" "$_sha" "$_event" "$_wf" "$_run" <<EOF
$(gh api "repos/$_repo/actions/runs/$_run/jobs?per_page=100" --jq \
    '.jobs[] | select(.status=="completed") | [.name, (.runner_name // "unknown"), .created_at, .started_at, .completed_at, (.conclusion // "unknown"), (.id|tostring)] | @tsv' 2>/dev/null)
EOF
)))
    done <<EOF
$(gh run list --repo "$_repo" --limit "$_limit" \
    --json databaseId,status,event,headSha,workflowName \
    --jq '.[] | "\(.databaseId)\t\(.headSha)\t\(.event)\t\(.workflowName)"' 2>/dev/null)
EOF
    printf '%s: %d record(s) under %s\n' "$PROG" "$_n" "$_out" >&2
}

main "$@"
