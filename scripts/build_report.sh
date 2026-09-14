#!/usr/bin/env bash
# build_report.sh — `make build-report` (APR-RELEASE-001 §5 P0·Instrument).
#
# Reads docs/build-ledger/<date>/<sha>-<host>-<job>.json records and prints:
#   - per-host_class table: n, p50/p95 total_s, p50/p95 queue_wait_s
#   - the 10 slowest `job` values by p95 total_s (per JOB, not per test target —
#     the ledger has no finer grain than that; the report says so)
#   - the §1 number: max PRs/train ~= 3 x 72h / p95(ci / gate total_s)
#   - a final `gate: ` line in the §7 report shape
#
# A "job record" is any ledger JSON object carrying a non-null total_s. Records
# without one (T-5 reconcile, fleet-pack, train ledger rows) are valid JSON,
# just not a job record: they are counted as "skipped", never as a reject.
#
# Exit codes (the contract scripts/check_build_report.sh asserts):
#   0  reported
#   1  a file under the ledger dir is not parseable JSON (stderr: `reject: ...`)
#   2  usage error, OR fewer than 20 job records under the ledger (§8 vacuity
#      floor — stderr: `decline: ...`)
#
# Percentile: nearest-rank over a 0-indexed ascending array a of length n:
#   p(k) = a[ceil(k/100 * n) - 1], clamped to [0, n-1].
# Pinned by `--self-test`'s case table — see run_self_test() below.
set -euo pipefail

PROG=${0##*/}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SELF="$ROOT/scripts/build_report.sh"
LEDGER_DEFAULT="$ROOT/docs/build-ledger"
JOB_FLOOR=20

usage() {
    printf 'Usage: %s [--ledger DIR] [--format text|json] [--self-test]\n' "$PROG"
}

# jq filter: percentile(k) — nearest-rank over the input array. See header.
# NOTE on style: this jq text is embedded as a bash single-quoted string, and
# bashrs' scanner reads $identifier[ as bash array syntax even inside quotes
# (documented false-positive class — see agent-memory
# feedback_bashrs_string_literal_scan_false_positives.md). Every array access
# below is written as `($name | .[idx])` / `($name | map(...))` instead of
# `$name[...]` / `$name[]` so the filter never contains that character
# sequence, and `bashrs lint` reports 0 errors on a program a human would
# otherwise write with normal jq idiom.
JQ_PERCENTILE='
def percentile(k):
  (sort) as $sorted
  | ($sorted | length) as $n
  | if $n == 0 then null
    else
      (((k / 100.0) * $n) | ceil) as $raw
      | (if $raw < 1 then 1 elif $raw > $n then $n else $raw end) as $idx
      | ($sorted | .[$idx - 1])
    end;
'

# jq filter: takes a slurped array of ALL valid ledger records (job records and
# non-job records alike) on stdin and emits one JSON "model" object.
JQ_MODEL="$JQ_PERCENTILE"'
. as $all
| ($all | length) as $total_valid
| ($all | map(select(has("total_s") and (.total_s != null)))) as $jobs
| ($jobs | length) as $job_count
| ($total_valid - $job_count) as $skipped_count
| ($jobs | map(.host_class) | unique | sort) as $host_classes
| (
    $host_classes
    | map(
        . as $hc
        | ($jobs | map(select(.host_class == $hc))) as $hc_jobs
        | {
            host_class: $hc,
            n: ($hc_jobs | length),
            p50_total_s: ($hc_jobs | map(.total_s) | percentile(50)),
            p95_total_s: ($hc_jobs | map(.total_s) | percentile(95)),
            p50_queue_wait_s: ($hc_jobs | map(select(has("queue_wait_s") and (.queue_wait_s != null)) | .queue_wait_s) | percentile(50)),
            p95_queue_wait_s: ($hc_jobs | map(select(has("queue_wait_s") and (.queue_wait_s != null)) | .queue_wait_s) | percentile(95))
          }
      )
  ) as $host_table
| ($jobs | map(.job) | unique | sort) as $job_names
| (
    $job_names
    | map(
        . as $jn
        | ($jobs | map(select(.job == $jn))) as $jn_jobs
        | {
            job: $jn,
            n: ($jn_jobs | length),
            p95_total_s: ($jn_jobs | map(.total_s) | percentile(95))
          }
      )
    | sort_by(-.p95_total_s, .job)
    | .[0:10]
  ) as $slowest_jobs
| ($jobs | map(select(.job == "ci / gate") | .total_s)) as $ci_gate_totals
| (if ($ci_gate_totals | length) > 0 then ($ci_gate_totals | percentile(95)) else null end) as $ci_gate_p95_s
| (if $ci_gate_p95_s == null then null else (((3 * 72 * 3600) / $ci_gate_p95_s) | floor) end) as $prs_per_train
| (
    ["intel", "yoga", "gx10"]
    | map(
        . as $hc
        | ($host_table | map(select(.host_class == $hc) | .p95_queue_wait_s)) as $m
        | (($m | .[0]) // null)
      )
  ) as $queue_p95_required
| ($queue_p95_required | .[0]) as $q_intel
| ($queue_p95_required | .[1]) as $q_yoga
| ($queue_p95_required | .[2]) as $q_gx10
| {
    total_valid_records: $total_valid,
    job_count: $job_count,
    skipped_count: $skipped_count,
    host_table: $host_table,
    slowest_jobs: $slowest_jobs,
    ci_gate_p95_s: $ci_gate_p95_s,
    prs_per_train: $prs_per_train,
    queue_p95_intel: $q_intel,
    queue_p95_yoga: $q_yoga,
    queue_p95_gx10: $q_gx10
  }
'

# disp <value> — "null" (jq -r on a JSON null) prints as "[U]", else as-is.
disp() {
    if [ "$1" = "null" ]; then
        printf '[U]'
    else
        printf '%s' "$1"
    fi
}

# build_ndjson <ledger_dir> <out_ndjson> <out_err> — writes one compact JSON
# object per line of every parseable file under ledger_dir to out_ndjson, and
# returns the count of files that failed to parse. Fast path: one batched jq
# invocation across every file (measured ~0.1s over 1092 files). Only on a
# batch failure does it fall back to classifying file-by-file (still correct,
# just O(files) jq spawns — bounded by the rare case of an actual parse error).
build_ndjson() {
    local ledger=$1 out=$2 err=$3
    local -a files=()
    if [ -d "$ledger" ]; then
        while IFS= read -r -d '' f; do
            files+=("$f")
        done < <(find "$ledger" -type f -name '*.json' -print0 | sort -z)
    fi

    : >"$out"
    local nfiles=${#files[@]}
    if [ "$nfiles" -eq 0 ]; then
        printf '0'
        return 0
    fi

    if printf '%s\0' "${files[@]}" | xargs -0 jq -c '.' >"$out" 2>"$err"; then
        printf '0'
        return 0
    fi

    : >"$out"
    local reject_count=0
    local f
    for f in "${files[@]}"; do
        if ! jq -c '.' "$f" >>"$out" 2>>"$err"; then
            reject_count=$((reject_count + 1))
        fi
    done
    printf '%s' "$reject_count"
    return 0
}

# report <ledger_dir> <format> <tmp_dir> — the whole read-classify-render
# pipeline. Returns 0/1/2 per the exit-code contract in the file header.
# tmp_dir is caller-owned (main() creates and cleans it via an EXIT trap) so a
# `set -e` termination mid-report still gets the directory removed.
report() {
    local ledger=$1 format=$2 tmp=$3

    local ndjson="$tmp/records.ndjson" err="$tmp/jq.err"
    local reject_count
    reject_count=$(build_ndjson "$ledger" "$ndjson" "$err")

    if [ "$reject_count" -gt 0 ]; then
        printf 'reject: %s parse error(s) under %s\n' "$reject_count" "$ledger" >&2
        return 1
    fi

    local model_json
    model_json=$(jq -s "$JQ_MODEL" "$ndjson")

    local job_count
    job_count=$(printf '%s' "$model_json" | jq -r '.job_count')

    if [ "$job_count" -lt "$JOB_FLOOR" ]; then
        printf 'decline: %s job record(s) under %s, floor %s\n' \
            "$job_count" "$ledger" "$JOB_FLOOR" >&2
        return 2
    fi

    if [ "$format" = "json" ]; then
        printf '%s\n' "$model_json"
        return 0
    fi

    render_text "$model_json" "$ledger"
    return 0
}

# render_text <model_json> <ledger_dir>
render_text() {
    local model_json=$1 ledger=$2

    local total_valid skipped_count ci_gate_p95_s prs_per_train
    local q_intel q_yoga q_gx10
    total_valid=$(printf '%s' "$model_json" | jq -r '.total_valid_records')
    skipped_count=$(printf '%s' "$model_json" | jq -r '.skipped_count')
    ci_gate_p95_s=$(printf '%s' "$model_json" | jq -r '.ci_gate_p95_s')
    prs_per_train=$(printf '%s' "$model_json" | jq -r '.prs_per_train')
    q_intel=$(printf '%s' "$model_json" | jq -r '.queue_p95_intel')
    q_yoga=$(printf '%s' "$model_json" | jq -r '.queue_p95_yoga')
    q_gx10=$(printf '%s' "$model_json" | jq -r '.queue_p95_gx10')

    printf 'build_report: ledger=%s valid=%s job_records=%s skipped=%s (not job records)\n' \
        "$ledger" "$total_valid" \
        "$(printf '%s' "$model_json" | jq -r '.job_count')" "$skipped_count"
    printf '\n'
    printf 'per host_class — total_s and queue_wait_s, seconds:\n'
    printf '  %-10s %6s %10s %10s %12s %12s\n' \
        "host_class" "n" "p50_total_s" "p95_total_s" "p50_queue_s" "p95_queue_s"
    printf '%s\n' "$model_json" | jq -r \
        '.host_table[] | [.host_class, (.n|tostring), (.p50_total_s|tostring), (.p95_total_s|tostring), (.p50_queue_wait_s|tostring), (.p95_queue_wait_s|tostring)] | @tsv' |
        while IFS=$'\t' read -r hc n p50t p95t p50q p95q; do
            printf '  %-10s %6s %10s %10s %12s %12s\n' \
                "$hc" "$n" "$(disp "$p50t")" "$(disp "$p95t")" "$(disp "$p50q")" "$(disp "$p95q")"
        done

    printf '\n'
    printf '10 slowest jobs by p95 total_s — the ledger is per JOB, not per test\n'
    printf 'target; this is the finest grain it can answer for §5, "10 slowest test targets":\n'
    printf '  %-20s %6s %12s\n' "job" "n" "p95_total_s"
    printf '%s\n' "$model_json" | jq -r \
        '.slowest_jobs[] | [.job, (.n|tostring), (.p95_total_s|tostring)] | @tsv' |
        while IFS=$'\t' read -r job n p95t; do
            printf '  %-20s %6s %12s\n' "$job" "$n" "$(disp "$p95t")"
        done

    printf '\n'
    if [ "$ci_gate_p95_s" = "null" ]; then
        printf '%s1 max PRs/train: [U] (no ci / gate records)\n' "$(printf '\xc2\xa7')"
    else
        printf '%s1 max PRs/train ~= 3 x 72h / p95(ci / gate total_s) = 3 x 259200 / %s = %s\n' \
            "$(printf '\xc2\xa7')" "$ci_gate_p95_s" "$prs_per_train"
    fi

    local p95_min
    if [ "$ci_gate_p95_s" = "null" ]; then
        p95_min="[U]"
    else
        p95_min=$(awk -v s="$ci_gate_p95_s" 'BEGIN { printf "%.1f", s / 60 }')
    fi

    printf '\n'
    printf 'gate:    p95 ci/gate %s | max PRs/train %s | queue p95 intel %s yoga %s gx10 %s\n' \
        "$p95_min" "$(disp "$prs_per_train")" "$(disp "$q_intel")" "$(disp "$q_yoga")" "$(disp "$q_gx10")"
}

# ---------------------------------------------------------------------------
# --self-test
# ---------------------------------------------------------------------------

# pct_case <label> <array_json> <k> <want> — one row of the percentile table.
pct_case() {
    local label=$1 arr=$2 k=$3 want=$4 got
    got=$(printf '%s' "$arr" | jq "$JQ_PERCENTILE"' . | percentile('"$k"')')
    if [ "$got" = "$want" ]; then
        printf '  ok   %-40s want=%-6s got=%s\n' "$label" "$want" "$got"
        return 0
    fi
    printf '  FAIL %-40s want=%-6s got=%s\n' "$label" "$want" "$got"
    return 1
}

# fixture_case <label> <nrecords> <inject_malformed> <want_rc>
fixture_case() {
    local label=$1 n=$2 inject_malformed=$3 want_rc=$4
    local dir got_rc
    dir=$(mktemp -d)
    local i=0
    while [ "$i" -lt "$n" ]; do
        printf '{"spec":"APR-RELEASE-001","sha":"selftest%03d","host":"intel-w1","host_class":"intel","job":"ci / gate","queue_wait_s":%d,"exec_s":%d,"total_s":%d,"exit":0}\n' \
            "$i" "$i" "$((i * 2))" "$((i * 3 + 1))" >"$dir/rec-$i.json"
        i=$((i + 1))
    done
    if [ "$inject_malformed" = "yes" ]; then
        printf 'not json at all\n' >"$dir/broken.json"
    fi

    set +e
    "$SELF" --ledger "$dir" >/dev/null 2>/dev/null
    got_rc=$?
    set -e
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        rm -rf "$dir"
    fi

    if [ "$got_rc" = "$want_rc" ]; then
        printf '  ok   %-40s want=%-6s got=%s\n' "$label" "$want_rc" "$got_rc"
        return 0
    fi
    printf '  FAIL %-40s want=%-6s got=%s\n' "$label" "$want_rc" "$got_rc"
    return 1
}

run_self_test() {
    local fail=0
    printf '== %s --self-test ==\n' "$PROG"

    printf -- '-- percentile(k) = a[ceil(k/100*n)-1], nearest-rank --\n'
    local vec100
    vec100=$(jq -n '[range(1;101)]')
    pct_case "percentile(50) over 1..100" "$vec100" 50 50 || fail=1
    pct_case "percentile(95) over 1..100" "$vec100" 95 95 || fail=1
    pct_case "percentile(100) over 1..100" "$vec100" 100 100 || fail=1
    pct_case "percentile(50) over [7] (n=1)" "[7]" 50 7 || fail=1
    pct_case "percentile(95) over [7] (n=1)" "[7]" 95 7 || fail=1

    printf -- '-- exit-code contract (fixtures under mktemp -d) --\n'
    fixture_case "0 records -> decline (exit 2)" 0 no 2 || fail=1
    fixture_case "19 records -> decline (exit 2)" 19 no 2 || fail=1
    fixture_case "20 records -> reported (exit 0)" 20 no 0 || fail=1
    fixture_case "1 malformed file -> reject (exit 1)" 20 yes 1 || fail=1

    if [ "$fail" -eq 0 ]; then
        printf '%s --self-test: PASS\n' "$PROG"
    else
        printf '%s --self-test: FAIL\n' "$PROG" >&2
    fi
    return "$fail"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

main() {
    local ledger="$LEDGER_DEFAULT" format="text" self_test=0

    while [ $# -gt 0 ]; do
        case "$1" in
            --ledger)
                [ $# -ge 2 ] || { printf '%s: --ledger requires a directory\n' "$PROG" >&2; usage >&2; exit 2; }
                ledger=$2
                shift 2
                ;;
            --format)
                [ $# -ge 2 ] || { printf '%s: --format requires text|json\n' "$PROG" >&2; usage >&2; exit 2; }
                format=$2
                shift 2
                ;;
            --self-test)
                self_test=1
                shift
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            *)
                printf '%s: unknown argument: %s\n' "$PROG" "$1" >&2
                usage >&2
                exit 2
                ;;
        esac
    done

    if [ "$format" != "text" ] && [ "$format" != "json" ]; then
        printf '%s: --format must be text or json, got: %s\n' "$PROG" "$format" >&2
        exit 2
    fi

    command -v jq >/dev/null 2>&1 || { printf '%s: jq is required\n' "$PROG" >&2; exit 2; }

    if [ "$self_test" -eq 1 ]; then
        run_self_test
        exit $?
    fi

    local tmp
    tmp=$(mktemp -d)
    trap '[ -n "$tmp" ] && [ -d "$tmp" ] && rm -rf "$tmp"' EXIT

    report "$ledger" "$format" "$tmp"
    exit $?
}

main "$@"
