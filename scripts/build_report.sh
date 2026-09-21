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
    printf 'Usage: %s [--ledger DIR] [--format text|json] [--self-test [--percentile-probe]]\n' "$PROG"
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
      ((((k * $n) + 99) / 100) | floor) as $raw
      | (if $raw < 1 then 1 elif $raw > $n then $n else $raw end) as $idx
      | ($sorted | .[$idx - 1])
    end;
'

# jq filter: takes a slurped array of ALL valid ledger records (job records and
# non-job records alike) on stdin and emits one JSON "model" object.
# host_class_of(host): the fleet is managed per CLASS (intel / gx10 / yoga /
# lambda / mini — fleet_utilization.sh and every operator rule use it) and the
# ledger has no per-host field worth grouping on, so §5 (amended 2026-09-21,
# ruling on #3271 round 2) reports per host class, DERIVED from the runner
# name's prefix. A record that already carries host_class keeps it; one that
# does not gets it from .host; a prefix nobody declared is "other", never
# dropped and never guessed. The case table in check_build_report.sh covers
# each declared prefix and the unknown one.
JQ_HOST_CLASS='
def host_class_of(host):
  if host == null then "other"
  elif (host | test("^intel"))  then "intel"
  elif (host | test("^gx10"))   then "gx10"
  elif (host | test("^yoga"))   then "yoga"
  elif (host | test("^lambda")) then "lambda"
  elif (host | test("^mini"))   then "mini"
  else "other" end;
'

JQ_MODEL="$JQ_PERCENTILE""$JQ_HOST_CLASS"'
. as $all
| ($all | length) as $total_valid
| ($all
   | map(select(has("total_s") and (.total_s != null)))
   | map(. + { host_class: (.host_class // host_class_of(.host)) })
  ) as $jobs
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
| (
    # The REQUIRED-CHECK SET, not one name (§5 amended 2026-09-21, ruling on
    # #3271 round 2). Two mechanisms answer "what is required on main" — classic
    # branch protection and ruleset 13878864 — and they spell the gate
    # differently (`ci / gate` vs bare `gate`); keying PRs/train on one name is
    # the one-mechanism error. The set is DERIVED at run time by
    # required_checks_on_main() below (protection + rulesets via gh) and passed
    # in as $required together with where it came from; when gh cannot answer,
    # the documented set is used and the report SAYS so. The slowest member binds.
    $required
    | map(
        . as $rn
        | ($jobs | map(select(.job == $rn) | .total_s)) as $t
        | if ($t | length) > 0
          then { job: $rn, n: ($t | length), p95_total_s: ($t | percentile(95)) }
          else empty end
      )
    | sort_by(-.p95_total_s, .job)
  ) as $required_measured
| (($required_measured | .[0]) // null) as $binding
| (if $binding == null then null else $binding.job end) as $binding_job
| (if $binding == null then null else $binding.p95_total_s end) as $binding_p95_s
| (if $binding == null then null else (((3 * 72 * 3600) / $binding_p95_s) | floor) end) as $prs_per_train
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
    required_checks: $required,
    required_source: $required_source,
    required_measured: $required_measured,
    binding_job: $binding_job,
    binding_p95_s: $binding_p95_s,
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

    # A ledger record is a JSON OBJECT. Anything else under the tree is a reject,
    # not a crash and not a silent drop: `has()` over a non-object aborts jq with
    # exit 5, and an empty or whitespace-only file parses to nothing and leaves no
    # trace at all. Both read as a pass. (Found 3/3 by the PMAT-3225 review quorum.)
    if printf '%s\0' "${files[@]}" \
        | xargs -0 jq -c 'if type == "object" then . else error("not a JSON object") end' \
            >"$out" 2>"$err"
    then
        # Every file must have produced exactly one record; fewer means a file
        # parsed to nothing (empty/whitespace). Attribute it in the slow path.
        if [ "$(wc -l <"$out")" -eq "$nfiles" ]; then
            printf '0'
            return 0
        fi
    fi

    : >"$out"
    : >"$err"
    local reject_count=0
    local f before
    for f in "${files[@]}"; do
        before=$(wc -l <"$out")
        if ! jq -c 'if type == "object" then . else error("not a JSON object") end' \
                "$f" >>"$out" 2>>"$err"
        then
            reject_count=$((reject_count + 1))
        elif [ "$(wc -l <"$out")" -eq "$before" ]; then
            # parsed cleanly to no value at all: empty or whitespace only
            printf 'empty or whitespace-only: %s\n' "$f" >>"$err"
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
# required_checks_on_main TMP -> a JSON array of required-check names on main,
# on stdout, from BOTH mechanisms (classic protection + rulesets), with the
# gate's second spelling added because the ledger records it under both. Writes
# where the set came from to TMP/required.source: `derived` or
# `fallback (<reason>)`. Returns 1 (and prints nothing) when gh cannot answer,
# so the caller falls back to the documented set and prints that it did — a
# report that cannot ask GitHub must not pretend it did.
required_checks_on_main() {
    local tmp=$1 prot rules names
    # A PINNED set wins over a derived one, and says so. This exists for the
    # case table: a self-test whose binding-check expectation depended on what
    # branch protection says TODAY would be a network- and state-dependent
    # guard (quorum round 3 on #3271, lane 1 — an asserted finding, and right).
    # It is an override for tests and operators, never a default.
    if [ -n "${BUILD_REPORT_REQUIRED_CHECKS:-}" ]; then
        if printf '%s' "$BUILD_REPORT_REQUIRED_CHECKS" | jq -e 'type == "array" and length > 0 and all(type == "string")' >/dev/null 2>&1; then
            printf 'pinned (BUILD_REPORT_REQUIRED_CHECKS)' > "$tmp/required.source"
            printf '%s' "$BUILD_REPORT_REQUIRED_CHECKS" | jq -c .
            return 0
        fi
        printf 'fallback (BUILD_REPORT_REQUIRED_CHECKS is not a non-empty JSON array of strings)' > "$tmp/required.source"
        return 1
    fi
    if ! command -v gh >/dev/null 2>&1; then
        printf 'fallback (gh not on PATH)' > "$tmp/required.source"; return 1
    fi
    prot=$(gh api repos/paiml/aprender/branches/main/protection \
             --jq '.required_status_checks.contexts // [] | .[]' 2>/dev/null) || {
        printf 'fallback (gh api branches/main/protection failed)' > "$tmp/required.source"; return 1; }
    rules=$(gh api repos/paiml/aprender/rules/branches/main \
             --jq '.[] | select(.type=="required_status_checks") | .parameters.required_status_checks[]?.context' 2>/dev/null) || rules=""
    names=$(printf '%s\n%s\n' "$prot" "$rules" | sed '/^$/d' | sort -u)
    [ -n "$names" ] || { printf 'fallback (both mechanisms returned an empty set)' > "$tmp/required.source"; return 1; }
    # The ledger records the gate under `ci / gate` AND bare `gate`; if either
    # spelling is required, measure both so the table is complete.
    if printf '%s\n' "$names" | grep -qxE 'ci / gate|gate'; then
        names=$(printf '%s\nci / gate\ngate\n' "$names" | sort -u)
    fi
    printf 'derived' > "$tmp/required.source"
    printf '%s\n' "$names" | jq -R . | jq -s .
}

report() {
    local ledger=$1 format=$2 tmp=$3

    local ndjson="$tmp/records.ndjson" err="$tmp/jq.err"
    local reject_count
    reject_count=$(build_ndjson "$ledger" "$ndjson" "$err")

    if [ "$reject_count" -gt 0 ]; then
        printf 'reject: %s parse error(s) under %s\n' "$reject_count" "$ledger" >&2
        return 1
    fi

    local required_json required_source
    required_json=$(required_checks_on_main "$tmp") || required_json='["ci / gate", "gate", "workspace-test"]'
    required_source=$(cat "$tmp/required.source" 2>/dev/null || printf 'fallback')

    local model_json
    model_json=$(jq -s --argjson required "$required_json" --arg required_source "$required_source" \
        "$JQ_MODEL" "$ndjson")

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

    local binding_job binding_p95_s
    binding_job=$(printf '%s' "$model_json" | jq -r '.binding_job')
    binding_p95_s=$(printf '%s' "$model_json" | jq -r '.binding_p95_s')

    printf '\n'
    printf 'required checks on main, p95 total_s (the SLOWEST one binds):\n'
    printf '%s' "$model_json" | jq -r \
        '.required_measured[] | [.job, (.n|tostring), (.p95_total_s|tostring)] | @tsv' |
        while IFS=$'\t' read -r job n p95t; do
            printf '  %-20s %6s %12s\n' "$job" "$n" "$(disp "$p95t")"
        done

    printf '\n'
    if [ "$binding_job" = "null" ]; then
        printf '%s1 max PRs/train: [U] (no record of any required check)\n' "$(printf '\xc2\xa7')"
    else
        printf '%s1 max PRs/train ~= 3 x 72h / p95(binding required check) = 3 x 259200 / %s = %s\n' \
            "$(printf '\xc2\xa7')" "$binding_p95_s" "$prs_per_train"
        printf '         binding check: %s (p95 %s s)\n' "$binding_job" "$binding_p95_s"
        printf '         required set: %s [%s]\n' \
            "$(printf '%s' "$model_json" | jq -r '.required_checks | join(", ")')" \
            "$(printf '%s' "$model_json" | jq -r '.required_source')"
        if [ "$ci_gate_p95_s" != "null" ] && [ "$binding_job" != "ci / gate" ]; then
            printf '         NOTE: %s1 keys this on `ci / gate` (p95 %s s), which is NOT the slowest\n' \
                "$(printf '\xc2\xa7')" "$ci_gate_p95_s"
            printf '         required check. A PR merges when the slowest one is green, so the\n'
            printf '         %s1 equation as written overstates throughput by %sx.\n' \
                "$(printf '\xc2\xa7')" \
                "$(awk -v b="$binding_p95_s" -v c="$ci_gate_p95_s" 'BEGIN { printf "%.1f", b / c }')"
        fi
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

# write_fixture_records <dir> <n> — n well-formed job records, used by both cases.
write_fixture_records() {
    local dir=$1 n=$2 i=0
    while [ "$i" -lt "$n" ]; do
        printf '{"spec":"APR-RELEASE-001","sha":"selftest%03d","host":"intel-w1","host_class":"intel","job":"ci / gate","queue_wait_s":%d,"exec_s":%d,"total_s":%d,"exit":0}\n' \
            "$i" "$i" "$((i * 2))" "$((i * 3 + 1))" >"$dir/rec-$i.json"
        i=$((i + 1))
    done
}

# fixture_case <label> <nrecords> <inject_malformed> <want_rc>
fixture_case() {
    local label=$1 n=$2 inject_malformed=$3 want_rc=$4
    local dir got_rc
    dir=$(mktemp -d)
    write_fixture_records "$dir" "$n"
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

# shape_case <name> <file-content> <want_rc> — a file that is valid-ish JSON but
# not an object, or that parses to no value at all, must land on the SAME contract
# as unparseable text: exit 1 with `reject:`. jq aborts with exit 5 on has() over a
# non-object, and drops an empty file without a trace; both read as a pass.
shape_case() {
    local name=$1 content=$2 want=$3
    local dir rc
    dir=$(mktemp -d)
    write_fixture_records "$dir" 20
    printf '%s' "$content" >"$dir/odd.json"
    set +e
    "$SELF" --ledger "$dir" >/dev/null 2>/dev/null
    rc=$?
    set -e
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
    if [ "$rc" = "$want" ]; then
        printf '  ok   %-40s want=%-6s got=%s\n' "$name" "$want" "$rc"
        return 0
    fi
    printf '  FAIL %-40s want=%-6s got=%s\n' "$name" "$want" "$rc"
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
    # Float ceil made these wrong: (7/100.0)*100 is 7.000000000000001, so p7 of
    # 1..100 answered 8. k=50/95/100 all land on exact binary fractions and never
    # reach it — the case table was too coarse, not wrong (PMAT-3225 quorum, 1/3).
    pct_case "percentile(7) over 1..100 (float-ceil trap)" "$vec100" 7 7 || fail=1
    pct_case "percentile(29) over 1..100 (float-ceil trap)" "$vec100" 29 29 || fail=1
    pct_case "percentile(1) over 1..100" "$vec100" 1 1 || fail=1
    pct_case "percentile(50) over 1..3 (rounds up)" "[1,2,3]" 50 2 || fail=1

    if [ "${PERCENTILE_PROBE:-0}" -eq 1 ]; then
        printf -- '-- percentile probe over 1..100 --\n'
        local k
        for k in 1 7 25 29 50 75 95 99 100; do
            printf 'p%s=%s\n' "$k" \
                "$(printf '%s' "$vec100" | jq -r "$JQ_PERCENTILE"' percentile('"$k"')')"
        done
    fi

    printf -- '-- exit-code contract (fixtures under mktemp -d) --\n'
    fixture_case "0 records -> decline (exit 2)" 0 no 2 || fail=1
    fixture_case "19 records -> decline (exit 2)" 19 no 2 || fail=1
    fixture_case "20 records -> reported (exit 0)" 20 no 0 || fail=1
    fixture_case "1 malformed file -> reject (exit 1)" 20 yes 1 || fail=1
    shape_case "JSON array []"        '[]'   1 || fail=1
    shape_case "JSON number"          '123'  1 || fail=1
    shape_case "JSON string"          '"s"'  1 || fail=1
    shape_case "JSON true"            'true' 1 || fail=1
    shape_case "JSON null"            'null' 1 || fail=1
    shape_case "empty file"           ''     1 || fail=1
    shape_case "whitespace-only file" '   '  1 || fail=1

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
            --percentile-probe)
                PERCENTILE_PROBE=1
                shift
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
