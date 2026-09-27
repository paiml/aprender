#!/usr/bin/env bash
# queue_inputs.sh -- FLOW-003 QM-01: measure the queue-model inputs over a trailing window (#4513)
#
#   queue_inputs.sh fetch   <raw-dir> [days]   pull the window's raw GitHub data into <raw-dir> (read-only API)
#   queue_inputs.sh compute <raw-dir>          print queue-inputs-v1 JSON; rc 1 (RED) on an empty window
#                                              or on any input with n = 0
#   queue_inputs.sh ident   <raw-dir>          write <raw-dir>/ident.tsv (red release cycles, mechanical rule)
#   queue_inputs.sh self-test                  planted fixtures, incl. the empty-window and [U] REDs
#
# Every input carries {value, n, window, command, method}. The raw files ARE the receipt: compute reads
# nothing else, so anyone can re-derive the numbers from the committed raw dir.
#
# Definitions (FLOW-003 §2, §3 Q4, §9.1 and infra-5a's v1.1 notes):
#   entry     one merge_group CI run (one per queue head_sha). pr-review-quorum is not a required
#             check in the queue (it failed on 61/62 entries that still merged), so CI alone decides.
#   T         median wall time of successful entries.
#   q'        failed / (failed + passed) entries, cancelled and in-progress dropped. This counts
#             re-entries, so it is q' (per ENTRY), not q.
#   q         q'·f/(1−q'), backed out of q' with measured f (Q4). Entries that were false ejections
#             (re-entered with no push, then merged) are flakes, and are removed first (§9.1).
#   f, F      after an ejection (failed_checks / checks_timed_out) followed by a push and a re-entry:
#             f = share whose next queue exit is `merged`; F = ejection -> first push, median.
#   R         ejection followed by a re-entry with NO push (a false ejection): ejection -> re-entry.
#   lambda    PRs created per hour (λ). Queue entries per hour are λ' (lambda_eff); both reported.
#   phi       share of harvested CI runs (test jobs) with a spurious failure: some test whose
#             TRY 1 failed and a later TRY passed (nextest FLAKY).
#   rho       mean duration of a nextest retry attempt (TRY n >= 2), split rho_mq (merge_group)
#             and rho_rel (release PRs); pooled over every CI run when a split has n = 0.
#   C         release-PR cycle: CI run created -> completed on a release PR head, median.
#   ident     red release cycles whose failing tests name the culprit PR / red release cycles, from
#             the committed classification <raw-dir>/ident.tsv (one row per red cycle).
set -euo pipefail

REPO="paiml/aprender"
# Release PRs: the fold / train branches that carry k PRs in one push (Thm 2's release PR).
RELEASE_BRANCH_RE='^(car/|rc/|release[/-]|batch/|fold/|replace/b[0-9])'
# Jobs whose logs carry nextest output.
TEST_JOB_RE='test|shard|x86-main|yoga|gx10|mac-check|cuda|gpu'

die() { printf 'queue_inputs: %s\n' "$*" >&2; exit 2; }

# ghj <out> <gh args...>: one API read into <out>, atomically. A 403 / rate-limit body never lands as data.
ghj() {
    local out="$1"; shift
    [ -s "$out" ] && return 0                     # resumable: a finished step is not re-fetched
    gh "$@" > "$out.tmp" || { rm -f -- "${out:?}.tmp"; return 1; }
    mv -- "$out.tmp" "$out"
}

fetch() {
    local d="$1" days="${2:-7}" start end
    mkdir -p -- "$d"
    if [ ! -s "$d/window.txt" ]; then             # a resumed fetch keeps its original window
        end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        start=$(date -u -d "$days days ago" +%Y-%m-%dT%H:%M:%SZ)
        printf '%s %s\n' "$start" "$end" > "$d/window.txt"
    fi
    read -r start end < "$d/window.txt"
    local runq='.workflow_runs[] | {id,name,event,head_sha,head_branch,status,conclusion,created_at,updated_at,run_attempt}'

    ghj "$d/mg_runs.jsonl" api --paginate "repos/$REPO/actions/runs?event=merge_group&created=$start..$end&per_page=100" -q "$runq"
    # The runs API returns at most 1000 rows per filtered query (a 7-day ci.yml window has more), so ask
    # one 12-hour slice at a time and refuse a slice that hit the cap: a truncated slice is not data.
    if [ ! -s "$d/ci_runs.jsonl" ]; then
        local a b s0 s1 rows
        a=$(date -u -d "$start" +%s); b=$(date -u -d "$end" +%s)
        : > "$d/ci_runs.part"
        while [ "$a" -lt "$b" ]; do
            s0=$(date -u -d "@$a" +%Y-%m-%dT%H:%M:%SZ)
            s1=$(date -u -d "@$(( a + 43200 < b ? a + 43200 : b ))" +%Y-%m-%dT%H:%M:%SZ)
            gh api --paginate "repos/$REPO/actions/workflows/ci.yml/runs?created=$s0..$s1&per_page=100" \
                -q "$runq" > "$d/slice.tmp" || return 1
            rows=$(wc -l < "$d/slice.tmp")
            [ "$rows" -lt 1000 ] || { printf 'queue_inputs: slice %s hit the 1000-row cap\n' "$s0" >&2; return 1; }
            cat -- "$d/slice.tmp" >> "$d/ci_runs.part"
            a=$(( a + 43200 ))
        done
        jq -sc 'unique_by(.id) | .[]' "$d/ci_runs.part" > "$d/ci_runs.jsonl"
        rm -f -- "${d:?}/ci_runs.part" "${d:?}/slice.tmp"
    fi
    ghj "$d/prs_created.json" pr list -R "$REPO" --state all --search "created:>=${start%T*}" --limit 1000 \
        --json number,createdAt,mergedAt,closedAt,state,headRefName,baseRefName

    # Queue timelines of every PR that entered the queue in the window.
    jq -r 'select(.name=="CI") | .head_branch | capture("pr-(?<n>[0-9]+)-").n' "$d/mg_runs.jsonl" \
        | sort -u > "$d/mq_prs.txt"
    local n
    touch "$d/timelines.jsonl"
    while read -r n; do
        jq -e --argjson n "$n" 'select(.number == $n)' "$d/timelines.jsonl" >/dev/null 2>&1 && continue
        gh api graphql -F n="$n" -f query='query($n:Int!){repository(owner:"paiml",name:"aprender"){pullRequest(number:$n){
            number headRefName
            timelineItems(first:100,itemTypes:[ADDED_TO_MERGE_QUEUE_EVENT,REMOVED_FROM_MERGE_QUEUE_EVENT,HEAD_REF_FORCE_PUSHED_EVENT]){
              nodes{__typename ... on AddedToMergeQueueEvent{createdAt} ... on RemovedFromMergeQueueEvent{createdAt reason}
                    ... on HeadRefForcePushedEvent{createdAt}}}
            commits(last:100){totalCount nodes{commit{oid committedDate}}}}}}' \
            -q '.data.repository.pullRequest' > "$d/tl.tmp" || return 1
        jq -c . "$d/tl.tmp" >> "$d/timelines.jsonl"
    done < "$d/mq_prs.txt"
    rm -f -- "${d:?}/tl.tmp"

    # nextest retry lines from the test jobs of every completed CI run. A run is marked harvested only
    # once EVERY test-job log came back: a failed log read must not count as "no retries" (that biases phi).
    touch "$d/retries.tsv" "$d/harvested_runs.txt"
    local run ev br job jname
    # Only the runs the model is about: queue entries and release-PR heads (every ci.yml log would be
    # thousands of reads against a quota the fleet shares).
    jq -r --arg relre "$RELEASE_BRANCH_RE" 'select((.conclusion=="success" or .conclusion=="failure")
            and (.event == "merge_group" or (.head_branch | test($relre))))
        | "\(.id)\t\(.event)\t\(.head_branch)"' "$d/ci_runs.jsonl" > "$d/to_harvest.tsv"
    while IFS=$'\t' read -r run ev br; do
        grep -qx -- "$run" "$d/harvested_runs.txt" && continue
        gh api "repos/$REPO/actions/runs/$run/jobs?per_page=100" \
            -q '.jobs[] | select(.conclusion=="success" or .conclusion=="failure") | "\(.id)\t\(.name)"' \
            > "$d/jobs.tmp" || return 1
        : > "$d/run.tmp"
        while IFS=$'\t' read -r job jname; do
            printf '%s' "$jname" | grep -Eqi -- "$TEST_JOB_RE" || continue
            gh api "repos/$REPO/actions/jobs/$job/logs" > "$d/log.tmp" || return 1
            sed 's/\x1b\[[0-9;]*m//g' "$d/log.tmp" \
                | { grep -E ' TRY [0-9]+ (PASS|FAIL|SIGSEGV|SIGABRT|TIMEOUT)' || true; } \
                | sed -E 's/^.* TRY ([0-9]+) ([A-Z]+) +\[ *([0-9.]+)s\] +(.*)$/\1\t\2\t\3\t\4/' \
                | awk -F'\t' -v r="$run" -v e="$ev" -v b="$br" -v j="$jname" 'NF==4{print r"\t"e"\t"b"\t"j"\t"$0}' \
                >> "$d/run.tmp"
            sleep 1                                   # the API quota is shared by the whole fleet
        done < "$d/jobs.tmp"
        cat -- "$d/run.tmp" >> "$d/retries.tsv"
        printf '%s\n' "$run" >> "$d/harvested_runs.txt"
    done < "$d/to_harvest.tsv"
    rm -f -- "${d:?}/jobs.tmp" "${d:?}/run.tmp" "${d:?}/log.tmp" "${d:?}/to_harvest.tsv"
}

compute() {
    local d="$1" f
    for f in window.txt mg_runs.jsonl ci_runs.jsonl prs_created.json timelines.jsonl retries.tsv \
             harvested_runs.txt ident.tsv; do
        [ -f "$d/$f" ] || die "missing $d/$f"
    done
    local start end
    read -r start end < "$d/window.txt"

    jq -n --arg start "$start" --arg end "$end" --arg relre "$RELEASE_BRANCH_RE" \
        --slurpfile mg <(cat "$d/mg_runs.jsonl") \
        --slurpfile ci <(cat "$d/ci_runs.jsonl") \
        --slurpfile prs "$d/prs_created.json" \
        --slurpfile tl <(cat "$d/timelines.jsonl") \
        --rawfile retr "$d/retries.tsv" \
        --rawfile harv "$d/harvested_runs.txt" \
        --rawfile ident "$d/ident.tsv" \
        -f /dev/stdin <<'JQ'
def median: sort | if length == 0 then null
    elif length % 2 == 1 then .[length / 2 | floor] else (.[length / 2 - 1] + .[length / 2]) / 2 end;
def mean: if length == 0 then null else add / length end;
def mins($a; $b): (($b | fromdate) - ($a | fromdate)) / 60;
def r4: if . == null then null else (. * 10000 | round) / 10000 end;
def inwin: . >= $start and . <= $end;
def input($v; $n; $cmd; $method): {value: ($v | r4), n: $n, window: "\($start)/\($end)", command: $cmd, method: $method};

($end | fromdate) - ($start | fromdate) | . / 3600 as $hours
| [$mg[] | select(.name == "CI" and (.created_at | inwin))]
  | group_by(.head_sha) | map(max_by(.run_attempt))
  | map(. + {pr: (.head_branch | capture("pr-(?<n>[0-9]+)-").n | tonumber)}) as $entries
| [$entries[] | select(.conclusion == "success")] as $pass
| [$entries[] | select(.conclusion == "failure")] as $fail

# Ejection chains from the PR timelines.
| [ $tl[] | . as $p
    | ([$p.timelineItems.nodes[] | select(.__typename != "HeadRefForcePushedEvent")] | sort_by(.createdAt)) as $q
    | ([($p.commits.nodes[] | .commit.committedDate),
        ($p.timelineItems.nodes[] | select(.__typename == "HeadRefForcePushedEvent") | .createdAt)] | sort) as $push
    | range(0; $q | length) as $i
    | $q[$i]
    | select(.__typename == "RemovedFromMergeQueueEvent"
             and (.reason == "FAILED_CHECKS" or .reason == "failed_checks"
                  or .reason == "CHECKS_TIMED_OUT" or .reason == "checks_timed_out")
             and (.createdAt | inwin))
    | .createdAt as $ej
    | ([$q[$i + 1:][] | select(.__typename == "AddedToMergeQueueEvent")] | first) as $re
    | ([$q[$i + 1:][] | select(.__typename == "RemovedFromMergeQueueEvent")] | first) as $exit
    | ([$push[] | select(. > $ej)] | first) as $fix
    | ([$entries[] | select(.pr == $p.number and $re != null and .created_at >= $re.createdAt)]
       | min_by(.created_at) // null) as $next
    | {pr: $p.number, ejected: $ej, reason: .reason, reentry: ($re.createdAt // null),
       fix_push: $fix,
       pushed_before_reentry: ($fix != null and $re != null and $fix < $re.createdAt),
       next_exit: ($exit.reason // null),
       next_entry: (if $next == null then null else $next.conclusion end)} ] as $ej

| [$ej[] | select(.reentry != null and .pushed_before_reentry
                 and (.next_entry == "success" or .next_entry == "failure"))] as $fixes
| [$ej[] | select(.reentry != null and (.pushed_before_reentry | not) and .next_entry == "success")] as $false_ej
| [$fixes[] | select(.next_entry == "success")] as $fix_ok

# Failed entries that were false ejections are flakes, not defects (§9.1).
| ([$false_ej[] | .pr] ) as $false_prs
| [$fail[] | . as $e | select([$false_ej[] | select(.pr == $e.pr and .ejected >= $e.created_at)] | length > 0)] as $fail_flaky
| ($fail | length) as $nf | ($pass | length) as $np
| (if ($nf + $np) > 0 then $nf / ($nf + $np) else null end) as $qprime_raw
| (if ($nf + $np - ($fail_flaky | length)) > 0
     then ($nf - ($fail_flaky | length)) / ($nf + $np - ($fail_flaky | length)) else null end) as $qprime
| (if ($fixes | length) > 0 then ($fix_ok | length) / ($fixes | length) else null end) as $f
| (if $qprime != null and $f != null and $qprime < 1 then $qprime * $f / (1 - $qprime) else null end) as $q

# nextest attempts.
| [$retr | split("\n")[] | select(length > 0) | split("\t")
   | {run: .[0], event: .[1], branch: .[2], job: .[3], try: (.[4] | tonumber), result: .[5],
      secs: (.[6] | tonumber), test: (.[7] | sub("^\\([^)]*\\) +"; ""))}] as $tries
| [$harv | split("\n")[] | select(length > 0)] as $harvested
| [$tries[] | select(.try >= 2)] as $retries
| [$tries | group_by(.run + "|" + .job + "|" + .test)[]
   | select(any(.[]; .try == 1 and .result != "PASS") and any(.[]; .try >= 2 and .result == "PASS"))
   | .[0].run] | unique as $flaky_runs
| [$retries[] | select(.event == "merge_group") | .secs] as $rho_mq_s
| [$retries[] | select(.branch | test($relre)) | .secs] as $rho_rel_s
| [$retries[] | .secs] as $rho_all_s

# Release cycles.
| [$ci[] | select(.event == "pull_request" and (.head_branch | test($relre)) and (.created_at | inwin)
                 and (.conclusion == "success" or .conclusion == "failure"))] as $rel
| [$rel[] | select(.conclusion == "failure")] as $rel_red
| [$ident | split("\n")[] | select(length > 0 and (startswith("#") | not)) | split("\t")
   | {run: .[0], names_culprit: .[1]}] as $ident_rows
| ([$ident_rows[] | .run] ) as $ident_runs
| [$rel_red[] | (.id | tostring) as $id | select([$ident_runs[] | select(. == $id)] | length == 0) | .id] as $unclassified

| [$prs[][] | select(.createdAt | inwin)] as $created
| [$tl[] | .timelineItems.nodes[] | select(.__typename == "AddedToMergeQueueEvent" and (.createdAt | inwin))] as $enq

| {
  schema: "queue-inputs-v1",
  spec: "FLOW-003",
  row: "QM-01",
  window: {start: $start, end: $end, hours: ($hours | r4)},
  inputs: {
    T: input([$pass[] | mins(.created_at; .updated_at)] | median; ($pass | length);
             "gh api repos/paiml/aprender/actions/runs?event=merge_group (CI, success)";
             "median created->updated of passed queue entries, min"),
    C: input([$rel[] | mins(.created_at; .updated_at)] | median; ($rel | length);
             "gh api repos/paiml/aprender/actions/workflows/ci.yml/runs (pull_request, release branches)";
             "median CI created->completed per release-PR push, min; branches ~ \($relre)"),
    q: input($q; ($nf + $np - ($fail_flaky | length));
             "q = q'*f/(1-q'), q' from merge_group CI entries";
             "per-PR defect rate backed out of per-entry q' (Q4); flaky failed entries removed"),
    phi: input(if ($harvested | length) > 0 then ($flaky_runs | length) / ($harvested | length) else null end;
               ($harvested | length);
               "gh api repos/paiml/aprender/actions/jobs/<id>/logs | grep ' TRY n '";
               "share of harvested CI runs (merge_group + release PRs) with a test whose TRY 1 failed and a later TRY passed"),
    lambda_per_h: input(($created | length) / $hours; ($created | length);
                        "gh pr list -R paiml/aprender --state all --json createdAt";
                        "lambda: PRs created per hour (not queue entries; see derived.lambda_eff_per_h)"),
    f: input($f; ($fixes | length); "GraphQL timelineItems (queue events) + commits, merge_group CI runs";
             "ejected PRs re-entered after a push: share whose next queue CI entry passed"),
    F: input([$fixes[] | mins(.ejected; .fix_push)] | median; ($fixes | length);
             "GraphQL timelineItems (queue events) + commits";
             "median ejection -> first push after it, min"),
    rho_mq: input((if ($rho_mq_s | length) > 0 then ($rho_mq_s | mean) else ($rho_all_s | mean) end) as $v | if $v == null then null else $v / 60 end;
                  if ($rho_mq_s | length) > 0 then ($rho_mq_s | length) else ($rho_all_s | length) end;
                  "nextest TRY n>=2 durations from job logs";
                  if ($rho_mq_s | length) > 0 then "mean retry-attempt seconds / 60, merge_group runs, min"
                  else "no merge_group retry in window: pooled over every CI run, min" end),
    rho_rel: input((if ($rho_rel_s | length) > 0 then ($rho_rel_s | mean) else ($rho_all_s | mean) end) as $v | if $v == null then null else $v / 60 end;
                   if ($rho_rel_s | length) > 0 then ($rho_rel_s | length) else ($rho_all_s | length) end;
                   "nextest TRY n>=2 durations from job logs";
                   if ($rho_rel_s | length) > 0 then "mean retry-attempt seconds / 60, release-PR runs, min"
                   else "no release-PR retry in window: pooled over every CI run, min" end),
    R: input([$false_ej[] | mins(.ejected; .reentry)] | median; ($false_ej | length);
             "GraphQL timelineItems (queue events) + commits";
             "median false ejection (re-entered with no push, and that entry passed) -> re-entry, min"),
    ident_rate: input(if ($ident_rows | length) > 0
                      then ([$ident_rows[] | select(.names_culprit == "yes")] | length) / ($ident_rows | length)
                      else null end;
                      ($ident_rows | length);
                      "ident.tsv: one row per red release-PR CI run (run id, names culprit yes|no, why)";
                      "red release cycles whose failing tests name the culprit PR / red release cycles"),
    window_days: input($hours / 24; 1; "window.txt"; "trailing window length"),
    n_runs: input(($entries | length); ($entries | length); "merge_group CI entries"; "queue entries in window")
  },
  derived: {
    q_prime_raw: ($qprime_raw | r4), q_prime_defect: ($qprime | r4),
    entries: {passed: $np, failed: $nf, failed_flaky: ($fail_flaky | length),
              cancelled: ([$entries[] | select(.conclusion == "cancelled")] | length),
              in_progress: ([$entries[] | select(.conclusion == null)] | length)},
    lambda_eff_per_h: (($enq | length) / $hours | r4), queue_entries_by_event: ($enq | length),
    ejections: {total: ($ej | length), fixed_then_reentered: ($fixes | length),
                false: ($false_ej | length), not_reentered: ([$ej[] | select(.reentry == null)] | length)},
    retries: {mq: ($rho_mq_s | length), release: ($rho_rel_s | length), all: ($rho_all_s | length)},
    release_cycles: {total: ($rel | length), red: ($rel_red | length), unclassified_red: $unclassified},
    ejection_rows: $ej
  }
}
| .unknown = [.inputs | to_entries[] | select(.value.n == 0 or .value.value == null) | .key]
| .low_n = [.inputs | to_entries[] | select(.value.n > 0 and .value.n < 5 and .key != "window_days") | .key]
| .verdict = (if .inputs.n_runs.n == 0 then "RED: empty window"
              elif (.unknown | length) > 0 then "RED: unmeasured inputs \(.unknown | join(","))"
              elif (.derived.release_cycles.unclassified_red | length) > 0 then "RED: red release cycles missing from ident.tsv"
              else "GREEN" end)
JQ
}

# ident <raw-dir>: write ident.tsv, one row per red release-PR CI run in the window. Mechanical rule:
# `yes` when the run's logs name a test that failed on every attempt (the failure points at code, so the
# batch can be bisected to a PR); `no` when nothing names a test (infra, timeout, guard/lint job, runner
# loss). This is an UPPER bound on identification: a named test in a k-PR batch can still touch several
# PRs' diffs. A human may overwrite a row; the file is the receipt either way.
ident() {
    local d="$1" start end
    read -r start end < "$d/window.txt"
    {
        printf '# run\tnames_culprit\twhy\n'
        jq -r --arg s "$start" --arg e "$end" --arg relre "$RELEASE_BRANCH_RE" \
            'select(.event == "pull_request" and (.head_branch | test($relre)) and .conclusion == "failure"
                    and .created_at >= $s and .created_at <= $e) | "\(.id)\t\(.head_branch)"' "$d/ci_runs.jsonl" \
        | while IFS=$'\t' read -r run br; do
            local named
            named=$(awk -F'\t' -v r="$run" '$1 == r { t = $8; sub(/^\([^)]*\) +/, "", t); k = $4 "|" t; if ($6 == "PASS") ok[k] = 1; seen[k] = 1 }
                END { for (k in seen) if (!ok[k]) { split(k, a, "|"); print a[2] } }' "$d/retries.tsv" | sort | head -3 | paste -sd, -)
            if [ -n "$named" ]; then
                printf '%s\tyes\t%s: test failed every attempt: %s\n' "$run" "$br" "$named"
            else
                printf '%s\tno\t%s: no test named in the harvested logs\n' "$run" "$br"
            fi
        done
    } > "$d/ident.tsv"
}

cmd_compute() {
    local out
    out=$(compute "$1")
    printf '%s\n' "$out"
    case "$(printf '%s' "$out" | jq -r .verdict)" in
        GREEN) return 0 ;;
        *) printf '%s\n' "$out" | jq -r '"queue_inputs: \(.verdict)"' >&2; return 1 ;;
    esac
}

self_test() {
    local t fails=0 rc out
    t=$(mktemp -d)
    trap 'rm -rf -- "${t:?}"' RETURN
    mk() {  # mk <dir>: a small planted window
        local d="$1"
        mkdir -p -- "$d"
        printf '2026-09-20T00:00:00Z 2026-09-27T00:00:00Z\n' > "$d/window.txt"
        # 4 entries: pr1 fails then passes after a fix; pr2 fails then passes with no push (flake); pr3 passes.
        cat > "$d/mg_runs.jsonl" <<'EOF'
{"id":1,"name":"CI","event":"merge_group","head_sha":"a1","head_branch":"gh-readonly-queue/main/pr-1-x","status":"completed","conclusion":"failure","created_at":"2026-09-21T00:00:00Z","updated_at":"2026-09-21T00:30:00Z","run_attempt":1}
{"id":2,"name":"CI","event":"merge_group","head_sha":"a2","head_branch":"gh-readonly-queue/main/pr-1-x","status":"completed","conclusion":"success","created_at":"2026-09-21T02:00:00Z","updated_at":"2026-09-21T02:20:00Z","run_attempt":1}
{"id":3,"name":"CI","event":"merge_group","head_sha":"b1","head_branch":"gh-readonly-queue/main/pr-2-x","status":"completed","conclusion":"failure","created_at":"2026-09-22T00:00:00Z","updated_at":"2026-09-22T00:10:00Z","run_attempt":1}
{"id":4,"name":"CI","event":"merge_group","head_sha":"b2","head_branch":"gh-readonly-queue/main/pr-2-x","status":"completed","conclusion":"success","created_at":"2026-09-22T01:00:00Z","updated_at":"2026-09-22T01:40:00Z","run_attempt":1}
{"id":5,"name":"CI","event":"merge_group","head_sha":"c1","head_branch":"gh-readonly-queue/main/pr-3-x","status":"completed","conclusion":"success","created_at":"2026-09-23T00:00:00Z","updated_at":"2026-09-23T00:30:00Z","run_attempt":1}
{"id":6,"name":"pr-review-quorum","event":"merge_group","head_sha":"c1","head_branch":"gh-readonly-queue/main/pr-3-x","status":"completed","conclusion":"failure","created_at":"2026-09-23T00:00:00Z","updated_at":"2026-09-23T00:01:00Z","run_attempt":2}
EOF
        cat > "$d/timelines.jsonl" <<'EOF'
{"number":1,"headRefName":"f1","timelineItems":{"nodes":[{"__typename":"AddedToMergeQueueEvent","createdAt":"2026-09-21T00:00:00Z"},{"__typename":"RemovedFromMergeQueueEvent","createdAt":"2026-09-21T00:30:00Z","reason":"FAILED_CHECKS"},{"__typename":"AddedToMergeQueueEvent","createdAt":"2026-09-21T02:00:00Z"},{"__typename":"RemovedFromMergeQueueEvent","createdAt":"2026-09-21T02:20:00Z","reason":"MERGED"}]},"commits":{"totalCount":2,"nodes":[{"commit":{"oid":"p1","committedDate":"2026-09-20T12:00:00Z"}},{"commit":{"oid":"p2","committedDate":"2026-09-21T01:00:00Z"}}]}}
{"number":2,"headRefName":"f2","timelineItems":{"nodes":[{"__typename":"AddedToMergeQueueEvent","createdAt":"2026-09-22T00:00:00Z"},{"__typename":"RemovedFromMergeQueueEvent","createdAt":"2026-09-22T00:10:00Z","reason":"FAILED_CHECKS"},{"__typename":"AddedToMergeQueueEvent","createdAt":"2026-09-22T01:00:00Z"},{"__typename":"RemovedFromMergeQueueEvent","createdAt":"2026-09-22T01:40:00Z","reason":"MERGED"}]},"commits":{"totalCount":1,"nodes":[{"commit":{"oid":"q1","committedDate":"2026-09-21T12:00:00Z"}}]}}
{"number":3,"headRefName":"f3","timelineItems":{"nodes":[{"__typename":"AddedToMergeQueueEvent","createdAt":"2026-09-23T00:00:00Z"},{"__typename":"RemovedFromMergeQueueEvent","createdAt":"2026-09-23T00:30:00Z","reason":"MERGED"}]},"commits":{"totalCount":1,"nodes":[{"commit":{"oid":"r1","committedDate":"2026-09-22T12:00:00Z"}}]}}
EOF
        # Release PR: one green cycle (60 min), one red cycle (90 min).
        cat > "$d/ci_runs.jsonl" <<'EOF'
{"id":10,"name":"CI","event":"pull_request","head_sha":"r1","head_branch":"car/0.70.0","status":"completed","conclusion":"failure","created_at":"2026-09-24T00:00:00Z","updated_at":"2026-09-24T01:30:00Z","run_attempt":1}
{"id":11,"name":"CI","event":"pull_request","head_sha":"r2","head_branch":"car/0.70.0","status":"completed","conclusion":"success","created_at":"2026-09-24T02:00:00Z","updated_at":"2026-09-24T03:00:00Z","run_attempt":1}
{"id":12,"name":"CI","event":"pull_request","head_sha":"z1","head_branch":"feat/x","status":"completed","conclusion":"success","created_at":"2026-09-24T02:00:00Z","updated_at":"2026-09-24T02:10:00Z","run_attempt":1}
EOF
        printf '[{"number":1,"createdAt":"2026-09-20T06:00:00Z"},{"number":2,"createdAt":"2026-09-21T06:00:00Z"},{"number":3,"createdAt":"2026-09-19T06:00:00Z"}]\n' \
            > "$d/prs_created.json"
        # Run 12 had a flaky test (TRY 1 FAIL, TRY 2 PASS, 30 s); run 11 a retry that failed again.
        printf '12\tpull_request\tfeat/x\tworkspace-test-shard (1/3)\t1\tFAIL\t10.0\t(─────) crate::t_flaky\n' > "$d/retries.tsv"
        printf '12\tpull_request\tfeat/x\tworkspace-test-shard (1/3)\t2\tPASS\t30.0\t(17/90) crate::t_flaky\n' >> "$d/retries.tsv"
        printf '11\tpull_request\tcar/0.70.0\tx86-main\t2\tFAIL\t90.0\tcrate::t_red\n' >> "$d/retries.tsv"
        printf '10\n11\n12\n' > "$d/harvested_runs.txt"
        printf '# run\tnames_culprit\twhy\n10\tyes\tplanted\n' > "$d/ident.tsv"
    }
    check() {  # check <label> <jq expr that must be true> <json>
        if printf '%s' "$3" | jq -e "$2" >/dev/null; then
            printf '  ok    %s\n' "$1"
        else
            printf '  FAIL  %s (%s)\n' "$1" "$2"; fails=$((fails + 1))
        fi
    }

    mk "$t/base"
    out=$(compute "$t/base")
    check "T = median of passed entries (20, 40, 30 -> 30)" '.inputs.T.value == 30 and .inputs.T.n == 3' "$out"
    check "q' raw = 2 failed / 5" '.derived.q_prime_raw == 0.4' "$out"
    check "flaky failed entry (pr2, no push) removed from q'" '.derived.entries.failed_flaky == 1 and .derived.q_prime_defect == 0.25' "$out"
    check "f = 1/1 fixed re-entry merged" '.inputs.f.value == 1 and .inputs.f.n == 1' "$out"
    check "q = q'f/(1-q') = 0.25/0.75" '.inputs.q.value == 0.3333' "$out"
    check "F = ejection 00:30 -> push 01:00 = 30 min" '.inputs.F.value == 30' "$out"
    check "R = false ejection 00:10 -> re-entry 01:00 = 50 min" '.inputs.R.value == 50 and .inputs.R.n == 1' "$out"
    check "lambda counts created PRs in window only (2 / 168 h)" '.inputs.lambda_per_h.n == 2' "$out"
    check "lambda_eff counts queue entries (5)" '.derived.queue_entries_by_event == 5' "$out"
    check "phi = 1 flaky run / 3 harvested (progress prefix differs per attempt)" '.inputs.phi.value == 0.3333' "$out"
    check "rho_rel from release retry (90 s = 1.5 min)" '.inputs.rho_rel.value == 1.5 and .inputs.rho_rel.n == 1' "$out"
    check "rho_mq pooled when no mq retry (mean 30,90 s = 1 min)" '.inputs.rho_mq.value == 1 and (.inputs.rho_mq.method | test("pooled"))' "$out"
    check "C = median of release cycles (90, 60 -> 75)" '.inputs.C.value == 75 and .inputs.C.n == 2' "$out"
    check "ident = 1/1" '.inputs.ident_rate.value == 1' "$out"
    check "pr-review-quorum is not an entry" '.inputs.n_runs.n == 5' "$out"
    check "GREEN with every input measured" '.verdict == "GREEN" and (.unknown | length) == 0' "$out"

    # Empty window: RED, never a pass.
    mk "$t/empty"
    printf '2026-01-01T00:00:00Z 2026-01-08T00:00:00Z\n' > "$t/empty/window.txt"
    rc=0; out=$(cmd_compute "$t/empty" 2>/dev/null) || rc=$?
    check "empty window is RED (rc 1)" "$( [ "$rc" = 1 ] && echo true || echo false) and (.verdict | startswith(\"RED: empty\"))" "$out"

    # An input with n = 0 is [U]: RED.
    mk "$t/noident"
    printf '# run\tnames_culprit\twhy\n' > "$t/noident/ident.tsv"
    rc=0; out=$(cmd_compute "$t/noident" 2>/dev/null) || rc=$?
    check "unmeasured ident_rate is RED (rc 1)" "$( [ "$rc" = 1 ] && echo true || echo false) and (.unknown == [\"ident_rate\"])" "$out"

    # A red release cycle that nobody classified: RED.
    mk "$t/unclass"
    printf '# run\tnames_culprit\twhy\n99\tyes\tsome other run\n' > "$t/unclass/ident.tsv"
    rc=0; out=$(cmd_compute "$t/unclass" 2>/dev/null) || rc=$?
    check "unclassified red release cycle is RED" "$( [ "$rc" = 1 ] && echo true || echo false) and (.verdict | test(\"ident.tsv\"))" "$out"

    # A no-push re-entry that FAILS again is a real failure, not a false ejection.
    mk "$t/refail"
    sed -i 's/"head_sha":"b2","head_branch":"gh-readonly-queue\/main\/pr-2-x","status":"completed","conclusion":"success"/"head_sha":"b2","head_branch":"gh-readonly-queue\/main\/pr-2-x","status":"completed","conclusion":"failure"/' "$t/refail/mg_runs.jsonl"
    out=$(compute "$t/refail")
    check "no-push re-entry that fails again is not a false ejection" '.derived.ejections.false == 0 and .inputs.R.n == 0' "$out"

    # Mutant: pr-review-quorum shares the entry's head_sha (attempt 2, failed). Counting it as CI must
    # turn pr3's passed entry into a failure -- proves the name filter is load-bearing.
    mk "$t/mut"
    sed -i 's/"name":"pr-review-quorum"/"name":"CI"/' "$t/mut/mg_runs.jsonl"
    out=$(compute "$t/mut")
    check "mutant: quorum counted as CI flips a passed entry to failed" '.derived.entries.failed == 3' "$out"

    if [ "$fails" -eq 0 ]; then printf 'queue_inputs self-test: PASS\n'; return 0; fi
    printf 'queue_inputs self-test: %s FAIL\n' "$fails"; return 1
}

case "${1:-}" in
    fetch) [ $# -ge 2 ] || die "usage: fetch <raw-dir> [days]"; fetch "$2" "${3:-7}" ;;
    compute) [ $# -eq 2 ] || die "usage: compute <raw-dir>"; cmd_compute "$2" ;;
    ident) [ $# -eq 2 ] || die "usage: ident <raw-dir>"; ident "$2" ;;
    self-test) self_test ;;
    *) die "usage: queue_inputs.sh fetch <raw-dir> [days] | compute <raw-dir> | self-test" ;;
esac
