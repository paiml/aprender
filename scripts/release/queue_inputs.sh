#!/usr/bin/env bash
# queue_inputs.sh -- FLOW-003 QM-01 + QM-08: queue-model inputs and per-class q over a trailing window (#4513, #4519)
#
#   queue_inputs.sh fetch   <raw-dir> [days]   pull the window's raw GitHub data into <raw-dir> (read-only API)
#   queue_inputs.sh compute <raw-dir>          print queue-inputs-v1 JSON; rc 1 (RED) on an empty window
#                                              or on any input with n = 0
#   queue_inputs.sh ident   <raw-dir>          write <raw-dir>/ident.tsv (red release cycles, mechanical rule)
#   queue_inputs.sh readset <tree> <raw-dir>   write <raw-dir>/readset.txt: the .md paths a build/test reads
#   queue_inputs.sh prop12  <queue-inputs.json> print the Prop 12 verdict per service class (#4519)
#   queue_inputs.sh untangle [inbox-dir] [days] weekly untangle tally: conflicts per PR + first-CI-run green
#                                              rate (target >= 0.9) from '| untangle |' inbox lines; no rows = NO-DATA, rc 1
#   queue_inputs.sh dora-fetch <raw-dir> [days] / dora <raw-dir>   weekly DORA table (lead time, CI p50,
#                                              PR age, conflicts, change-fail, release cycle, merge commits)
#   queue_inputs.sh runner-wait-fetch <raw-dir> [days] / runner-wait <raw-dir> [runner-re]   queued->started
#                                              p50/p90 of jobs on matching runners (default ^framework16, infra#1237)
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
#             f = share whose next queue CI entry passed; F = ejection -> first push, median.
#   R         ejection, then a re-entry with NO push whose entry passed (a false ejection): ejection -> re-entry.
#   lambda    PRs created per hour (λ). Queue entries per hour are λ' (lambda_eff); both reported.
#   phi       share of harvested CI runs (test jobs) with a spurious failure: some test whose
#             TRY 1 failed and a later TRY passed (nextest FLAKY).
#   rho       mean duration of a nextest retry attempt (TRY n >= 2), split rho_mq (merge_group)
#             and rho_rel (release PRs); pooled over every CI run when a split has n = 0.
#   C         release-PR cycle: CI run created -> completed on a release PR head, median.
#   ident     red release cycles whose failing tests name the culprit PR / red release cycles, from
#             the committed classification <raw-dir>/ident.tsv (one row per red cycle).
#
# Service classes (FLOW-003 v1.1 §5.1, QM-08 / #4519). A queue entry takes its PR's class:
#   d  docs-only: every changed file is .md, the list is complete (<= 100 files), and no file is in the
#      read set: an include_str!/include_bytes! target, a ".md" string literal in any .rs/.sh/.py/.yml/
#      .toml/Makefile (tests, guards and build.rs read files by name), or under a directory literal.
#      Conservative by construction: a literal "README.md" sends EVERY README.md to x.
#   a  maintainer-attested fork: isCrossRepository and the $ATTEST_LABEL label. Checked after d.
#   x  every other PR.
# Per class: pi (entry share), pi_eff (C1: share of entries with no x entry building ahead of them),
# q_eff (per entry, flakes removed), q (backed out with the global f), T (median passed entry), Q
# (review minutes; not logged anywhere, so [U]). Prop 12: q_c < q* = Q_saving/(kappa_diff*D), inputs [A]
# from §5.1 (30/(0.5*240) = 0.25). PASS needs the 95% Wilson upper bound on q_c below q*; q_c >= q* is
# FAIL; anything between, or n = 0, is NOT-DECIDED (S-2: an unmeasured input never decides).
set -euo pipefail

REPO="paiml/aprender"
# Release PRs: the fold / train branches that carry k PRs in one push (Thm 2's release PR).
RELEASE_BRANCH_RE='^(car/|rc/|release[/-]|batch/|fold/|replace/b[0-9])'
# Closed forms from FLOW-003 §4, shared by compute and the self-test oracle rows.
MODEL_JQ=$(cat <<'JQ'
# Lemma 1: EM(k) = sum_{n>=0} 1 - (1 - q(1-f)^n)^k (truncated once a term is < 1e-12, at most 500 terms).
def em($k; $q; $f): [range(0; 500) as $n | 1 - pow(1 - $q * pow(1 - $f; $n); $k)] | map(select(. >= 1e-12)) | add // 0;
# Theorem 1: E[T_fold(k,r)] = EM (C + r rho + F) + (C + rho S_r) / (1 - phi^(r+1)), S_r = sum_{j=1..r} phi^j.
def t_fold($k; $r; $q; $f; $c; $ff; $phi; $rho):
  em($k; $q; $f) * ($c + $r * $rho + $ff) + ($c + $rho * ([range(1; $r + 1) as $j | pow($phi; $j)] | add // 0)) / (1 - pow($phi; $r + 1));
# Corollary 3 break-even: phi (C - rho) / (1 - phi^2) = rho EM, positive root.
def phi_star($em; $c; $rho): if $rho <= 0 or $em <= 0 then null
  else ((-($c - $rho)) + ((($c - $rho) * ($c - $rho)) + 4 * $rho * $em * $rho * $em | sqrt)) / (2 * $rho * $em) end;
JQ
)
# The maintainer attestation label (§5.1 class a). No such label exists in paiml/aprender as of
# 2026-09-27, so class a is empty and never PASSes.
ATTEST_LABEL="${ATTEST_LABEL:-maintainer-attested}"
# Prop 12 inputs, all [A] (FLOW-003 v1.1 §5.1 illustration).
P12_Q_SAVING=30 P12_KAPPA_DIFF=0.5 P12_D=240
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

# fetch_ci_runs <dir> <start> <end> <jq> [endpoint] [out]: every run created in the window into <dir>/<out>
# (default: ci.yml runs into ci_runs.jsonl; endpoint "actions/runs" lists every workflow).
fetch_ci_runs() {
    local d="$1" start="$2" end="$3" runq="$4" ep="${5:-actions/workflows/ci.yml/runs}" out="${6:-ci_runs.jsonl}"
    # The runs API returns at most 1000 rows per filtered query (a 7-day ci.yml window has more), so ask
    # one 12-hour slice at a time and refuse a slice that hit the cap: a truncated slice is not data.
    if [ ! -s "$d/$out" ]; then
        local a b s0 s1 rows
        a=$(date -u -d "$start" +%s); b=$(date -u -d "$end" +%s)
        : > "$d/ci_runs.part"
        while [ "$a" -lt "$b" ]; do
            s0=$(date -u -d "@$a" +%Y-%m-%dT%H:%M:%SZ)
            s1=$(date -u -d "@$(( a + 43200 < b ? a + 43200 : b ))" +%Y-%m-%dT%H:%M:%SZ)
            gh api --paginate "repos/$REPO/$ep?created=$s0..$s1&per_page=100" \
                -q "$runq" > "$d/slice.tmp" || return 1
            rows=$(wc -l < "$d/slice.tmp")
            [ "$rows" -lt 1000 ] || { printf 'queue_inputs: slice %s hit the 1000-row cap\n' "$s0" >&2; return 1; }
            cat -- "$d/slice.tmp" >> "$d/ci_runs.part"
            a=$(( a + 43200 ))
        done
        jq -sc 'unique_by(.id) | .[]' "$d/ci_runs.part" > "$d/$out"
        rm -f -- "${d:?}/ci_runs.part" "${d:?}/slice.tmp"
    fi
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
    fetch_ci_runs "$d" "$start" "$end" "$runq" || return 1
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

    # Class inputs: changed files, fork flag and labels of every queued PR.
    if [ ! -s "$d/pr_meta.jsonl" ]; then
        local batch q
        : > "$d/pr_meta.part"
        while mapfile -t -n 25 batch && [ "${#batch[@]}" -gt 0 ]; do
            q='query{repository(owner:"paiml",name:"aprender"){'
            for n in "${batch[@]}"; do
                q+="p$n:pullRequest(number:$n){number isCrossRepository changedFiles
                    files(first:100){nodes{path}} labels(first:30){nodes{name}}}"
            done
            q+='}}'
            gh api graphql -f query="$q" -q '.data.repository[] | {number, isCrossRepository, changedFiles,
                files: [.files.nodes[].path], labels: [.labels.nodes[].name]}' | jq -c . >> "$d/pr_meta.part" || return 1
        done < <({ cat -- "$d/mq_prs.txt"; jq -r --arg s "$start" '.[] | select(.createdAt >= $s) | .number' \
                    "$d/prs_created.json"; } | sort -un)
        mv -- "$d/pr_meta.part" "$d/pr_meta.jsonl"
    fi

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
             harvested_runs.txt ident.tsv pr_meta.jsonl readset.txt; do
        [ -f "$d/$f" ] || die "missing $d/$f"
    done
    local start end
    read -r start end < "$d/window.txt"

    { printf '%s\n' "$MODEL_JQ"; cat <<'JQ'
def median: sort | if length == 0 then null
    elif length % 2 == 1 then .[length / 2 | floor] else (.[length / 2 - 1] + .[length / 2]) / 2 end;
def mean: if length == 0 then null else add / length end;
def mins($a; $b): (($b | fromdate) - ($a | fromdate)) / 60;
def r4: if . == null then null else (. * 10000 | round) / 10000 end;
def inwin: . >= $start and . <= $end;
def input($v; $n; $cmd; $method): {value: ($v | r4), n: $n, window: "\($start)/\($end)", command: $cmd, method: $method, mark: "[V]"};
def wilson_hi($k; $n): if $n == 0 then null else
    ($k / $n) as $p | 1.959964 as $z | ($z * $z) as $z2
    | ($p + $z2 / (2 * $n) + $z * ((($p * (1 - $p)) / $n + $z2 / (4 * $n * $n)) | sqrt)) / (1 + $z2 / $n) end;
def backout($qe; $f): if $qe == null or $f == null or $qe >= 1 then null else $qe * $f / (1 - $qe) end;

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

# Service classes (§5.1).
| [$readset | split("\n")[] | select(length > 0 and (startswith("#") | not))] as $rs
| (reduce $meta[] as $m ({};
      . + {($m.number | tostring):
        (if ($m.changedFiles <= 100 and ($m.files | length) == $m.changedFiles and ($m.files | length) > 0
             and all($m.files[]; endswith(".md")
                 and (. as $p | any($rs[]; . as $l | $p == $l or ($p | endswith("/" + $l))
                                              or (($l | endswith("/")) and ($p | startswith($l)))) | not)))
         then "d"
         elif ($m.isCrossRepository and any($m.labels[]; . == $attest)) then "a"
         else "x" end)})) as $cls
| [$entries[] | . + {class: ($cls[.pr | tostring] // "x")}] as $centries
# One row per PR that left the queue merged in the window.
| [ $tl[] | . as $p
    | ([$p.timelineItems.nodes[] | select(.__typename == "AddedToMergeQueueEvent" or .__typename == "RemovedFromMergeQueueEvent")]
       | sort_by(.createdAt)) as $mq
    | ([$mq[] | select(.__typename == "RemovedFromMergeQueueEvent" and (.reason | ascii_downcase) == "merged"
                        and (.createdAt | inwin))] | first) as $m
    | select($m != null)
    | ([$mq[] | select(.__typename == "RemovedFromMergeQueueEvent" and .createdAt <= $m.createdAt)]) as $exits
    | ([$mq[] | select(.__typename == "AddedToMergeQueueEvent" and .createdAt <= $m.createdAt)]) as $adds
    | {pr: $p.number, class: ($cls[$p.number | tostring] // "x"),
       mq_wait_min: (if ($adds | length) > 0 then (mins($adds[0].createdAt; $m.createdAt) | r4) else null end),
       entries: ($adds | length),
       first_try: (($exits[0].reason // "") | ascii_downcase
                   | if . == "merged" then "pass" elif . == "failed_checks" or . == "checks_timed_out" then "fail" else . end),
       ejects: ([$exits[] | select((.reason | ascii_downcase) == "failed_checks" or (.reason | ascii_downcase) == "checks_timed_out")] | length),
       merged_at: $m.createdAt} ] as $merged_rows
| ($qsave / ($kdiff * $dcost)) as $qstar
| (reduce ("d", "a", "x") as $c ({};
    . + {($c): (
      [$centries[] | select(.class == $c)] as $ce
      | [$ce[] | select(.conclusion == "success")] as $cp
      | [$ce[] | select(.conclusion == "failure")] as $cfa
      | [$cfa[] | . as $e | select([$fail_flaky[] | select(.id == $e.id)] | length > 0)] as $cff
      | (($cp | length) + ($cfa | length) - ($cff | length)) as $cn
      | (($cfa | length) - ($cff | length)) as $ck
      | (if $cn > 0 then $ck / $cn else null end) as $cqe
      | backout($cqe; $f) as $cq
      | backout(wilson_hi($ck; $cn); $f) as $cqhi
      | [$ce[] | . as $e | select($c != "x" and ([$centries[] | select(.class == "x" and .created_at < $e.created_at
                                                   and .updated_at > $e.created_at)] | length) == 0)] as $cheap
      | {pi: input(if ($entries | length) > 0 then ($ce | length) / ($entries | length) else null end; ($entries | length);
                   "merge_group CI entries + pr_meta.jsonl class"; "class share of queue entries"),
         pi_eff: (input(if ($entries | length) > 0 then ($cheap | length) / ($entries | length) else null end; ($entries | length);
                   "merge_group CI entries + pr_meta.jsonl class";
                   "C1: class entries with no x-class entry building ahead at creation / all entries (x: 1 - others)")),
         q_eff: input($cqe; $cn; "merge_group CI entries of this class (flakes removed)"; "per-entry failure rate q'_c"),
         q: (input($cq; $cn; "q_c = q'_c*f/(1-q'_c) with the global f"; "per-PR defect rate, backed out per class (Q4)") | .mark = "[C]"),
         q_upper95: ($cqhi | r4),
         T: input([$cp[] | mins(.created_at; .updated_at)] | median; ($cp | length);
                  "merge_group CI entries of this class, success"; "median passed-entry build time today (full CI for every class), min"),
         Q: {value: null, n: 0, window: "\($start)/\($end)", command: "none: review minutes are not logged", method: "review cost per PR", mark: "[U]"},
         prop12: {q_star: ($qstar | r4), mark: "[A]",
                  verdict: (if $cn == 0 or $cq == null then "NOT-DECIDED"
                            elif $cq >= $qstar then "FAIL"
                            elif $cqhi != null and $cqhi < $qstar then "PASS"
                            else "NOT-DECIDED" end)}}
    )})) as $classes0
| ($classes0 | .x.pi_eff.value = ((1 - ($classes0.d.pi_eff.value // 0) - ($classes0.a.pi_eff.value // 0)) | r4)) as $classes

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
             "per-PR defect rate backed out of per-entry q' (Q4); flaky failed entries removed")
       + {mark: "[C]", measuredRate: "q"},
    phi: input(if ($harvested | length) > 0 then ($flaky_runs | length) / ($harvested | length) else null end;
               ($harvested | length);
               "gh api repos/paiml/aprender/actions/jobs/<id>/logs | grep ' TRY n '";
               "share of harvested CI runs (merge_group + release PRs) with a test whose TRY 1 failed and a later TRY passed"),
    q_eff: (input($qprime; ($nf + $np - ($fail_flaky | length));
                  "gh api repos/paiml/aprender/actions/runs?event=merge_group (CI, success|failure)";
                  "per-entry failure rate q' (includes re-entries), flaky failed entries removed")
             | .measuredRate = "q_eff"),
    lambda_eff_per_h: (input(($enq | length) / $hours; ($enq | length);
                             "GraphQL AddedToMergeQueueEvent per queued PR";
                             "lambda': queue entries per hour (includes re-entries)") | .measuredRate = "lambda_eff"),
    lambda_per_h: input(($created | length) / $hours; ($created | length);
                        "gh pr list -R paiml/aprender --state all --json createdAt";
                        "lambda: PRs created per hour (not queue entries)") + {measuredRate: "lambda"},
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
  classes: $classes,
  review: {Q_saving: {value: $qsave, mark: "[A]"}, kappa_diff: {value: $kdiff, mark: "[A]"}, D: {value: $dcost, mark: "[A]"},
           kappa_full: {value: null, mark: "[U]"}, kappa_cheap: {value: null, mark: "[U]"},
           source: "FLOW-003 v1.1 §5.1 Prop 12 illustration"},
  derived: {
    class_of_pr: $cls,
    created_prs_by_class: ([$created[] | $cls[.number | tostring] // "unfetched"] | group_by(.) | map({(.[0]): length}) | add),
    q_prime_raw: ($qprime_raw | r4), q_prime_defect: ($qprime | r4),
    entries: {passed: $np, failed: $nf, failed_flaky: ($fail_flaky | length),
              cancelled: ([$entries[] | select(.conclusion == "cancelled")] | length),
              in_progress: ([$entries[] | select(.conclusion == null)] | length)},
    lambda_eff_per_h: (($enq | length) / $hours | r4), queue_entries_by_event: ($enq | length),
    ejections: {total: ($ej | length), fixed_then_reentered: ($fixes | length),
                false: ($false_ej | length), not_reentered: ([$ej[] | select(.reentry == null)] | length)},
    retries: {mq: ($rho_mq_s | length), release: ($rho_rel_s | length), all: ($rho_all_s | length)},
    release_cycles: {total: ($rel | length), red: ($rel_red | length), unclassified_red: $unclassified},
    ejection_rows: $ej,
    merged_pr_rows: $merged_rows
  }
}
# Prop 11 (B = 1, r = 0): rho_HOL = lambda' * sum(pi_eff q'_c) * sum(pi_eff T_c); blind = lambda' q' T.
| .derived.rho_hol = (
    (.inputs.lambda_eff_per_h.value / 60) as $lm
    | [.classes[] | select(.pi_eff.value > 0)] as $cs
    | (if any($cs[]; .q_eff.value == null or .T.value == null) then null
       else {split: ($lm * ([$cs[] | .pi_eff.value * .q_eff.value] | add) * ([$cs[] | .pi_eff.value * .T.value] | add) | r4),
             blind: ($lm * .inputs.q_eff.value * .inputs.T.value | r4),
             formula: "FLOW-003 v1.1 Prop 11: lambda' qbar' Tbar, B=1, r=0 (lower bound, Thm 8 tightness)"} end))
# Fold size (Thm 1, Lemma 1, Cor 3) at the measured inputs, k = 1..8, r = 0..3. The bisect bound adds
# ceil(log2 k) cycles to each defect cycle that names no test (share 1 - ident_rate; Thm 2 remark, QM-07).
| .inputs as $in
| .derived.fold = (
    if any($in.q, $in.f, $in.C, $in.F, $in.phi, $in.rho_rel, $in.ident_rate; .value == null) then null
    else [range(1; 9) as $k
          | em($k; $in.q.value; $in.f.value) as $em
          | [range(0; 4) as $r | t_fold($k; $r; $in.q.value; $in.f.value; $in.C.value; $in.F.value; $in.phi.value; $in.rho_rel.value)] as $t
          | ($t | index($t | min)) as $rbest
          | {k: $k, EM: ($em | r4), E_T_fold_by_r: ($t | map(r4)), r_best: $rbest,
             E_T_fold_bisect_upper: ($t[$rbest] + $em * (1 - $in.ident_rate.value)
                                       * ([range(0; 4)] | map(select(pow(2; .) >= $k)) | first) * $in.C.value | r4),
             phi_star_release: (phi_star($em; $in.C.value; $in.rho_rel.value) | r4)}]
    end)
| .unknown = [.inputs | to_entries[] | select(.value.n == 0 or .value.value == null) | .key]
| .low_n = [.inputs | to_entries[] | select(.value.n > 0 and .value.n < 5 and .key != "window_days") | .key]
| .verdict = (if .inputs.n_runs.n == 0 then "RED: empty window"
              elif (.unknown | length) > 0 then "RED: unmeasured inputs \(.unknown | join(","))"
              elif (.derived.release_cycles.unclassified_red | length) > 0 then "RED: red release cycles missing from ident.tsv"
              else "GREEN" end)
JQ
    } | jq -n --arg start "$start" --arg end "$end" --arg relre "$RELEASE_BRANCH_RE" \
        --slurpfile mg <(cat "$d/mg_runs.jsonl") \
        --slurpfile ci <(cat "$d/ci_runs.jsonl") \
        --slurpfile prs "$d/prs_created.json" \
        --slurpfile tl <(cat "$d/timelines.jsonl") \
        --rawfile retr "$d/retries.tsv" \
        --rawfile harv "$d/harvested_runs.txt" \
        --rawfile ident "$d/ident.tsv" \
        --slurpfile meta <(cat "$d/pr_meta.jsonl") \
        --rawfile readset "$d/readset.txt" \
        --arg attest "$ATTEST_LABEL" \
        --argjson qsave "$P12_Q_SAVING" --argjson kdiff "$P12_KAPPA_DIFF" --argjson dcost "$P12_D" \
        -f /dev/stdin
}

# readset <tree> <raw-dir>: the paths a build, test or guard reads by name, from the tree at <tree>.
# One per line: a repo path, a bare literal (matches as a path suffix), or a directory ending in "/".
readset() {
    local tree="$1" d="$2" f lit
    {
        printf '# tree %s\n' "$(git -C "$tree" rev-parse HEAD)"
        # include_str!/include_bytes! targets, resolved against the including file's directory.
        git -C "$tree" grep -nE 'include_(str|bytes)!\s*\(\s*"' -- '*.rs' \
            | sed -E 's/^([^:]+):[0-9]+:.*include_(str|bytes)!\s*\(\s*"([^"]+)".*$/\1\t\3/' \
            | while IFS=$'\t' read -r f lit; do
                realpath -m --relative-to="$tree" "$tree/$(dirname -- "$f")/$lit"
            done
        # ".md" string literals wherever code, guards or build scripts name a file.
        git -C "$tree" grep -hoE '"[^" ]+\.md"' -- '*.rs' '*.sh' '*.py' '*.yml' '*.yaml' '*.toml' 'Makefile' '*.mk' \
            | tr -d '"' | sed 's#^\./##'
        # Directory literals (read_dir, globs): any literal with a "/" naming a directory in the tree.
        git -C "$tree" grep -hoE '"[A-Za-z0-9_./-]+/[A-Za-z0-9_.-]+/?"' -- '*.rs' '*.sh' '*.py' '*.yml' '*.yaml' 'Makefile' \
            | tr -d '"' | sed 's#^\./##; s#/$##' | { grep -v '^\.\./\|^/' || true; } | sort -u | while read -r lit; do
                if [ -d "$tree/$lit" ]; then printf '%s/\n' "$lit"; fi
            done
    } | awk 'NR == 1 || !seen[$0]++' > "$d/readset.txt"
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

prop12_lines() {  # stdin: queue-inputs JSON
    jq -r '.classes | to_entries[] | "PROP12 class=\(.key) n=\(.value.q.n) q=\(.value.q.value) q_upper95=\(.value.q_upper95) q*=\(.value.prop12.q_star) \(.value.prop12.verdict)"'
}

cmd_compute() {
    local out
    out=$(compute "$1")
    printf '%s\n' "$out"
    printf '%s' "$out" | prop12_lines >&2
    case "$(printf '%s' "$out" | jq -r .verdict)" in
        GREEN) return 0 ;;
        *) printf '%s\n' "$out" | jq -r '"queue_inputs: \(.verdict)"' >&2; return 1 ;;
    esac
}

self_test() {
    local t fails=0 rc out uo
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
        # Classes: pr1 code (x); pr2 docs (d); pr3 edits a README that a test include_str!s (x, planted).
        cat > "$d/pr_meta.jsonl" <<'EOF'
{"number":1,"isCrossRepository":false,"changedFiles":1,"files":["src/lib.rs"],"labels":[]}
{"number":2,"isCrossRepository":false,"changedFiles":1,"files":["docs/guide.md"],"labels":[]}
{"number":3,"isCrossRepository":false,"changedFiles":1,"files":["crates/c/README.md"],"labels":[]}
EOF
        printf '# tree planted\ncrates/c/README.md\nCLAUDE.md\ncontracts/\n' > "$d/readset.txt"
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
    check "merged rows: pr1 x, 140 min wait, 2 entries, first try fail, 1 eject" '.derived.merged_pr_rows | map(select(.pr == 1))[0] | .class == "x" and .mq_wait_min == 140 and .entries == 2 and .first_try == "fail" and .ejects == 1' "$out"
    check "merged rows: pr2 docs, 100 min; pr3 first try pass, 0 ejects" '(.derived.merged_pr_rows | map(select(.pr == 2))[0] | .class == "d" and .mq_wait_min == 100) and (.derived.merged_pr_rows | map(select(.pr == 3))[0] | .first_try == "pass" and .ejects == 0)' "$out"
    # Theorem 1 oracle rows, spec §6.1 (recomputed independently there to 4 s.f.).
    while read -r k r q f c ff phi rho want; do
        check "Thm 1 oracle k=$k r=$r q=$q f=$f C=$c F=$ff phi=$phi rho=$rho -> $want" '. == true' \
            "$(jq -n --argjson w "$want" "$MODEL_JQ"' (t_fold('"$k; $r; $q; $f; $c; $ff; $phi; $rho"') - $w | fabs) < 0.005')"
    done <<'ROWS'
5 0 0.1 0.8 80 20 0.02 10 134.67
5 2 0.1 0.8 80 20 0.02 10 143.86
5 1 0.2 0.8 30 20 0.1 10 85.69
8 1 0.1 0.8 80 20 0.05 10 164.12
ROWS
    check "fold table k=1..8 with EM rising in k" '.derived.fold | length == 8 and (map(.EM) | . == sort)' "$out"
    # Untangle tally: #1 corrected by a later line; #2 red; #3 claim only; empty input = NO-DATA.
    printf '%s\n' '| 09:00 | untangle | #1 | pushed conflicts=4 generated=3 first-run: red |' \
        '| 10:00 | untangle | #1 | re-run conflicts=2 generated=1 first-run: green |' \
        '| 10:05 | aprender-89 | #9 | conflicts=99 first-run: red (not an untangle line) |' \
        '09:36Z | untangle | #2 | conflicts=1 generated=0 first-run=red' \
        '09:40Z | untangle | #3 | CLAIM' > "$t/untangle.md"
    uo=$(untangle_tally "$t/untangle.md")
    check "untangle: last line per PR wins, other sessions ignored" '.conflicts.total == 3 and .conflicts.generated == 1 and .conflicts.per_pr == 1.5 and .prs == 3' "$uo"
    check "untangle: 1 green of 2 first runs -> MISSED" '.first_run.rate == 0.5 and .verdict == "MISSED"' "$uo"
    : > "$t/empty.md"
    check "untangle: no lines is NO-DATA, never MET" '.verdict == "NO-DATA" and .first_run.rate == null' "$(untangle_tally "$t/empty.md")"
    check "rho_HOL blind = lambda'/60 * q' * T" '.derived.rho_hol.blind == ((.inputs.lambda_eff_per_h.value / 60 * .inputs.q_eff.value * .inputs.T.value * 10000 | round) / 10000)' "$out"
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

    check "class d: docs PR" '.derived.class_of_pr["2"] == "d"' "$out"
    check "class x: docs PR editing an include_str! target (planted)" '.derived.class_of_pr["3"] == "x"' "$out"
    check "class d with n = 1 is NOT-DECIDED, never PASS" '.classes.d.q.n == 1 and .classes.d.prop12.verdict == "NOT-DECIDED"' "$out"
    check "class a empty: n = 0, NOT-DECIDED" '.classes.a.q.n == 0 and .classes.a.prop12.verdict == "NOT-DECIDED"' "$out"
    check "q* = 30/(0.5*240) = 0.25" '.classes.d.prop12.q_star == 0.25' "$out"
    check "inputs carry mark + measuredRate (v1.1 §11.3)" '(.inputs | length) == 15 and all(.inputs[]; .mark != null) and .inputs.q_eff.measuredRate == "q_eff" and .inputs.lambda_per_h.measuredRate == "lambda"' "$out"

    # Planted class with q_c = 0.5: pr1 + pr2 as docs -> 3 entries, 1 real failure, q' = 1/3, q = 0.5.
    mk "$t/p12"
    sed -i 's#"files":\["src/lib.rs"\]#"files":["docs/other.md"]#' "$t/p12/pr_meta.jsonl"
    out=$(compute "$t/p12")
    check "planted class q_c = 0.5 prints FAIL" '.classes.d.q.value == 0.5 and .classes.d.prop12.verdict == "FAIL"' "$out"
    check "prop12 line printed" "$(printf '%s' "$out" | prop12_lines | grep -q 'class=d .* FAIL$' && echo true || echo false)" "$out"

    # 20 clean docs entries -> Wilson upper bound below q*: PASS. One of them starts while an x entry
    # (pr3, 09-23 00:00-00:30) is building, so C1 does not count it as cheap.
    mk "$t/pass"
    for i in $(seq 10 29); do
        printf '{"id":%d,"name":"CI","event":"merge_group","head_sha":"d%d","head_branch":"gh-readonly-queue/main/pr-2-x","status":"completed","conclusion":"success","created_at":"2026-09-25T%02d:00:00Z","updated_at":"2026-09-25T%02d:05:00Z","run_attempt":1}\n' \
            "$i" "$i" "$((i - 10))" "$((i - 10))" >> "$t/pass/mg_runs.jsonl"
    done
    sed -i 's/"head_sha":"d29",\(.*\)"created_at":"2026-09-25T19:00:00Z","updated_at":"2026-09-25T19:05:00Z"/"head_sha":"d29",\1"created_at":"2026-09-23T00:10:00Z","updated_at":"2026-09-23T00:15:00Z"/' "$t/pass/mg_runs.jsonl"
    out=$(compute "$t/pass")
    check "21 docs entries, 0 real failures: PASS" '.classes.d.q.n == 21 and .classes.d.prop12.verdict == "PASS"' "$out"
    check "C1: docs entry behind a building x entry is not cheap" '.classes.d.pi_eff.value < .classes.d.pi.value' "$out"

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

    # DORA: planted week with a known answer per metric, then the same files emptied (NO-DATA, never MET).
    local dd="$t/dora"; mkdir -p "$dd"
    printf '2026-01-01T00:00:00Z 2026-01-08T00:00:00Z\n' > "$dd/window.txt"
    cat > "$dd/merged.json" <<'J'
[{"number":1,"createdAt":"2026-01-02T00:00:00Z","mergedAt":"2026-01-02T01:00:00Z","headRefName":"b1","baseRefName":"main","title":"code"},
 {"number":2,"createdAt":"2026-01-02T00:00:00Z","mergedAt":"2026-01-03T00:00:00Z","headRefName":"b2","baseRefName":"main","title":"docs"},
 {"number":3,"createdAt":"2026-01-02T00:00:00Z","mergedAt":"2026-01-02T12:00:00Z","headRefName":"fold/x","baseRefName":"main","title":"fold"}]
J
    printf '%s\n' '[{"number":4,"createdAt":"2026-01-07T00:00:00Z","headRefName":"b4","baseRefName":"main","isDraft":false,"mergeable":"CONFLICTING"}]' > "$dd/open.json"
    cat > "$dd/pr_dora.jsonl" <<'J'
{"number":1,"changedFiles":1,"files":["a.rs"],"armed":"2026-01-02T00:30:00Z","queued":"2026-01-02T00:40:00Z","last_push":"2026-01-02T00:00:00Z","merge_commits":[{"date":"2026-01-02T00:10:00Z","headline":"Merge branch 'main' into b1"},{"date":"2025-12-30T00:00:00Z","headline":"merge origin/main (before the window)"}]}
{"number":2,"changedFiles":1,"files":["README.md"],"armed":null,"queued":"2026-01-02T22:00:00Z","last_push":"2026-01-02T00:00:00Z","merge_commits":[]}
{"number":3,"changedFiles":2,"files":["a.rs","b.rs"],"armed":"2026-01-02T11:00:00Z","queued":null,"last_push":"2026-01-02T10:00:00Z","merge_commits":[{"date":"2026-01-02T10:00:00Z","headline":"fold b9 into fold/x"}]}
{"number":4,"changedFiles":1,"files":["a.rs"],"armed":null,"queued":null,"last_push":"2026-01-07T00:00:00Z","merge_commits":[]}
J
    cat > "$dd/ci_runs.jsonl" <<'J'
{"id":1,"event":"pull_request","head_branch":"b1","conclusion":"success","created_at":"2026-01-02T00:00:00Z","updated_at":"2026-01-02T00:10:00Z"}
{"id":2,"event":"pull_request","head_branch":"b2","conclusion":"success","created_at":"2026-01-02T00:00:00Z","updated_at":"2026-01-02T00:03:00Z"}
{"id":3,"event":"pull_request","head_branch":"fold/x","conclusion":"failure","created_at":"2026-01-02T00:00:00Z","updated_at":"2026-01-02T00:20:00Z"}
{"id":4,"event":"pull_request","head_branch":"b1","conclusion":"cancelled","created_at":"2026-01-02T00:00:00Z","updated_at":"2026-01-02T09:00:00Z"}
{"id":5,"event":"merge_group","head_branch":"gh-readonly-queue/main/pr-1-x","conclusion":"success","created_at":"2026-01-02T00:40:00Z","updated_at":"2026-01-02T00:55:00Z"}
{"id":6,"event":"merge_group","head_branch":"gh-readonly-queue/main/pr-2-x","conclusion":"failure","created_at":"2026-01-02T22:00:00Z","updated_at":"2026-01-02T23:30:00Z"}
{"id":7,"event":"merge_group","head_branch":"gh-readonly-queue/main/pr-2-y","conclusion":"success","created_at":"2026-01-02T23:30:00Z","updated_at":"2026-01-02T23:55:00Z"}
J
    : > "$dd/main_commits.jsonl"
    for i in $(seq 1 19); do printf '{"sha":"s%s","date":"2026-01-02T00:00:00Z","subject":"feat %s","parents":1}\n' "$i" "$i" >> "$dd/main_commits.jsonl"; done
    printf '%s\n' '{"sha":"r","date":"2026-01-03T00:00:00Z","subject":"Revert \"feat 1\"","parents":1}' >> "$dd/main_commits.jsonl"
    uo=$(dora_compute "$dd")
    check "dora: lead time = first arm (else queue add) -> merged; p50 of 30,60,120 = 60 misses < 60" '.metrics.lead_time_p50_min.value == 60 and .metrics.lead_time_p50_min.n == 3 and .metrics.lead_time_p50_min.ok == false' "$uo"
    check "dora: CI p50 code 10 min, cancelled run ignored, fold branch not a code PR" '.metrics.ci_p50_code_min.value == 10 and .metrics.ci_p50_code_min.n == 1' "$uo"
    check "dora: all-.md PR is docs, 3 min misses <= 2" '.metrics.ci_p50_docs_min.value == 3 and .metrics.ci_p50_docs_min.ok == false' "$uo"
    check "dora: conflicted open PR pushed 24 h ago counts" '.metrics.conflicted_over_4h.value == 1 and .detail.conflicted_prs == [4]' "$uo"
    check "dora: 1 revert in 20 = 0.05 misses < 5%" '.metrics.change_fail_rate.value == 0.05 and .metrics.change_fail_rate.ok == false' "$uo"
    check "dora: release cycle from the fold branch run" '.metrics.release_cycle_p50_min.value == 20 and .metrics.release_cycle_p50_min.ok == true' "$uo"
    check "dora: main-in merge counted, fold merge and pre-window merge not" '.metrics.merge_commit_resolutions.value == 1 and .metrics.fold_merges.value == 1' "$uo"
    check "dora: MQ wait = first queue add -> merged; [20,120] p50 20 (lower rank), PR 3 never queued" '.metrics.mq_wait_p50_min.value == 20 and .metrics.mq_wait_p50_min.n == 2 and .metrics.mq_wait_p90_min.ok == true' "$uo"
    check "dora: MQ entry = successful merge_group runs only; [15,25] p50 15, failure excluded" '.metrics.mq_entry_p50_min.value == 15 and .metrics.mq_entry_p50_min.n == 2' "$uo"
    check "dora: open PRs counted (1 <= 10)" '.metrics.open_prs.value == 1 and .metrics.open_prs.ok == true' "$uo"
    check "dora: verdict names the misses" '.verdict | startswith("MISSED") and test("change_fail_rate")' "$uo"
    for f in ci_runs.jsonl pr_dora.jsonl main_commits.jsonl; do : > "$dd/$f"; done
    printf '[]\n' > "$dd/merged.json"; printf '[]\n' > "$dd/open.json"
    uo=$(dora_compute "$dd")
    check "dora: empty window is NO-DATA on every metric, never MET" '(.verdict | startswith("NO-DATA")) and ([.metrics[] | select(.ok == true)] | length) == 0' "$uo"

    # runner wait: 3 planted fw16 jobs (10 s, 30 s, and a re-run with started < created) + a foreign runner.
    local rw="$t/rw"; mkdir -p "$rw"
    printf '2026-01-01T00:00:00Z 2026-01-08T00:00:00Z\n' > "$rw/window.txt"
    cat > "$rw/jobs.jsonl" <<'J'
{"id":1,"runner_name":"framework16","created_at":"2026-01-02T00:00:00Z","started_at":"2026-01-02T00:00:10Z"}
{"id":2,"runner_name":"framework16-2","created_at":"2026-01-02T00:00:00Z","started_at":"2026-01-02T00:00:30Z"}
{"id":3,"runner_name":"framework16","created_at":"2026-01-02T01:00:00Z","started_at":"2026-01-02T00:50:00Z"}
{"id":4,"runner_name":"intel-clean-room-1","created_at":"2026-01-02T00:00:00Z","started_at":"2026-01-02T05:00:00Z"}
J
    uo=$(runner_wait "$rw")
    check "runner wait: fw16 only, negative re-run wait excluded, p50 of 10,30 = 10" '.all.n == 2 and .all.p50_s == 10 and .excluded_negative_waits == 1' "$uo"
    uo=$(runner_wait "$rw" '^nomatch')
    check "runner wait: no matching runner is NO-DATA" '.verdict == "NO-DATA" and .all.n == 0' "$uo"

    if [ "$fails" -eq 0 ]; then printf 'queue_inputs self-test: PASS\n'; return 0; fi
    printf 'queue_inputs self-test: %s FAIL\n' "$fails"; return 1
}

# untangle_tally <file>...: one JSON verdict over every '| untangle |' line in the files. Per PR the LAST
# conflicts=N generated=M and the LAST first-run: green|red win (a later line corrects an earlier one).
untangle_tally() {
    { grep -h -- '| untangle |' "$@" 2>/dev/null || true; } | jq -R -s '
      [split("\n")[] | select(length > 0)
       | {pr: (capture("#(?<n>[0-9]+)").n // null),
          conflicts: ((capture("conflicts=(?<v>[0-9]+)").v // null) | if . == null then null else tonumber end),
          generated: ((capture("generated=(?<v>[0-9]+)").v // null) | if . == null then null else tonumber end),
          first_run: (capture("first-run[:=] *(?<v>green|red)").v // null)}
       | select(.pr != null)] as $l
      | [$l | group_by(.pr)[]
         | {pr: (.[0].pr | tonumber),
            conflicts: ([.[] | select(.conflicts != null) | .conflicts] | last),
            generated: ([.[] | select(.generated != null) | .generated] | last),
            first_run: ([.[] | select(.first_run != null) | .first_run] | last)}] as $rows
      | [$rows[] | select(.conflicts != null)] as $c
      | [$rows[] | select(.first_run != null)] as $fr
      | ([$fr[] | select(.first_run == "green")] | length) as $g
      | {lines: ($l | length), prs: ($rows | length),
         conflicts: {n: ($c | length), total: ([$c[] | .conflicts] | add // 0),
                     per_pr: (if ($c | length) > 0 then (([$c[] | .conflicts] | add) / ($c | length) * 100 | round / 100) else null end),
                     generated: ([$c[] | .generated // 0] | add // 0)},
         first_run: {n: ($fr | length), green: $g, red: (($fr | length) - $g),
                     rate: (if ($fr | length) > 0 then ($g / ($fr | length) * 1000 | round / 1000) else null end), target: 0.9},
         rows: $rows}
      | .verdict = (if .first_run.n == 0 then "NO-DATA" elif .first_run.rate >= 0.9 then "MET" else "MISSED" end)'
}

# untangle_week [inbox-dir] [days]: inbox.md plus processed/inbox-<date>.md for the last <days> days.
untangle_week() {
    local dir="${1:-/mnt/nvme-raid0/cop-inbox}" days="${2:-7}" i dt out
    local files=("$dir/inbox.md")
    for ((i = 0; i < days; i++)); do
        dt=$(date -u -d "-$i day" +%F)
        if [ -f "$dir/processed/inbox-$dt.md" ]; then files+=("$dir/processed/inbox-$dt.md"); fi
    done
    out=$(untangle_tally "${files[@]}" | jq --arg d "$days" '. + {window_days: ($d | tonumber)}')
    printf '%s\n' "$out"
    printf '%s' "$out" | jq -e '.verdict == "MET"' >/dev/null
}

# ---- runner wait (infra#1237: fw16 wrapper before/after) --------------------------------------------
# runner_wait_fetch <raw-dir> [days]: every workflow run in the window, then every job of every attempt.
runner_wait_fetch() {
    local d="$1" days="${2:-7}" start end
    mkdir -p -- "$d/jobs"
    if [ ! -s "$d/window.txt" ]; then
        end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        start=$(date -u -d "$days days ago" +%Y-%m-%dT%H:%M:%SZ)
        printf '%s %s\n' "$start" "$end" > "$d/window.txt"
    fi
    read -r start end < "$d/window.txt"
    fetch_ci_runs "$d" "$start" "$end" '.workflow_runs[] | {id, name, event, run_attempt, created_at}' \
        actions/runs runs.jsonl || return 1
    # One file per run so a killed fetch resumes; filter=all returns the jobs of every attempt.
    jq -r '.id' "$d/runs.jsonl" | xargs -P 6 -I{} bash -c '
        f="$1/jobs/$2.jsonl"; [ -s "$f" ] && exit 0
        gh api --paginate "repos/$3/actions/runs/$2/jobs?filter=all&per_page=100" \
            -q ".jobs[] | {id, run_id, run_attempt, name, runner_name, labels, status, conclusion, created_at, started_at, completed_at}" \
            > "$f.tmp" && mv -- "$f.tmp" "$f"' _ "$d" {} "$REPO" || return 1
    local have want
    have=$(find "$d/jobs" -name '*.jsonl' | wc -l); want=$(wc -l < "$d/runs.jsonl")
    [ "$have" -eq "$want" ] || { printf 'queue_inputs: jobs for %s of %s runs\n' "$have" "$want" >&2; return 1; }
    find "$d/jobs" -name '*.jsonl' -exec cat -- {} + | jq -sc 'unique_by(.id) | .[]' > "$d/jobs.jsonl"
}

# runner_wait <raw-dir> [runner-name-regex]: queued -> started (job started_at - created_at) for jobs that ran
# on a matching runner. n = 0 is NO-DATA.
runner_wait() {
    local d="$1" re="${2:-^framework16}"
    [ -s "$d/jobs.jsonl" ] || die "missing $d/jobs.jsonl (run runner-wait-fetch)"
    jq -s --arg re "$re" --arg win "$(cat "$d/window.txt")" '
      def pct($p): sort | if length == 0 then null else .[((length - 1) * $p) | floor] end;
      def r1: if . == null then null else (. * 10 | round) / 10 end;
      def stats: {n: length, p50_s: (pct(0.5) | r1), p90_s: (pct(0.9) | r1), max_s: (max | r1)};
      [.[] | select(.runner_name != null and (.runner_name | test($re)) and .started_at != null and .created_at != null)
       | {runner: .runner_name, day: .created_at[:10], wait: ((.started_at | fromdate) - (.created_at | fromdate))}] as $all
      # A re-run attempt can carry the previous attempt'"'"'s started_at (started < created): not a wait, excluded.
      | [$all[] | select(.wait >= 0)] as $j
      | {window: $win, runner_re: $re,
         method: "jobs of every workflow run created in the window (all attempts), runner_name =~ runner_re; wait = started_at - created_at",
         all: ([$j[].wait] | stats),
         by_runner: ($j | group_by(.runner) | map({(.[0].runner): ([.[].wait] | stats)}) | add),
         excluded_negative_waits: ([$all[] | select(.wait < 0)] | length),
         first_job_day: ([$all[] | .day] | min),
         verdict: (if ($j | length) == 0 then "NO-DATA" else "MEASURED" end)}' "$d/jobs.jsonl"
}

# ---- DORA weekly (operator ask via the cop, 2026-09-27) -------------------------------------------
# dora_fetch <raw-dir> [days]: merged + open PRs, their arm events / files / branch commits, ci.yml runs
# and main's commits over the window. Read-only; resumable like fetch.
dora_fetch() {
    local d="$1" days="${2:-7}" start end
    mkdir -p -- "$d"
    if [ ! -s "$d/window.txt" ]; then
        end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        start=$(date -u -d "$days days ago" +%Y-%m-%dT%H:%M:%SZ)
        printf '%s %s\n' "$start" "$end" > "$d/window.txt"
    fi
    read -r start end < "$d/window.txt"
    local runq='.workflow_runs[] | {id,name,event,head_sha,head_branch,status,conclusion,created_at,updated_at,run_attempt}'
    fetch_ci_runs "$d" "$start" "$end" "$runq" || return 1
    ghj "$d/merged.json" pr list -R "$REPO" --state merged --search "merged:$start..$end" --limit 1000 \
        --json number,createdAt,mergedAt,headRefName,baseRefName,title || return 1
    [ "$(jq length "$d/merged.json")" -lt 1000 ] || die "merged PR list hit the 1000-row cap"
    ghj "$d/open.json" pr list -R "$REPO" --state open --limit 500 \
        --json number,createdAt,headRefName,baseRefName,isDraft,mergeable || return 1
    ghj "$d/main_commits.jsonl" api --paginate "repos/$REPO/commits?sha=main&since=$start&until=$end&per_page=100" \
        -q '.[] | {sha, date: .commit.committer.date, subject: (.commit.message | split("\n")[0]), parents: (.parents | length)}' \
        || return 1
    if [ ! -s "$d/pr_dora.jsonl" ]; then
        local batch q n
        : > "$d/pr_dora.part"
        while mapfile -t -n 25 batch && [ "${#batch[@]}" -gt 0 ]; do
            q='query{repository(owner:"paiml",name:"aprender"){'
            for n in "${batch[@]}"; do
                q+="p$n:pullRequest(number:$n){number changedFiles files(first:100){nodes{path}}
                    timelineItems(first:50,itemTypes:[AUTO_MERGE_ENABLED_EVENT,ADDED_TO_MERGE_QUEUE_EVENT]){
                      nodes{__typename ... on AutoMergeEnabledEvent{createdAt} ... on AddedToMergeQueueEvent{createdAt}}}
                    commits(last:100){nodes{commit{committedDate messageHeadline parents{totalCount}}}}}"
            done
            q+='}}'
            gh api graphql -f query="$q" -q '.data.repository[] | select(. != null)
                | {number, changedFiles, files: [.files.nodes[].path],
                   armed: ([.timelineItems.nodes[] | select(.__typename == "AutoMergeEnabledEvent") | .createdAt] | min),
                   queued: ([.timelineItems.nodes[] | select(.__typename == "AddedToMergeQueueEvent") | .createdAt] | min),
                   last_push: ([.commits.nodes[].commit.committedDate] | max),
                   merge_commits: [.commits.nodes[].commit | select(.parents.totalCount > 1)
                                   | {date: .committedDate, headline: .messageHeadline}]}' \
                >> "$d/pr_dora.part" || return 1
            sleep 1
        done < <(jq -r '.[].number' "$d/merged.json" "$d/open.json" | sort -un)
        jq -c . "$d/pr_dora.part" > "$d/pr_dora.jsonl"
        rm -f -- "${d:?}/pr_dora.part"
    fi
}

# dora_compute <raw-dir>: the weekly DORA table as JSON. Every metric carries {value, n, target, ok};
# a metric with n = 0 is ok = null (NO-DATA), never a pass. rc 1 when any metric misses or has no data.
dora_compute() {
    local d="$1" f
    for f in window.txt ci_runs.jsonl merged.json open.json main_commits.jsonl pr_dora.jsonl; do
        [ -f "$d/$f" ] || die "missing $d/$f"
    done
    local start end
    read -r start end < "$d/window.txt"
    jq -n --arg start "$start" --arg end "$end" --arg relre "$RELEASE_BRANCH_RE" \
        --slurpfile ci <(cat "$d/ci_runs.jsonl") --slurpfile merged "$d/merged.json" --slurpfile open "$d/open.json" \
        --slurpfile main <(cat "$d/main_commits.jsonl") --slurpfile pd <(cat "$d/pr_dora.jsonl") -f /dev/stdin <<'JQ'
def mins($a; $b): (($b | fromdate) - ($a | fromdate)) / 60;
def pct($p): sort | if length == 0 then null else .[((length - 1) * $p) | floor] end;
def r2: if . == null then null else (. * 100 | round) / 100 end;
def m($v; $n; $target; $ok; $method): {value: ($v | r2), n: $n, target: $target, ok: (if $n == 0 or $v == null then null else $ok end), method: $method};
def inwin: . >= $start and . <= $end;
$merged[0] as $mg | $open[0] as $op
| (reduce $pd[] as $p ({}; . + {($p.number | tostring): $p})) as $by
| (reduce ($mg + $op)[] as $p ({}; . + {($p.headRefName): $p.number})) as $pr_of_branch
| def docs($n): ($by[$n | tostring]) as $p
    | $p != null and $p.changedFiles > 0 and $p.changedFiles <= 100 and all($p.files[]; endswith(".md"));
# Lead time: first auto-merge arm (else first queue add) -> merged, PRs into main.
  [$mg[] | select(.baseRefName == "main") | . as $p | $by[$p.number | tostring] as $x
   | ($x.armed // $x.queued) as $arm | select($arm != null) | mins($arm; $p.mergedAt)] as $lead
| [$mg[] | select(.baseRefName == "main") | select(($by[.number | tostring] | (.armed // .queued)) == null)] as $unarmed
# CI p50 per class: completed pull_request ci.yml runs of the window's PRs, created -> updated.
| [$ci[] | select(.event == "pull_request" and (.conclusion == "success" or .conclusion == "failure")
                  and (.head_branch | test($relre) | not))
   | $pr_of_branch[.head_branch] as $n | select($n != null)
   | {docs: docs($n), dur: mins(.created_at; .updated_at)}] as $ciruns
| [$ciruns[] | select(.docs | not) | .dur] as $ci_code
| [$ciruns[] | select(.docs) | .dur] as $ci_docs
# PR age: merged PRs created -> merged; open non-draft PRs created -> window end.
| [$mg[] | mins(.createdAt; .mergedAt) / 60] as $age_merged
| [$op[] | select(.isDraft | not) | mins(.createdAt; $end) / 60] as $age_open
# Merge-queue wait: first queue add -> merged, PRs into main (includes re-entries after an ejection).
| [$mg[] | select(.baseRefName == "main") | . as $p | $by[$p.number | tostring].queued as $q
   | select($q != null) | mins($q; $p.mergedAt)] as $mqwait
# Merge-queue entry build: successful merge_group ci.yml runs, created -> updated (T in queue-inputs-v1).
| [$ci[] | select(.event == "merge_group" and .conclusion == "success") | mins(.created_at; .updated_at)] as $mqentry
# Conflicted > 4 h: open PRs GitHub reports CONFLICTING whose last push is > 4 h before window end.
| [$op[] | select(.mergeable == "CONFLICTING") | . as $p | $by[$p.number | tostring].last_push as $lp
   | select($lp != null and mins($lp; $end) > 240) | $p.number] as $conflicted
| [$op[] | select(.mergeable == "UNKNOWN")] as $unknown_mergeable
# Change-fail: reverts landed on main / commits landed on main.
| [$main[] | select(.subject | test("^(Revert|revert)[ :(\"]"))] as $reverts
# Release cycle: ci.yml runs on release branches, created -> updated.
| [$ci[] | select(.event == "pull_request" and (.head_branch | test($relre))
                  and (.conclusion == "success" or .conclusion == "failure")) | mins(.created_at; .updated_at)] as $rel
# Merge-commit resolutions: two-parent commits on PR branches, committed in the window.
# A two-parent commit that pulls main in is a resolution; one that pulls a PR branch into a fold is a fold.
| [$pd[] | .number as $n | .merge_commits[] | select(.date | inwin) | . + {pr: $n}] as $mc_all
| [$mc_all[] | select(.headline | test("\\bmain\\b"))] as $mc
| {window: "\($start)/\($end)",
   metrics: {
     lead_time_p50_min: m($lead | pct(0.5); ($lead | length); "< 60"; (($lead | pct(0.5)) < 60);
                          "first auto-merge arm (else first queue add) -> mergedAt, PRs into main"),
     lead_time_p90_min: m($lead | pct(0.9); ($lead | length); "report"; true; "same, p90"),
     ci_p50_code_min: m($ci_code | pct(0.5); ($ci_code | length); "<= 10"; (($ci_code | pct(0.5)) <= 10);
                        "completed pull_request ci.yml runs of non-docs PRs, created -> updated"),
     ci_p50_docs_min: m($ci_docs | pct(0.5); ($ci_docs | length); "<= 2"; (($ci_docs | pct(0.5)) <= 2);
                        "same, PRs whose every file is .md"),
     pr_age_p90_h_merged: m($age_merged | pct(0.9); ($age_merged | length); "< 24"; (($age_merged | pct(0.9)) < 24);
                            "merged PRs, createdAt -> mergedAt"),
     pr_age_p90_h_open: m($age_open | pct(0.9); ($age_open | length); "< 24"; (($age_open | pct(0.9)) < 24);
                          "open non-draft PRs at window end"),
     mq_wait_p50_min: m($mqwait | pct(0.5); ($mqwait | length); "report"; true;
                        "first AddedToMergeQueue -> mergedAt, merged PRs into main"),
     mq_wait_p90_min: m($mqwait | pct(0.9); ($mqwait | length); "<= 120"; (($mqwait | pct(0.9)) <= 120);
                        "same, p90 (operator stop rule: p90 > 120 m)"),
     mq_entry_p50_min: m($mqentry | pct(0.5); ($mqentry | length); "report"; true;
                         "successful merge_group ci.yml runs, created -> updated"),
     mq_entry_p90_min: m($mqentry | pct(0.9); ($mqentry | length); "report"; true; "same, p90"),
     open_prs: m($op | length; ($op | length); "<= 10"; (($op | length) <= 10);
                 "open PRs at fetch time, drafts included (operator cap 10); \([$op[] | select(.isDraft)] | length) draft"),
     conflicted_over_4h: m($conflicted | length; ($op | length); "0"; (($conflicted | length) == 0);
                           "open PRs mergeable=CONFLICTING with last push > 4 h ago (GitHub: \($unknown_mergeable | length) UNKNOWN)"),
     change_fail_rate: m(if ($main | length) > 0 then ($reverts | length) / ($main | length) else null end; ($main | length);
                         "< 0.05"; (($main | length) > 0 and (($reverts | length) / ($main | length)) < 0.05); "Revert commits / commits on main"),
     release_cycle_p50_min: m($rel | pct(0.5); ($rel | length); "<= 30"; (($rel | pct(0.5)) <= 30);
                              "ci.yml pull_request runs on release branches, created -> updated"),
     merge_commit_resolutions: m($mc | length; ($pd | length); "<= 2 (-> 0)"; (($mc | length) <= 2);
                                 "two-parent commits on PR branches, committed in the window, whose headline pulls in main"),
     fold_merges: m(($mc_all | length) - ($mc | length); ($pd | length); "report"; true;
                    "other two-parent commits on PR branches (folds of PR branches into a batch)")},
   detail: {conflicted_prs: $conflicted, reverts: [$reverts[] | .subject], merge_commits: $mc, merge_resolutions_by_pr: ($mc | group_by(.pr) | map({pr: .[0].pr, n: length}) | sort_by(-.n)),
            unarmed_merged_into_main: ($unarmed | length), ci_runs: {code: ($ci_code | length), docs: ($ci_docs | length)}}}
| .missed = [.metrics | to_entries[] | select(.value.ok == false) | .key]
| .no_data = [.metrics | to_entries[] | select(.value.ok == null) | .key]
| .verdict = (if (.missed | length) > 0 then "MISSED \(.missed | join(","))"
              elif (.no_data | length) > 0 then "NO-DATA \(.no_data | join(","))" else "MET" end)
JQ
}

dora_line() {  # one inbox-sized line from dora_compute JSON on stdin
    jq -r '.metrics as $m | "DORA 7d: lead p50 \($m.lead_time_p50_min.value)m (n\($m.lead_time_p50_min.n)) | CI p50 code \($m.ci_p50_code_min.value)m docs \($m.ci_p50_docs_min.value)m | MQ wait p50 \($m.mq_wait_p50_min.value)m p90 \($m.mq_wait_p90_min.value)m (n\($m.mq_wait_p50_min.n)) | MQ entry p50 \($m.mq_entry_p50_min.value)m p90 \($m.mq_entry_p90_min.value)m (n\($m.mq_entry_p50_min.n)) | PR age p90 merged \($m.pr_age_p90_h_merged.value)h open \($m.pr_age_p90_h_open.value)h | open PRs \($m.open_prs.value) | conflicted>4h \($m.conflicted_over_4h.value) | change-fail \($m.change_fail_rate.value) | release p50 \($m.release_cycle_p50_min.value)m | main-merges \($m.merge_commit_resolutions.value) (+\($m.fold_merges.value) fold) -> \(.verdict)"'
}

case "${1:-}" in
    fetch) [ $# -ge 2 ] || die "usage: fetch <raw-dir> [days]"; fetch "$2" "${3:-7}" ;;
    compute) [ $# -eq 2 ] || die "usage: compute <raw-dir>"; cmd_compute "$2" ;;
    ident) [ $# -eq 2 ] || die "usage: ident <raw-dir>"; ident "$2" ;;
    readset) [ $# -eq 3 ] || die "usage: readset <tree> <raw-dir>"; readset "$2" "$3" ;;
    prop12) [ $# -eq 2 ] || die "usage: prop12 <queue-inputs.json>"; prop12_lines < "$2" ;;
    untangle) shift; untangle_week "$@" ;;
    dora-fetch) [ $# -ge 2 ] || die "usage: dora-fetch <raw-dir> [days]"; dora_fetch "$2" "${3:-7}" ;;
    runner-wait-fetch) [ $# -ge 2 ] || die "usage: runner-wait-fetch <raw-dir> [days]"; runner_wait_fetch "$2" "${3:-7}" ;;
    runner-wait) [ $# -ge 2 ] || die "usage: runner-wait <raw-dir> [runner-regex]"; runner_wait "$2" "${3:-^framework16}" ;;
    dora-line) dora_line ;;
    dora) [ $# -eq 2 ] || die "usage: dora <raw-dir>"; out=$(dora_compute "$2"); printf '%s\n' "$out"
          printf '%s' "$out" | dora_line >&2; printf '%s' "$out" | jq -e '.verdict == "MET"' >/dev/null ;;
    self-test) self_test ;;
    *) die "usage: queue_inputs.sh fetch <raw-dir> [days] | compute <raw-dir> | self-test" ;;
esac
