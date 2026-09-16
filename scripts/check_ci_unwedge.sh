#!/usr/bin/env bash
# check_ci_unwedge.sh -- find CI runs wedged on a parked aggregator, and free them.
#
# THE DEFECT (#3229)
# ------------------
# A push cancels the in-flight run. Its jobs go `cancelled`. The aggregator in the
# SHA-pinned reusable workflow carries `if: always()`
# (paiml/.github@main sovereign-ci.yml:1474), so it is SCHEDULED rather than
# skipped -- it enters `queued`, the run is cancelling so no runner is ever
# assigned, and it sits there. `timeout-minutes: 5` cannot save it: a job timeout
# starts when the job STARTS, and this one never does. That is why nothing times
# out.
#
# While it sits `queued` the RUN is `queued`, the run holds the branch's
# concurrency group, and the next run is created with ZERO jobs. Every later push
# stacks another empty run behind the same zombie. The branch shows a run "in
# progress", the required checks are never created, and the PR sits BLOCKED with
# no red anywhere to look at.
#
# Cost, measured 2026-09-13/14: main merged NOTHING for 8.5 hours with three
# groups wedged this way, and it recurred on #3205 at 04:28Z the same night.
#
# `gh run cancel` does NOT clear it -- the run stays `queued` through the cancel
# request. Only POST .../force-cancel works. After it, the blocked run starts
# within ~45 s (measured: run 34806202374 went 0 -> 14 jobs).
#
# THIS IS CONTAINMENT, NOT THE FIX. The fix is upstream, and the fix this issue
# originally proposed -- `if: always()` -- is already in place and is nearer the
# cause. `if: ${{ !cancelled() }}` is the candidate; until it is tested and the
# pin is bumped, something has to free the queue.
#
# THE PREDICATE NEEDS BOTH SIGNALS
# --------------------------------
# `jobs == 0` alone is NOT the signal. Run 34803859272 read jobs=0 for a moment
# while fourteen real jobs were coming; acting on that cancels healthy runs. A run
# is wedged when:
#
#   (a) at least one job was CANCELLED -- the fingerprint of a supersede, AND
#   (b) at least one job is still pending, AND
#   (c) EVERY pending job is an aggregator -- one that only reads needs.*.result
#       and can never start once something it needs stopped.
#
# (a) IS CANCELLED, NOT "FAILED OR CANCELLED". The first draft of this file said
# "failure or cancelled" and, run live in dry-run against this repo at 04:49Z,
# proposed force-cancelling run 34805623711 -- #3060's merge group at QUEUE
# POSITION 1. That run had `guard-tree` FAILED, everything else finished, and
# `ci / gate` queued waiting for a runner on a saturated fleet. It was perfectly
# alive: its gate would get a runner and report the failure. Cancelling it would
# have destroyed the head of the queue's verdict.
#
# A failure is an ANSWER; the gate still runs and reports it. A CANCELLATION is
# what a supersede leaves behind, and it is the only state in which the gate is
# scheduled into a run that can never assign it a runner. H6 is that run's real
# job list, committed, so the distinction cannot be lost again.
#
# (c) is precise, not a count. "<= 1 pending job" both MISSES a group wedged on
# two aggregators (`ci / gate` AND `gate`) and FIRES on a run genuinely down to
# its last real job. Both are rows in the table below.
#
# THE SECOND DEFECT: A MERGE GROUP WHOSE REF IS GONE (measured 2026-09-14)
# ------------------------------------------------------------------------
# Every merge changes the base of every group behind it, so GitHub deletes those
# queue refs and creates new ones. The CI runs on the OLD refs are not cancelled.
# They keep building, and they draw runners to compute a verdict on a ref that no
# longer exists -- nothing can ever read it.
#
# Measured at 04:43Z with a 9-deep queue: five non-completed merge_group CI runs,
# and TWO of them (`pr-3006-0c740b04`, `pr-3056-e2ef10`) had 404 refs while
# holding gx10-pool3, intel-clean-room-8 and intel-clean-room-14, with more jobs
# queued behind them. Forty per cent of the merge-group load, computing nothing.
#
# This is ORTHOGONAL to the wedge above and not redundant with it: run
# 34805561949 still had `workspace-test` pending, so the wedge predicate
# correctly answered HEALTHY. It was dead for an entirely different reason.
#
# VACUITY IS THE WHOLE RISK HERE. If the branch -> ref derivation breaks, EVERY
# run reads 404 and the janitor cancels the entire queue. So the decision is
# gated on corroboration: at least one merge_group candidate must resolve LIVE
# before any DEAD verdict is acted on. Zero live and one or more dead means the
# LOOKUP is broken, not the fleet -- refuse, and say so.
#
# THE THIRD DEFECT: A SUPERSEDED HEAD (APR-RELEASE-001, revised 2026-09-16,
# "P0 Unwedge", rule 1)
# ---------------------------------------------------------------------------
# A push creates a new run at the new head and leaves the OLD run building at
# a commit that no longer merges. Any `pull_request` run whose `head_sha` !=
# the PR's CURRENT head is cancelled -- this is the rule that would have freed
# `ci-3354`. NEVER `merge_group`, under any condition: the merge queue owns
# those runs and cancelling one ejects a PR mid-build.
#
# This rule is independent of the stalled-run rule above: a stale head_sha is
# cancellable whether the run has 0 jobs or 18, and whether every pool is idle
# or saturated. It needs neither a job-count gate nor the capacity input --
# the head mismatch IS the evidence, read straight off the run and the PR, not
# inferred from age or a bare `queued`. That inference is exactly what went
# wrong hand-cancelling run 35078448806 (`queued`, 16 of 18 jobs already done):
# the cancel manufactured the `gate` failure it then reported. See
# superseded_verdict, below, for the predicate this rule is not allowed to be.
#
# Usage
#   bash scripts/check_ci_unwedge.sh --self-test        the committed case table
#   bash scripts/check_ci_unwedge.sh --scan [--dry-run] [--limit N] [--repo O/R]
#   bash scripts/check_ci_unwedge.sh --verdict FILE      one jobs payload -> verdict
#   bash scripts/check_ci_unwedge.sh --help
set -uo pipefail

REPO_DEFAULT="paiml/aprender"
AGGREGATOR_RE='^(ci / )?gate$'
CASES_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )/lib/ci_unwedge_cases"

usage() {
    cat <<'USAGE'
check_ci_unwedge.sh -- free CI runs wedged on a parked aggregator (#3229).

  --self-test           run the committed case table (fixtures, no network)
  --scan                look at recent runs and force-cancel the wedged ones
  --dry-run             with --scan: report, change nothing
  --limit N             with --scan: how many recent runs to consider (default 15)
  --repo OWNER/NAME     with --scan: the repository (default paiml/aprender)
  --capacity FILE       with --scan: a HOST-SIDE capacity reading (one row per
                        host: labels, listeners, workers). Without it the
                        stalled-run rule REFUSES -- it never cancels blind, and it
                        never asks GitHub for the runner list
  --emit-capacity-row HOST LABELS_CSV
                        print this host's row from /proc, for the caller to collect
  --verdict FILE        classify one `actions/runs/<id>/jobs` payload
  --help
USAGE
}

# unwedge_verdict FILE -> prints "WEDGED <reason>" or "HEALTHY <reason>"
# Pure: a jobs payload in, a verdict out. No network, so the table below can pin
# both polarities against committed fixtures.
unwedge_verdict() {
    local f="${1:-}" stopped pending nonagg
    if [ -z "$f" ] || [ ! -r "$f" ]; then
        printf 'ENV unreadable jobs payload: %s\n' "${f:-<none>}"
        return 2
    fi
    stopped=$(jq '[.jobs[]? | select(.conclusion == "cancelled")] | length' "$f" 2>/dev/null) || stopped=""
    pending=$(jq '[.jobs[]? | select(.status == "queued" or .status == "in_progress")] | length' "$f" 2>/dev/null) || pending=""
    if [ -z "$stopped" ] || [ -z "$pending" ]; then
        printf 'ENV jobs payload is not valid JSON: %s\n' "$f"
        return 2
    fi
    nonagg=$(jq -r --arg re "$AGGREGATOR_RE" \
        '[.jobs[]? | select(.status == "queued" or .status == "in_progress")
                   | select(.name | test($re) | not)] | length' "$f" 2>/dev/null) || nonagg=1

    if [ "$stopped" -eq 0 ]; then
        printf 'HEALTHY nothing cancelled: %s pending; a failure is an answer, the gate still reports it\n' "$pending"; return 0
    fi
    if [ "$pending" -eq 0 ]; then
        printf 'HEALTHY every job is terminal\n'; return 0
    fi
    if [ "$nonagg" -gt 0 ]; then
        printf 'HEALTHY %s pending job(s), %s of them real work that can still finish\n' "$pending" "$nonagg"; return 0
    fi
    printf 'WEDGED %s cancelled, %s pending, all aggregators\n' "$stopped" "$pending"
    return 0
}

# queue_ref_path BRANCH -> the `git/ref/heads/...` path to query, or empty when
# BRANCH is not a merge-queue ref. Pure: string in, string out, so the table can
# pin it without a network call.
queue_ref_path() {
    local br="${1:-}"
    case "$br" in
        gh-readonly-queue/*) ;;
        *) return 1 ;;
    esac
    # The API wants the ref path with its slashes percent-encoded.
    printf 'heads/%s\n' "${br//\//%2F}"
}

# deadref_decision N_LIVE N_DEAD -> ACT | REFUSE | NOTHING
# The corroboration gate. See VACUITY IS THE WHOLE RISK HERE, above.
deadref_decision() {
    local live="${1:-0}" dead="${2:-0}"
    if [ "$dead" -eq 0 ]; then printf 'NOTHING\n'; return 0; fi
    if [ "$live" -eq 0 ]; then printf 'REFUSE\n'; return 0; fi
    printf 'ACT\n'
}

# ---- #3292 RULE 1: a superseded head is cancelled outright -------------------
# APR-RELEASE-001 (revised 2026-09-16), "P0 Unwedge", rule 1: any `pull_request`
# run whose `head_sha` != the PR's CURRENT head is cancelled. This is the rule
# that would have freed `ci-3354` -- a push creates a NEW run at the new head
# and leaves the OLD run building at a commit nobody can merge; nothing about
# that old run's job count or the fleet's capacity makes its verdict useful.
#
# NEVER `merge_group`, UNDER ANY CONDITION. The merge queue owns those runs;
# cancelling one ejects a PR mid-build. This is a hard exclusion on event type,
# checked FIRST, before head_sha is even compared.
#
# THE COMPARISON IS THE EVIDENCE. A stale head_sha is read straight off the run
# and the PR, not inferred from age or `queued` (see rule 2's own history,
# directly below: the hand-cancel of run 35078448806 read `queued` alone with
# 16 of 18 jobs already done, and its cancel MANUFACTURED the `gate` failure it
# then reported). A head mismatch is positive evidence a newer run already
# exists; nothing here needs a guess.
#
# JOBS_FILE and CAPACITY_FILE are accepted and IGNORED -- on purpose. This rule
# is independent of rule 2's job-count and capacity gate: a superseded run is
# cancellable whether it has 0 jobs or 18, and whether every pool is idle or
# saturated. Passing rule 2's own fixtures here and getting the identical
# verdict (self-test rows R1D/R1E) is that independence, proved, not asserted.
superseded_verdict() {
    local event="${1:-}" run_head="${2:-}" pr_head="${3:-}"
    case "$event" in
        merge_group)
            printf 'LEAVE a merge_group run is never cancelled by this rule\n'; return 0 ;;
    esac
    if [ -z "$run_head" ] || [ -z "$pr_head" ]; then
        printf 'REFUSE head sha is unknown -- never cancel on a guess\n'; return 0
    fi
    if [ "$run_head" != "$pr_head" ]; then
        printf 'CANCEL run head %s != PR head %s -- superseded\n' "$run_head" "$pr_head"; return 0
    fi
    printf 'LEAVE run head matches the PR head\n'
}

# ---- #3292: a run may not hold a concurrency group while capacity sits idle ----
# THE CONTRACT. No run holds its concurrency group for longer than one sweeper
# period while capacity to serve its pending jobs sits IDLE.
#
# THE TRIGGER IS NOT `queued`, AND IT IS NOT A JOB COUNT. Both halves are
# measured in #3358. Run 35078448806 sat `queued` on #3354 for two hours with
# SIXTEEN of its eighteen jobs already run -- jobs trickling onto congested pools
# from 09:27 to 11:00. That is a QUEUE. The hand-cancel that read `queued` as a
# verdict killed the two jobs still going and produced a `gate` that failed in
# four seconds; the red was manufactured by the cancel. Meanwhile the two runs
# stacked BEHIND it read jobs == 0 -- they were the victims, and cancelling a
# victim frees nothing because it is not the run holding the group.
#
# So the rule fires only on evidence of a WEDGE: no dispatch progress AND idle
# capacity that could have served it. Capacity is a required input; where it
# cannot be measured the rule REFUSES (see pool_idle).
STALL_AGE_MIN=30       # how long a run may hold its group before it is a candidate
STALL_WINDOW_MIN=30    # a job started inside this window is dispatch PROGRESS

# pool_idle CAPACITY_JSON LABELS_CSV -> how many listeners could take the work
# right now, summed over the hosts whose listeners carry EVERY requested label.
#
# CAPACITY IS A HOST-SIDE READING, NOT A GITHUB ANSWER. `GET
# /orgs/{org}/actions/runners` is not available to this work; idle capacity on this
# fleet is measured on the hosts themselves by the ONE sanctioned oracle,
# `fleet-bin.sh verify-effective-paths` (live listeners, by /proc comm), plus
# `pgrep -c -x Runner.Worker` for busy -- the reading `verify-fleet-bin` reports. The payload is
# therefore an INPUT (`--capacity FILE` / UNWEDGE_CAPACITY_JSON), one row per host:
#
#   {"hosts":[{"host":"intel","labels":["self-hosted","Linux","X64","clean-room"],
#              "listeners":16,"workers":15}, ...]}
#
# EMPTY OUTPUT MEANS UNKNOWN, AND UNKNOWN IS NOT ZERO. Zero is a measurement ("the
# pool is full"); empty is the absence of one, and the two lead to opposite
# actions -- so they are different values, never both `0`.
#
# The label test is containment, not equality: gx10's GPU listeners were free
# throughout the incident and could not have taken one clean-room job, because they
# do not carry `clean-room`. An idle box that cannot serve the labels is not
# capacity (row C2). A host reading more workers than listeners contributes 0, never
# a negative that would cancel out another host's real capacity (row C3).
pool_idle() {
    local f="${1:-}" want="${2:-}" n
    if [ -z "$f" ] || [ ! -r "$f" ] || [ -z "$want" ]; then printf '\n'; return 2; fi
    n=$(jq -r --arg want "$want" '
            ($want | split(",") | map(select(length > 0))) as $need
            | [ .hosts[]?
                | select( ($need - (.labels // [])) | length == 0 )
                | (((.listeners // 0) - (.workers // 0)) | if . > 0 then . else 0 end) ]
            | add // 0' "$f" 2>/dev/null) || n=""
    case "$n" in ''|*[!0-9]*) printf '\n'; return 2 ;; esac
    printf '%s\n' "$n"
}

# capacity_row HOST LABELS_CSV LISTENERS WORKERS -> one row of the payload above.
# The acquisition is the CALLER's business -- this only fixes the shape, so the
# host-side one-liner and the predicate cannot drift:
#
#   listeners=$(pgrep -fc "[R]unner.Listener"); workers=$(pgrep -fc "[R]unner.Worker")
#   bash check_ci_unwedge.sh --emit-capacity-row "$(hostname -s)" 'self-hosted,Linux,clean-room'
capacity_row() {
    local h="${1:-}" labels="${2:-}" listeners="${3:-}" workers="${4:-}"
    case "$listeners$workers" in ''|*[!0-9]*) printf '\n'; return 2 ;; esac
    jq -n -c --arg host "$h" --arg labels "$labels" \
            --argjson listeners "$listeners" --argjson workers "$workers" \
        '{host: $host, labels: ($labels | split(",") | map(select(length > 0))),
          listeners: $listeners, workers: $workers}'
}

# emit_capacity_row HOST LABELS_CSV -> this host's row, read from /proc.
# The bracket in the pattern keeps pgrep from matching its own command line.
emit_capacity_row() {
    # THE ORACLE, NOT A REIMPLEMENTATION OF IT (APR-RELEASE-001 §5: "fleet-bin.sh
    # is the only runner oracle. Do not reimplement it"). Live listeners come from
    # `fleet-bin.sh verify-effective-paths`, which discovers by /proc comm ==
    # Runner.Listener; the `pgrep -fc "[R]unner.Listener"` this used to run matches
    # COMMAND LINES and counted 17 on a box with 16 listeners -- the 17th was the
    # ssh command that carried the pgrep (fleet-bin.sh, "DISCOVERY IS BY comm, NOT
    # pgrep"). Busy is `pgrep -c -x Runner.Worker`: -x is an exact comm match.
    # No oracle => EMPTY output, rc 2 -- unknown, never 0 (zero is "the pool is
    # full" and would license a cancel).
    local h="${1:-}" labels="${2:-}" oracle="${FLEET_BIN:-fleet-bin.sh}" listeners workers
    command -v "$oracle" > /dev/null 2>&1 || { printf '\n'; return 2; }
    listeners=$( "$oracle" verify-effective-paths 2>/dev/null \
                 | sed -n 's/.*effective PATH: \([0-9][0-9]*\) live listener(s).*/\1/p' | head -1 )
    case "$listeners" in ''|*[!0-9]*) printf '\n'; return 2 ;; esac
    workers=$(pgrep -c -x Runner.Worker 2>/dev/null) || workers="${workers:-0}"
    case "$workers" in ''|*[!0-9]*) workers=0 ;; esac
    capacity_row "$h" "$labels" "$listeners" "$workers"
}

# dispatch_state JOBS_JSON NOW_ISO WINDOW_MIN -> PROGRESSING | STALLED
#
# PROGRESSING is the safe answer and every uncertain case returns it. A run with
# NOTHING pending is progressing (it is finishing), and a run with ZERO jobs is
# progressing (H4: zero jobs is a slow start, or it is a victim queued behind the
# real holder -- never a verdict on its own).
dispatch_state() {
    local f="${1:-}" now="${2:-}" win="${3:-30}" now_s pending recent
    if [ -z "$f" ] || [ ! -r "$f" ]; then printf 'PROGRESSING\n'; return 2; fi
    now_s=$(date -u -d "$now" +%s 2>/dev/null) || now_s=""  # bashrs disable-line=DET002
    case "$now_s" in ''|*[!0-9]*) printf 'PROGRESSING\n'; return 2 ;; esac
    pending=$(jq '[.jobs[]? | select(.status == "queued" or .status == "in_progress")] | length' "$f" 2>/dev/null) || pending=""
    case "$pending" in ''|*[!0-9]*) printf 'PROGRESSING\n'; return 2 ;; esac
    if [ "$pending" -eq 0 ]; then printf 'PROGRESSING\n'; return 0; fi
    recent=$(jq -r --argjson now "$now_s" --argjson win "$win" '
            [ .jobs[]? | select(.started_at != null)
              | (.started_at | fromdateiso8601)
              | select(($now - .) <= ($win * 60)) ] | length' "$f" 2>/dev/null) || recent=""
    case "$recent" in ''|*[!0-9]*) printf 'PROGRESSING\n'; return 2 ;; esac
    if [ "$recent" -gt 0 ]; then printf 'PROGRESSING\n'; else printf 'STALLED\n'; fi
}

# pending_labels JOBS_JSON -> the union of the labels the still-pending jobs ask
# for. That union, not the run, is the pool whose idleness decides the verdict.
pending_labels() {
    local f="${1:-}"
    if [ -z "$f" ] || [ ! -r "$f" ]; then printf '\n'; return 2; fi
    jq -r '[ .jobs[]? | select(.status == "queued" or .status == "in_progress") | .labels[]? ]
           | unique | join(",")' "$f" 2>/dev/null || printf '\n'
}

# stall_candidate_branch BRANCH -> 0 when this rule may judge the branch.
# A merge-queue run is NEVER a candidate: cancelling a queue build throws away the
# verdict the queue is waiting on. Dead queue refs are the dead-ref pass's job,
# and that pass has its own corroboration gate.
stall_candidate_branch() {
    case "${1:-}" in
        gh-readonly-queue/*) return 1 ;;
        '') return 1 ;;
        *) return 0 ;;
    esac
}

# stall_verdict AGE_MIN IDLE DISPATCH -> CANCEL | UNTOUCHED | REFUSE, plus why.
# Pure: three scalars in, a verdict out, so the case table pins every polarity
# without a network call.
stall_verdict() {
    local age="${1:-}" idle="${2:-}" disp="${3:-}"
    case "$age" in ''|*[!0-9]*) printf 'REFUSE the run age is unknown\n'; return 0 ;; esac
    case "$idle" in ''|*[!0-9]*)
        printf 'REFUSE idle capacity is unknown -- evidence of a wedge is required, and absence of evidence is not it\n'
        return 0 ;;
    esac
    if [ "$disp" != "STALLED" ]; then
        printf 'UNTOUCHED jobs are still being dispatched (%s)\n' "$disp"; return 0
    fi
    if [ "$age" -lt "$STALL_AGE_MIN" ]; then
        printf 'UNTOUCHED %s min held, under the %s min floor\n' "$age" "$STALL_AGE_MIN"; return 0
    fi
    if [ "$idle" -eq 0 ]; then
        printf 'UNTOUCHED %s min held but 0 idle runners serve those labels -- a queue, not a wedge\n' "$age"; return 0
    fi
    printf 'CANCEL %s min held, no dispatch, %s idle runner(s) could have served it\n' "$age" "$idle"
}

self_test() {
    local fails=0 rows=0 n want got
    if [ ! -d "$CASES_DIR" ]; then
        printf 'FAIL (vacuity): no fixtures at %s\n' "$CASES_DIR"; return 1
    fi
    n=$(find "$CASES_DIR" -name '*.json' | wc -l | tr -d ' ')
    if [ "$n" -lt 5 ]; then
        printf 'FAIL (vacuity): %s fixture(s), expected 5+. The table is broken, not the code.\n' "$n"; return 1
    fi

    _row() { # _row WANT LABEL FIXTURE
        local want="$1" label="$2" fx="$3" out
        rows=$(( rows + 1 ))
        out="$( unwedge_verdict "$CASES_DIR/$3" )"
        got="${out%% *}"
        if [ "$got" = "$want" ]; then printf 'ok    %-4s %s\n' "$got" "$label"
        else printf 'FAIL  %s: got %s, wanted %s -- %s\n' "$label" "$got" "$want" "$out"; fails=1; fi
    }

    _row WEDGED  'W1 the measured shape: needs cancelled, only `ci / gate` left'  w1_gate_only_pending.json
    # W2 is why (c) is a PATTERN and not a count: two aggregators parked at once,
    # which "<= 1 pending" cannot see.
    _row WEDGED  'W2 two aggregators parked (a count-based rule misses this)'     w2_two_aggregators.json
    _row WEDGED  'W3 supersede: nothing FAILED, everything cancelled'             w3_cancelled_not_failed.json

    # H1 IS THE DISCRIMINATION ROW. Same job count and same "one failure, one
    # pending" shape as W1 -- but the pending job is real work. Widening the
    # aggregator pattern to match everything turns exactly this row red, which is
    # what makes W1 evidence rather than decoration.
    _row HEALTHY 'H1 one failure, one pending REAL job -- not wedged'             h1_last_real_job.json
    _row HEALTHY 'H2 aggregator queued but nothing cancelled: an ordinary run'   h2_no_failure.json
    _row HEALTHY 'H3 every job terminal'                                          h3_all_terminal.json
    # H4: the mistake this predicate exists to refuse. jobs == 0 is a slow start.
    _row HEALTHY 'H4 zero jobs is a slow start, never a verdict on its own'       h4_zero_jobs.json
    _row HEALTHY 'H5 an aggregator AND a real job pending'                        h5_agg_plus_real.json
    # H6 IS THE ROW THIS PREDICATE WAS WRONG ABOUT. The real job list of run
    # 34805623711 -- #3060 at QUEUE POSITION 1 -- taken live at 04:49Z: one
    # FAILED job, everything else finished, `ci / gate` queued for a runner. The
    # "failure or cancelled" draft answered WEDGED and would have force-cancelled
    # the head of the queue. A failure is an answer; only a cancellation parks
    # the gate in a run that can never assign it one.
    _row HEALTHY 'H6 a FAILED job is an answer, not a supersede (#3060 at pos 1)'  h6_failed_not_cancelled.json

    # An unreadable payload is ENV, never a licence to cancel.
    rows=$(( rows + 1 ))
    unwedge_verdict "$CASES_DIR/does-not-exist.json" > /dev/null 2>&1
    if [ $? -eq 2 ]; then printf 'ok    ENV  E1 a missing payload is ENV (exit 2), never WEDGED\n'
    else printf 'FAIL  E1 a missing payload did not answer ENV\n'; fails=1; fi

    _eq() { # _eq LABEL WANT GOT
        rows=$(( rows + 1 ))
        if [ "$3" = "$2" ]; then printf 'ok    %s\n' "$1"
        else printf 'FAIL  %s: got "%s", wanted "%s"\n' "$1" "$3" "$2"; fails=1; fi
    }

    printf '%s\n' '-- dead queue ref rows --'
    # D1/D2: the derivation, both polarities. A merge-queue branch yields a ref
    # path; an ordinary branch yields none, so a PR run is never even a candidate.
    _eq 'D1 a merge-queue branch yields a percent-encoded ref path' \
        'heads/gh-readonly-queue%2Fmain%2Fpr-3006-0c740b04' \
        "$( queue_ref_path 'gh-readonly-queue/main/pr-3006-0c740b04' )"
    rows=$(( rows + 1 ))
    if queue_ref_path 'PMAT-1098-some-branch' > /dev/null 2>&1; then
        printf 'FAIL  D2 an ordinary branch was treated as a queue ref\n'; fails=1
    else
        printf 'ok    D2 an ordinary branch is not a queue ref (never a candidate)\n'
    fi

    # D3 IS THE ROW THAT MATTERS. If the derivation breaks, every run reads 404.
    # Zero corroborating LIVE refs means the LOOKUP is broken, not the fleet.
    _eq 'D3 zero live + some dead -> REFUSE (the lookup is broken, not the fleet)' \
        'REFUSE' "$( deadref_decision 0 2 )"
    _eq 'D4 at least one live corroborates the lookup -> ACT' \
        'ACT'    "$( deadref_decision 1 1 )"
    _eq 'D5 nothing dead -> NOTHING, whatever the live count'  \
        'NOTHING' "$( deadref_decision 0 0 )"
    _eq 'D6 all live -> NOTHING'  'NOTHING' "$( deadref_decision 3 0 )"

    printf '%s\n' '-- superseded-head rows (#3292 rule 1: cancel a pull_request run whose head is stale) --'
    # (a) THE RULE ITSELF: a pull_request run's head_sha no longer matches the
    # PR's current head -- a newer run already exists at 'bbb222', so this run's
    # verdict is for a commit nobody can merge.
    _eq 'R1a pull_request, run head != PR head -> CANCEL' \
        'CANCEL' "$( superseded_verdict pull_request 'aaa111' 'bbb222' | head -1 | cut -d' ' -f1 )"
    # (b) THE DISCRIMINATION ROW. Same event, heads agree -> nothing superseded it.
    _eq 'R1b pull_request, run head == PR head -> LEAVE' \
        'LEAVE' "$( superseded_verdict pull_request 'aaa111' 'aaa111' | head -1 | cut -d' ' -f1 )"
    # (c) THE HARD EXCLUSION. Same stale-head shape as (a), but merge_group: the
    # merge queue owns this run and cancelling it ejects a PR mid-build. Never,
    # under any condition -- checked before the head comparison even runs.
    _eq 'R1c merge_group with a stale head -> LEAVE (never cancels a merge_group run)' \
        'LEAVE' "$( superseded_verdict merge_group 'aaa111' 'bbb222' | head -1 | cut -d' ' -f1 )"
    # (d) JOB COUNT IS IRRELEVANT. h1_last_real_job.json is rule 2's own "almost
    # done" fixture -- completed+in_progress jobs, the exact shape a job-count
    # gate would call healthy. Fed here as the (accepted, ignored) JOBS_FILE
    # argument, it changes nothing: the verdict is identical to row (a).
    _eq 'R1d stale head with a mostly-complete job list (rule 2 fixture) -> still CANCEL' \
        'CANCEL' "$( superseded_verdict pull_request 'aaa111' 'bbb222' \
                     "$CASES_DIR/h1_last_real_job.json" | head -1 | cut -d' ' -f1 )"
    # (e) CAPACITY IS IRRELEVANT. c2_capacity_saturated.json is rule 2's own
    # "every clean-room host saturated, 0 idle" fixture -- the exact reading
    # that makes rule 2 answer UNTOUCHED (row S3). Fed here as the (accepted,
    # ignored) CAPACITY_FILE argument, it changes nothing: still CANCEL.
    _eq 'R1e stale head while every pool is saturated (rule 2 fixture) -> still CANCEL' \
        'CANCEL' "$( superseded_verdict pull_request 'aaa111' 'bbb222' \
                     "$CASES_DIR/h1_last_real_job.json" "$CASES_DIR/c2_capacity_saturated.json" \
                     | head -1 | cut -d' ' -f1 )"
    _eq 'R1f head sha unknown -> REFUSE, never cancel on a guess' \
        'REFUSE' "$( superseded_verdict pull_request '' 'bbb222' | head -1 | cut -d' ' -f1 )"

    printf '%s\n' '-- stalled-run rows (#3292: a run may not hold a group while capacity sits idle) --'
    # THE CONTRACT: no run holds a concurrency group for longer than one sweeper
    # period while capacity to serve it sits IDLE. The trigger is evidence of a
    # WEDGE -- no dispatch progress AND idle capacity -- never run-level `queued`
    # alone, and never a job count. On #3354 (measured, #3358) the holder had 16 of
    # 18 jobs running; cancelling it on `queued` manufactured a red `gate` out of
    # its own cancel.
    _eq 'S1 queued > 30 min, pool has idle capacity, no dispatch -> CANCEL' \
        'CANCEL' "$( stall_verdict 45 2 STALLED | head -1 | cut -d' ' -f1 )"
    _eq 'S2 queued < 30 min (idle capacity, no dispatch) -> UNTOUCHED, it is young' \
        'UNTOUCHED' "$( stall_verdict 12 2 STALLED | head -1 | cut -d' ' -f1 )"
    # S3 IS THE REAL INCIDENT. 09:16-11:00Z: intel 15/16 busy, gx10 and yoga loaded.
    # Two hours queued and the rule must still do NOTHING -- it targets a wedge, not
    # a queue. A rule that fires here destroys the verdict of a run that is working.
    _eq 'S3 queued > 30 min but the pool is FULLY BUSY -> UNTOUCHED (a queue, not a wedge)' \
        'UNTOUCHED' "$( stall_verdict 120 0 STALLED | head -1 | cut -d' ' -f1 )"
    _eq 'S4 queued > 30 min, idle capacity, but jobs are PROGRESSING -> UNTOUCHED' \
        'UNTOUCHED' "$( stall_verdict 45 2 PROGRESSING | head -1 | cut -d' ' -f1 )"
    # Capacity is a REQUIRED input. The workflow token cannot read
    # orgs/<org>/actions/runners, so the unreadable case is the COMMON one and it
    # must refuse, not guess.
    _eq 'S5 capacity unknown -> REFUSE (never cancel without the evidence)' \
        'REFUSE' "$( stall_verdict 45 '' STALLED | head -1 | cut -d' ' -f1 )"

    # (c) THE COLLECTOR READS THE ORACLE. A fake fleet-bin.sh printing the oracle's
    # own line yields its count; no oracle on PATH yields EMPTY (unknown), never 0.
    local od; od="$(mktemp -d)" || return 2
    printf '%s\n' '#!/bin/sh' 'echo "fleet-bin: effective PATH: 16 live listener(s) over 17 runner dir(s) -- converged=16 stale=0 unknown=0 foreign=0 idle_dirs=1"' > "$od/fake-fleet-bin.sh"
    chmod +x "$od/fake-fleet-bin.sh"
    _eq 'C-ORACLE-a collector reads live listeners from fleet-bin.sh verify-effective-paths' \
        '16' "$( FLEET_BIN="$od/fake-fleet-bin.sh" emit_capacity_row intel self-hosted,Linux,clean-room | jq -r '.listeners' )"
    _eq 'C-ORACLE-b no oracle on the host -> EMPTY row (unknown), never a zero' \
        '' "$( FLEET_BIN="$od/absent-fleet-bin.sh" emit_capacity_row intel self-hosted,Linux,clean-room )"
    rm -rf "${od:?}"

    # The two inputs, each from a committed payload.
    _eq 'C1 idle = listeners - workers, summed over hosts carrying the labels' \
        '3' "$( pool_idle "$CASES_DIR/c1_capacity_idle.json" 'self-hosted,Linux,clean-room' )"
    # C2: gx10 has two free listeners and they carry `gpu`, not `clean-room`. An
    # idle box that cannot serve the labels is not capacity for this job.
    _eq 'C2 the measured incident: every clean-room host saturated -> 0 idle' \
        '0' "$( pool_idle "$CASES_DIR/c2_capacity_saturated.json" 'self-hosted,Linux,clean-room' )"
    _eq 'C3 workers > listeners contributes 0, never a negative' \
        '0' "$( pool_idle "$CASES_DIR/c3_capacity_worker_skew.json" 'self-hosted,Linux,clean-room' )"
    # C4 IS THE ROW THE WHOLE RULE RESTS ON. No reading means UNKNOWN, and unknown
    # must not read as `0` -- `0` is a measurement that says "a queue", while
    # unknown must reach stall_verdict's REFUSE (row S5).
    _eq 'C4 an absent capacity reading is UNKNOWN (empty), never 0' \
        '' "$( pool_idle "$CASES_DIR/does-not-exist.json" 'self-hosted,clean-room' )"
    _eq 'C5 the host-side row shape the caller must produce' \
        '{"host":"intel","labels":["self-hosted","Linux","clean-room"],"listeners":16,"workers":15}' \
        "$( capacity_row 'intel' 'self-hosted,Linux,clean-room' 16 15 )"
    _eq 'D-STALL pending work and nothing started in the window -> STALLED' \
        'STALLED' "$( dispatch_state "$CASES_DIR/s1_jobs_stalled.json" '2026-09-16T11:00:00Z' 30 )"
    _eq 'D-PROG one job started 4 min ago -> PROGRESSING (trickling, not wedged)' \
        'PROGRESSING' "$( dispatch_state "$CASES_DIR/s2_jobs_progressing.json" '2026-09-16T11:00:00Z' 30 )"
    # A run with no jobs is the VICTIM of a wedge, never its holder (H4 again).
    _eq 'D-ZERO zero jobs is never STALLED -- it is a slow start or a victim' \
        'PROGRESSING' "$( dispatch_state "$CASES_DIR/h4_zero_jobs.json" '2026-09-16T11:00:00Z' 30 )"
    _eq 'L1 the label set a run waits on is the union of its PENDING jobs' \
        'Linux,clean-room,self-hosted' "$( pending_labels "$CASES_DIR/s1_jobs_stalled.json" )"
    rows=$(( rows + 1 ))
    if stall_candidate_branch 'gh-readonly-queue/main/pr-3354-0c740b04'; then
        printf 'FAIL  S6 a merge-queue run was accepted as a stall candidate\n'; fails=1
    else
        printf 'ok    S6 a merge-queue run is NEVER a stall candidate (its build is the verdict)\n'
    fi
    rows=$(( rows + 1 ))
    if stall_candidate_branch 'PMAT-3292-ci-wedge-cannot-recur'; then
        printf 'ok    S7 an ordinary PR branch is a candidate\n'
    else
        printf 'FAIL  S7 an ordinary PR branch was refused as a candidate\n'; fails=1
    fi

    printf '\n%s row(s), %s\n' "$rows" "$( [ "$fails" -eq 0 ] && echo '0 red / FALSIFIER GREEN' || echo 'RED' )"
    return "$fails"
}

# stall_pass REPO RUN BRANCH CREATED_ISO RUN_STATUS JOBS CAPACITY NOW DRY
# -> 0 when it force-cancelled, 1 otherwise. Reads the jobs payload the wedge pass
# already fetched, so the whole rule costs NO extra API call at all -- its other
# input is a file. Everything it decides on is printed, including the refusals --
# a rule that cancels silently is how a hand-cancel got blamed on the code.
stall_pass() {
    local repo="$1" run="$2" br="$3" created="$4" rstatus="$5" jobs="$6" capacity="$7" now="$8" dry="$9"
    local age labels idle disp verdict created_s now_s
    stall_candidate_branch "$br" || return 1
    case "$rstatus" in queued|pending) ;; *) return 1 ;; esac
    created_s=$(date -u -d "$created" +%s 2>/dev/null) || created_s=""  # bashrs disable-line=DET002
    now_s=$(date -u -d "$now" +%s 2>/dev/null) || now_s=""  # bashrs disable-line=DET002
    case "$created_s$now_s" in ''|*[!0-9]*) age="" ;; *) age=$(( (now_s - created_s) / 60 )) ;; esac
    labels="$( pending_labels "$jobs" )"
    if [ -z "$labels" ]; then
        # No pending job asks for anything: there is no pool to call idle, and a
        # run with zero jobs is a victim, never the holder.
        return 1
    fi
    idle="$( pool_idle "$capacity" "$labels" )"
    disp="$( dispatch_state "$jobs" "$now" "$STALL_WINDOW_MIN" )"
    verdict="$( stall_verdict "$age" "$idle" "$disp" )"
    case "$verdict" in
        CANCEL*)
            if [ "$dry" = 1 ]; then
                printf 'WOULD-FREE %s %s -- STALLED %s [%s]\n' "$run" "$br" "${verdict#CANCEL }" "$labels"
                return 1
            fi
            if gh api -X POST "repos/$repo/actions/runs/$run/force-cancel" > /dev/null 2>&1; then
                printf 'FREED %s %s -- STALLED %s [%s]\n' "$run" "$br" "${verdict#CANCEL }" "$labels"
                return 0
            fi
            printf 'FAILED-TO-FREE %s %s (stalled)\n' "$run" "$br"
            return 1 ;;
        REFUSE*)
            printf 'refuse %s %s -- %s\n' "$run" "$br" "${verdict#REFUSE }"
            return 1 ;;
        *)  return 1 ;;
    esac
}

# superseded_pass REPO RUN BRANCH EVENT RUN_HEAD DRY -> 0 when force-cancelled,
# 1 otherwise. #3292 rule 1, wired for `scan`. `merge_group` is excluded here
# too, before any network call: the hard exclusion in superseded_verdict is the
# proof, this is the budget optimisation that follows from it -- there is no
# PR to look up for a merge-queue ref anyway.
superseded_pass() {
    local repo="$1" run="$2" br="$3" event="$4" run_head="$5" dry="$6"
    local pr_head verdict
    case "$event" in merge_group) return 1 ;; esac
    pr_head=$(gh pr view "$br" --repo "$repo" --json headRefOid -q '.headRefOid' 2>/dev/null) || pr_head=""
    verdict="$( superseded_verdict "$event" "$run_head" "$pr_head" )"
    case "$verdict" in
        CANCEL*)
            if [ "$dry" = 1 ]; then
                printf 'WOULD-FREE %s %s -- %s\n' "$run" "$br" "$verdict"
                return 1
            fi
            if gh api -X POST "repos/$repo/actions/runs/$run/force-cancel" > /dev/null 2>&1; then
                printf 'FREED %s %s -- %s\n' "$run" "$br" "$verdict"
                return 0
            fi
            printf 'FAILED-TO-FREE %s %s (superseded)\n' "$run" "$br"
            return 1 ;;
        REFUSE*)
            printf 'refuse %s %s -- %s\n' "$run" "$br" "${verdict#REFUSE }"
            return 1 ;;
        *)  return 1 ;;
    esac
}

scan() {
    local repo="$1" limit="$2" dry="$3" tmp run st freed=0 looked=0 verdict
    command -v gh > /dev/null 2>&1 || { printf 'ENV: gh is not on PATH\n' >&2; return 2; }
    tmp="$(mktemp -d)" || return 2
    # ${tmp:?} refuses an empty or unset value instead of expanding to nothing;
    # the RETURN trap also covers the early returns the explicit rm's missed.
    trap 'rm -rf "${tmp:?}"' RETURN
    # ONE list call carries status+conclusion, so the per-run jobs call is paid
    # only for candidates (feedback_gh_api_budget_and_guard_tree_runtime).
    gh run list --repo "$repo" --limit "$limit" \
        --json databaseId,status,conclusion,workflowName,headBranch,createdAt,event,headSha \
        -q '.[] | select(.workflowName=="CI") | select(.status != "completed") | "\(.databaseId)|\(.headBranch)|\(.createdAt)|\(.status)|\(.event)|\(.headSha)"' \
        > "$tmp/candidates" 2>/dev/null || { printf 'ENV: gh run list failed\n' >&2; return 2; }

    # ---- capacity, supplied ONCE by the caller (#3292) ----------------------
    # THE SWEEP DOES NOT ASK GITHUB. Idle capacity is a host-side reading
    # (`pgrep -fc "[R]unner.Listener"` / `"[R]unner.Worker"` per host, the reading
    # `make -C machines/intel verify-fleet-bin` reports); the org runners endpoint
    # is not available to this work. So the payload is an input, and its absence
    # REFUSES rather than guesses -- a rule that cancels without capacity evidence
    # is the `queued`-alone rule this one exists to replace.
    if [ -n "${CAPACITY_FILE:-}" ] && [ -r "${CAPACITY_FILE:-}" ]; then
        cp "$CAPACITY_FILE" "$tmp/capacity.json"
    else
        : > "$tmp/capacity.json"
        printf 'refuse STALL-RULE capacity unknown -- no --capacity/UNWEDGE_CAPACITY_JSON reading; nothing is cancelled on age alone\n'
    fi
    local now stalled=0 superseded=0
    now="$(date -u +%FT%TZ)"  # bashrs disable-line=DET002

    while IFS='|' read -r run br created rstatus revent rhead; do
        [ -n "$run" ] || continue
        looked=$(( looked + 1 ))
        # RULE 1 first, and independent of everything below: a superseded head
        # is cancelled on its own evidence, whatever the jobs/capacity would say.
        if superseded_pass "$repo" "$run" "$br" "$revent" "$rhead" "$dry"; then
            superseded=$(( superseded + 1 )); freed=$(( freed + 1 )); continue
        fi
        gh api "repos/$repo/actions/runs/$run/jobs" --paginate > "$tmp/jobs.json" 2>/dev/null \
            || { printf 'skip  %s (jobs unreadable)\n' "$run"; continue; }
        verdict="$( unwedge_verdict "$tmp/jobs.json" )"
        case "$verdict" in
            WEDGED*)
                if [ "$dry" = 1 ]; then
                    printf 'WOULD-FREE %s %s -- %s\n' "$run" "$br" "$verdict"
                else
                    if gh api -X POST "repos/$repo/actions/runs/$run/force-cancel" > /dev/null 2>&1; then
                        printf 'FREED %s %s -- %s\n' "$run" "$br" "$verdict"; freed=$(( freed + 1 ))
                    else
                        printf 'FAILED-TO-FREE %s %s\n' "$run" "$br"
                    fi
                fi ;;
            ENV*) printf 'skip  %s -- %s\n' "$run" "$verdict" ;;
            *)    if stall_pass "$repo" "$run" "$br" "$created" "$rstatus" \
                                "$tmp/jobs.json" "$tmp/capacity.json" "$now" "$dry"; then
                      stalled=$(( stalled + 1 ))
                  fi ;;
        esac
    done < "$tmp/candidates"

    # ---- PASS 2: merge groups whose queue ref no longer exists --------------
    # Every merge rebases the groups behind it, so GitHub deletes those refs and
    # makes new ones -- and leaves the old runs building. Collect first, decide
    # second: acting per-row would let a broken derivation cancel the whole queue
    # before anything noticed (deadref_decision, rows D3/D4).
    local live=0 dead=0 refp
    : > "$tmp/dead"
    while IFS='|' read -r run br created rstatus revent rhead; do
        [ -n "$run" ] || continue
        refp="$( queue_ref_path "$br" )" || continue
        if gh api "repos/$repo/git/ref/$refp" --jq '.object.sha' > /dev/null 2>&1; then
            live=$(( live + 1 ))
        else
            dead=$(( dead + 1 )); printf '%s|%s\n' "$run" "$br" >> "$tmp/dead"
        fi
    done < "$tmp/candidates"

    case "$( deadref_decision "$live" "$dead" )" in
        NOTHING) : ;;
        REFUSE)
            printf 'ENV: %s merge-group run(s) read as dead and NONE read as live.\n' "$dead" >&2
            printf '     The branch -> ref derivation is broken, not the fleet. Nothing cancelled.\n' >&2
            rm -f "$tmp/dead"
            # A wall-clock stamp on an operational log line is the point of the
            # line -- it says WHEN the sweep ran. bashrs disable-line is the
            # suppression bashrs itself names for this, and the idiom this repo
            # already uses (check_llama_pin.sh:248, ci_target_watch.sh:158).
            printf '%s UNWEDGE looked=%s freed=%s deadref_refused=%s dry_run=%s\n' \
                "$(date -u +%FT%TZ)" "$looked" "$freed" "$dead" "$dry"  # bashrs disable-line=DET002
            return 2 ;;
        ACT)
            while IFS='|' read -r run br created rstatus; do
                [ -n "$run" ] || continue
                if [ "$dry" = 1 ]; then
                    printf 'WOULD-FREE %s %s -- DEAD-REF (queue ref is gone; nothing can read this verdict)\n' "$run" "${br##*/}"
                else
                    # force-cancel, not `gh run cancel`: cancelling leaves the
                    # aggregator parked and the run `queued`, which is the very
                    # wedge above. Measured on 34805561949 and 34804495711.
                    if gh api -X POST "repos/$repo/actions/runs/$run/force-cancel" > /dev/null 2>&1; then
                        printf 'FREED %s %s -- DEAD-REF\n' "$run" "${br##*/}"; freed=$(( freed + 1 ))
                    else
                        printf 'FAILED-TO-FREE %s %s (dead ref)\n' "$run" "${br##*/}"
                    fi
                fi
            done < "$tmp/dead" ;;
    esac
    printf '%s UNWEDGE looked=%s freed=%s superseded_freed=%s stalled_freed=%s dry_run=%s\n' \
        "$(date -u +%FT%TZ)" "$looked" "$freed" "$superseded" "$stalled" "$dry"  # bashrs disable-line=DET002
    return 0
}

MODE=""; DRY=0; LIMIT=15; REPO="$REPO_DEFAULT"; CAPACITY_FILE="${UNWEDGE_CAPACITY_JSON:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test|--selftest) MODE=self; shift ;;
        --scan)                 MODE=scan; shift ;;
        --verdict)              MODE=verdict; VFILE="${2:-}"; shift 2 ;;
        --dry-run)              DRY=1; shift ;;
        --limit)                LIMIT="${2:-15}"; shift 2 ;;
        --repo)                 REPO="${2:-$REPO_DEFAULT}"; shift 2 ;;
        --capacity)             CAPACITY_FILE="${2:-}"; shift 2 ;;
        --emit-capacity-row)    MODE=caprow; CAPHOST="${2:-}"; CAPLABELS="${3:-}"; shift 3 ;;
        --help|-h)              usage; exit 0 ;;
        *) printf 'check_ci_unwedge.sh: unknown argument %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

# A BARE invocation runs the case table, never the scan. guard_tree.sh runs every
# scripts/check_*.sh with no arguments; a guard that needs network and
# `actions: write` to answer must not be what a bare run reaches. The predicate's
# soundness is the assertion here -- the scan is an ACTION, and actions are wired
# by a workflow with the permission to take them.
case "${MODE:-self}" in
    self)    self_test; exit $? ;;
    scan)    scan "$REPO" "$LIMIT" "$DRY"; exit $? ;;
    verdict) unwedge_verdict "${VFILE:-}"; exit $? ;;
    caprow)  emit_capacity_row "${CAPHOST:-}" "${CAPLABELS:-}"; exit $? ;;
    *)       usage >&2; exit 2 ;;
esac
