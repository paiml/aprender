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

    # The two inputs, each from a committed payload.
    _eq 'P1 idle capacity counts only ONLINE, not-busy runners that carry the labels' \
        '2' "$( pool_idle "$CASES_DIR/p1_runners_idle.json" 'self-hosted,Linux,clean-room' )"
    # P2: gx10-blackwell is idle and cannot serve clean-room. An idle box that does
    # not carry the labels is not capacity for this job.
    _eq 'P2 the measured incident: every clean-room runner busy -> 0 idle' \
        '0' "$( pool_idle "$CASES_DIR/p2_runners_saturated.json" 'self-hosted,Linux,clean-room' )"
    _eq 'P3 an OFFLINE listener is not idle capacity' \
        '0' "$( pool_idle "$CASES_DIR/p3_runners_offline.json" 'self-hosted,Linux,clean-room' )"
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
        --json databaseId,status,conclusion,workflowName,headBranch \
        -q '.[] | select(.workflowName=="CI") | select(.status != "completed") | "\(.databaseId)|\(.headBranch)"' \
        > "$tmp/candidates" 2>/dev/null || { printf 'ENV: gh run list failed\n' >&2; return 2; }

    while IFS='|' read -r run br; do
        [ -n "$run" ] || continue
        looked=$(( looked + 1 ))
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
            *)    : ;;
        esac
    done < "$tmp/candidates"

    # ---- PASS 2: merge groups whose queue ref no longer exists --------------
    # Every merge rebases the groups behind it, so GitHub deletes those refs and
    # makes new ones -- and leaves the old runs building. Collect first, decide
    # second: acting per-row would let a broken derivation cancel the whole queue
    # before anything noticed (deadref_decision, rows D3/D4).
    local live=0 dead=0 refp
    : > "$tmp/dead"
    while IFS='|' read -r run br; do
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
            while IFS='|' read -r run br; do
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
    printf '%s UNWEDGE looked=%s freed=%s dry_run=%s\n' "$(date -u +%FT%TZ)" "$looked" "$freed" "$dry"  # bashrs disable-line=DET002
    return 0
}

MODE=""; DRY=0; LIMIT=15; REPO="$REPO_DEFAULT"
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test|--selftest) MODE=self; shift ;;
        --scan)                 MODE=scan; shift ;;
        --verdict)              MODE=verdict; VFILE="${2:-}"; shift 2 ;;
        --dry-run)              DRY=1; shift ;;
        --limit)                LIMIT="${2:-15}"; shift 2 ;;
        --repo)                 REPO="${2:-$REPO_DEFAULT}"; shift 2 ;;
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
    *)       usage >&2; exit 2 ;;
esac
