#!/usr/bin/env bash
# Every silicon axis we claim to cover must have been EXERCISED — and every axis
# we have DEFERRED must still be deferred for a reason.
#
# WHY THIS EXISTS (paiml/infra#361).
#
# aprender's correctness is silicon-dependent: SIMD paths, CUDA kernels, arch
# intrinsics, float behaviour. Today two axes are exercised nightly — x86_64-CPU
# and aarch64+Blackwell — and the gap widened on purpose: paiml/aprender#2740
# makes cuda-nightly gx10-only because its x86_64 GPU leg ran on lambda-labs,
# which must never be a CI host (paiml/infra#359). sm_89 was the STABLE reference
# and sm_121 the higher-risk target with a history of JIT bugs. We now gate only
# the risky one.
#
# That is a defensible trade, but an UNRECORDED one decays into "we test on
# whatever is plugged in". So the axes live in .github/silicon-coverage.txt and
# this asks GitHub what actually ran on them.
#
# COVERAGE IS A RUN, NOT A RUNNER (R-5, docs/specifications/yoga-nightly-job.md
# §9.2 — the recommendation this guard's own output produced).
#
# Until 2026-09-09 the question was "could some online runner serve this
# selector?" and the answer was called coverage. It is not coverage, and the
# fleet proved it the same week: paiml/infra#486 registered `yoga-gpu` with
# labels gpu,yoga,cuda,ada, and this guard immediately reported
#
#     PROMOTE   x86_64-cuda-sm89   marked pending:361, but yoga-gpu can serve it NOW
#
# on a fleet where NOT ONE JOB had ever carried that selector. Promoting the
# ledger line on that evidence is exactly the blanket exemption the ledger
# exists to prevent — the axis would have read `required … ok` for ever while
# nothing executed on the box. So the question is now:
#
#     did a job CARRYING this selector CONCLUDE inside its cadence window?
#
# and a runner that can serve an axis nothing has run is `ready`, which is a
# different and much weaker claim than `covered`.
#
# TWO DIRECTIONS, and the second is the one that keeps the ledger honest:
#
#   required   -> FAIL if nothing has run. A lane that silently stops covering
#                 an architecture looks exactly like one that never covered it.
#   pending:N  -> FAIL if something HAS run. At that moment the only thing
#                 between us and the coverage is a line in a policy file, and a
#                 "pending" that survives its own blocker is how a debt ledger
#                 becomes a blanket exemption.
#
# SELECTOR SEMANTICS, because getting this backwards is the fleet's recorded
# mistake (paiml/infra#352): a `runs-on` list matches a runner when EVERY named
# label is present on that runner. Extra runner labels never exclude. So "can
# this axis run?" is "does some online runner's label set CONTAIN all of ours?",
# and "did it run?" is the same containment against the labels a JOB asked for.
#
# INSTRUMENT PROBES, all three mandatory:
#   1. --self-test runs committed fixtures through the matcher and demands each
#      verdict. A matcher that cannot tell a subset from a superset exits 2.
#   2. POSITIVE CONTROLS on the live API: the runner listing AND the concluded-job
#      ledger must both be non-empty. Zero looks identical whether the fleet is
#      down or the token lost its scope, and blind is a NO-GO. "0 violations over
#      0 files" is this fleet's signature defect.
#   3. --fixture <dir> runs the whole classification against a committed
#      runner-listing + job-ledger pair with no network at all. That is what
#      scripts/test_silicon_coverage_run_probe.sh drives, in both polarities and
#      under a mutation that removes the run probe.
#
# AUTH: the runner's ambient org-scoped gh auth. GPU runners are REPO-scoped and
# invisible in the org listing, so BOTH lists are read and merged — that
# invisibility is exactly why yoga-gpu's absence went unnoticed for months.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${SILICON_POLICY:-$REPO_ROOT/.github/silicon-coverage.txt}"
ORG="${ORG:-paiml}"
REPO="${GITHUB_REPOSITORY:-paiml/aprender}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

# CADENCE, mirrored from machines/clean-room/lane-liveness.sh in paiml/infra: an
# axis is late when its newest run is older than max(STALE_DAYS, 2 x its own
# observed interval). Two sources, the WIDER wins, so neither can tighten the
# other.
#
# WHY THE FLOOR AND NOT ONLY THE OBSERVED GAPS. lane-liveness measured what
# happens without one: a new weekly lane with two runs was reported DEAD on days
# 4-7 of every week for its first three weeks, because a gap estimator needs
# samples it does not have yet. The floor is what makes a BRAND NEW axis
# judgeable at all, and D-2 (operator, 2026-09-09: "yoga should test nightly")
# makes 3 days the right one for every axis here.
#
# Unlike lane-liveness this reads no `cron:`: an axis is not a workflow. Several
# workflows may carry one selector and one workflow may carry several axes, so
# there is no single declaration to read — only the runs.
STALE_DAYS="${SILICON_STALE_DAYS:-3}"
LOOKBACK_DAYS="${SILICON_LOOKBACK_DAYS:-30}"
RUN_CAP="${SILICON_RUN_CAP:-60}"
TAB="$(printf '\t')"

# labels_contain <runner-label-csv> <required-label-csv>
# 0 when every required label is present in the runner's set. Case-insensitive:
# GitHub's built-in labels are `Linux`/`X64`/`ARM64` and a selector may spell
# them either way.
# All locals are `_lc_`-prefixed. The first version used `_have`/`_want`, which
# are also the loop variables in selftest() — POSIX sh has no function scope, so
# the call CLOBBERED the caller's expectation and every case compared a verdict
# against a label string. The self-test caught it on the first run, which is the
# only reason to have one.
labels_contain() {
    _lc_have=",$(printf '%s' "$1" | tr 'A-Z' 'a-z'),"
    _lc_want="$(printf '%s' "$2" | tr 'A-Z' 'a-z')"
    _lc_ifs="$IFS"; IFS=','
    for _lc_l in $_lc_want; do
        [ -n "$_lc_l" ] || continue
        case "$_lc_have" in
            *",$_lc_l,"*) ;;
            *) IFS="$_lc_ifs"; return 1 ;;
        esac
    done
    IFS="$_lc_ifs"
    return 0
}

selftest() {
    _broken=0
    # runner labels                    | selector                | want (0=serves)
    for _case in \
        "self-hosted,linux,x64,clean-room,intel|self-hosted,clean-room,intel|0" \
        "self-hosted,linux,arm64,gpu,gx10,cuda,blackwell,gb10|self-hosted,gpu,gx10,cuda,blackwell|0" \
        "self-hosted,linux,x64,clean-room,intel|self-hosted,gpu,gx10|1" \
        "self-hosted,linux,arm64,gpu,gx10,cuda|self-hosted,gpu,gx10,cuda,blackwell|1" \
        "self-hosted,Linux,ARM64,gpu,gx10,cuda,blackwell|self-hosted,gpu,gx10,cuda,blackwell|0" \
        "self-hosted,linux,x64,cuda,gpu,lambda-4090|self-hosted,gpu,yoga,cuda,ada|1" \
        ; do
        _have="${_case%%|*}"; _rest="${_case#*|}"
        _want_sel="${_rest%%|*}"; _want="${_rest#*|}"
        if labels_contain "$_have" "$_want_sel"; then _got=0; else _got=1; fi
        if [ "$_got" != "$_want" ]; then
            printf '  [%s] vs [%s]: expected %s (0=serves), got %s\n' \
                "$_have" "$_want_sel" "$_want" "$_got"
            _broken=$((_broken + 1))
        fi
    done
    if [ "$_broken" -gt 0 ]; then
        printf 'INSTRUMENT BROKEN — the label matcher cannot classify its own cases.\n'
        printf 'Refusing to report on the fleet with a matcher that failed its own tests.\n'
        return 2
    fi
    printf 'self-test: 6 label cases, matcher classifies all of them as specified\n'
    return 0
}

# iso_epoch <ISO-8601 Z> -> seconds since epoch, or nothing (rc 1) if unparseable.
# Parses a timestamp the API GAVE us; the only wall clock read is `now` below.
iso_epoch() {
    [ -n "${1:-}" ] || return 1
    date -u -d "$1" +%s 2>/dev/null
}

# ── THE RUN PROBE ───────────────────────────────────────────────────────────
# axis_evidence <jobs-tsv> <selector> <job-name glob>
#
# Every concluded job that CARRIED this selector AND answers to this name,
# newest first, one per line:
#
#     <completed_at>\t<conclusion>\t<workflow>\t<job>
#
# and nothing at all when none did. ISO-8601 Z is fixed width, so a lexical
# reverse sort IS a chronological one — no date arithmetic needed to order them
# (the same trick lane-liveness.sh uses to pick a newest run).
#
# The ledger's third column is a job's `labels`, which the API defines as the
# `runs-on` list the job ASKED FOR — not the label set of the runner that
# happened to take it. That is the same containment question the policy declares,
# asked of a job instead of a machine.
#
# WHY THE NAME AND NOT ONLY THE LABELS. Measured on the first live run of this
# rewrite, 2026-09-09:
#
#     ok  x86_64-cpu  failure 2026-09-09T09:00:26Z 0d ago — Silicon Nightly / summary
#
# `summary` is silicon-nightly's needs-aggregator. It runs `runs-on:
# [self-hosted, clean-room, intel]` and no tests at all, so it carries the axis's
# selector perfectly — as does every other job in the repo that lands on the
# intel pool, of which there were 100+ in the same window. A pool label set does
# not name an axis, and scoring coverage off one means `x86_64-cpu` reads green
# for as long as ANY job touches intel. That is the same shape as the defect R-5
# fixes, one level down: a probe answering a question adjacent to the one asked.
#
# So the policy's 4th column NAMES the job, and the link is fail-safe: rename the
# job and the axis goes UNCOVERED (red), never silently green. An axis with no
# 4th column still matches on labels alone and SAYS SO in its verdict line.
#
# MUTATION SEAM (§9.2). scripts/test_silicon_coverage_run_probe.sh replaces this
# function's body with a fixed fresh positive and REQUIRES the UNCOVERED fixture
# to go green. If it stays red the probe is not what produces the verdict, and
# the whole R-5 change is decoration. Keep the BEGIN/END markers and the closing
# brace in column 0; the mutator finds the body by them.
# MUTATION-SEAM-BEGIN axis_evidence
axis_evidence() {
    _ax_jobs="$1"; _ax_sel="$2"; _ax_glob="${3:-*}"
    while IFS="$TAB" read -r _ax_ts _ax_concl _ax_labels _ax_job _ax_wf _ax_rid; do
        [ -n "$_ax_ts" ] || continue
        labels_contain "$_ax_labels" "$_ax_sel" || continue
        # Unquoted on purpose: the 4th policy column is a GLOB, matched against
        # the job name the API reports (a matrix leg appends " (params)").
        # shellcheck disable=SC2254
        case "$_ax_job" in $_ax_glob) ;; *) continue ;; esac
        printf '%s\t%s\t%s\t%s\n' "$_ax_ts" "$_ax_concl" "${_ax_wf:-?}" "${_ax_job:-?}"
    done < "$_ax_jobs" | sort -r
}
# MUTATION-SEAM-END axis_evidence

# build_jobs <out-tsv>
# The concluded-job ledger for this repo, over the lookback window:
#
#     <completed_at>\t<conclusion>\t<runs-on csv>\t<job>\t<workflow>\t<run id>
#
# ASK THE API FOR THE EVENT, NEVER FILTER A MIXED PAGE. `runs?per_page=100` is
# the newest 100 runs of ALL events; on a repo this busy that is a few hours of
# pull_request traffic and can hold zero scheduled runs. paiml/infra's dead-man's
# switch reported a weekly lane DEAD — with its never-succeeded sentinel — off
# exactly that page, for a lane whose last two scheduled runs both succeeded. It
# reported an absence it had never looked for.
#
# `success` and `failure` both count as EXERCISED: a red nightly ran the silicon
# and is reported red by its own lane. `cancelled`, `skipped` and a null
# conclusion do not — nothing executed, which is the state this guard exists to
# make visible. A queued job on a purged runner registration is `null` for ever,
# and that is the ~14-day failure this whole lane is built around.
build_jobs() {
    _bj_out="$1"
    : > "$_bj_out"
    _bj_cut=$(date -u -d "-${LOOKBACK_DAYS} days" +%s 2>/dev/null || printf '0')
    _bj_runs="$(mktemp)" || return 1
    for _bj_ev in schedule workflow_dispatch; do
        gh api "repos/$REPO/actions/runs?event=${_bj_ev}&per_page=100" \
            --jq '.workflow_runs[] | [(.id|tostring), .created_at, .name] | @tsv' \
            2>/dev/null >> "$_bj_runs" || true
    done
    # Newest first, so RUN_CAP truncates the OLD tail rather than a random one.
    sort -u "$_bj_runs" | sort -t"$TAB" -k2,2r > "${_bj_runs}.s" && mv "${_bj_runs}.s" "$_bj_runs"
    runs_seen=0; runs_scanned=0; runs_capped=0; runs_old=0
    while IFS="$TAB" read -r _bj_id _bj_created _bj_name; do
        [ -n "$_bj_id" ] || continue
        runs_seen=$((runs_seen + 1))
        _bj_ts=$(iso_epoch "$_bj_created") || continue
        if [ "$_bj_ts" -lt "$_bj_cut" ]; then runs_old=$((runs_old + 1)); continue; fi
        if [ "$runs_scanned" -ge "$RUN_CAP" ]; then runs_capped=$((runs_capped + 1)); continue; fi
        runs_scanned=$((runs_scanned + 1))
        gh api "repos/$REPO/actions/runs/${_bj_id}/jobs?per_page=100" \
            --jq '.jobs[] | select(.conclusion=="success" or .conclusion=="failure")
                          | [.completed_at, .conclusion, ([.labels[]] | join(",")), .name] | @tsv' \
            2>/dev/null \
            | awk -v wf="$_bj_name" -v rid="$_bj_id" -F'\t' 'NF >= 4 { print $0 "\t" wf "\t" rid }' \
            >> "$_bj_out"
    done < "$_bj_runs"
    rm -f "$_bj_runs"
    return 0
}

MODE=live
FIXTURE=""
case "${1:-}" in
    --self-test)
        printf -- '-- instrument self-test --\n'
        selftest; exit $? ;;
    --fixture)
        FIXTURE="${2:-}"
        [ -d "$FIXTURE" ] || { printf 'usage: %s --fixture <dir>\n' "$0"; exit 2; }
        MODE=fixture ;;
    '') ;;
    *)
        printf 'usage: %s [--self-test | --fixture <dir>]\n' "$0"; exit 2 ;;
esac

printf '== silicon coverage ==\n'

RUNNERS=""; JOBS=""
if [ "$MODE" = "fixture" ]; then
    POLICY="$FIXTURE/policy.txt"
    RUNNERS="$FIXTURE/runners.tsv"
    JOBS="$FIXTURE/jobs.tsv"
    printf 'MODE: fixture (%s) — no network, no live API\n' "$FIXTURE"
    for f in "$POLICY" "$RUNNERS" "$JOBS"; do
        [ -f "$f" ] || { printf 'NO-GO: fixture is missing %s\n' "$f"; exit 2; }
    done
fi
printf 'policy: %s\n\n' "$POLICY"
[ -f "$POLICY" ] || { printf 'NO-GO: %s does not exist.\n' "$POLICY"; exit 2; }

printf -- '-- instrument self-test --\n'
selftest || exit 2

# ── the runners: org-scoped AND repo-scoped ─────────────────────────────────
if [ "$MODE" = "live" ]; then
    RUNNERS="$(mktemp)"; JOBS="$(mktemp)"
    trap 'rm -f "$RUNNERS" "$JOBS"' EXIT
    : > "$RUNNERS"
    for src in "orgs/$ORG/actions/runners" "repos/$REPO/actions/runners"; do
        gh api --paginate "$src?per_page=100" \
            --jq '.runners[] | select(.status=="online") | (.name) + "\t" + ([.labels[].name] | join(","))' \
            2>/dev/null >> "$RUNNERS" || true
    done
    sort -u -o "$RUNNERS" "$RUNNERS"
fi
n_runners="$(grep -cve '^[[:space:]]*$' "$RUNNERS" || true)"

printf -- '\n-- runners --\n'
printf 'online runners visible (org + this repo): %s\n' "${n_runners:-0}"
# Probe 2a: the positive control on the runner listing.
if [ "${n_runners:-0}" -eq 0 ]; then
    printf 'NO-GO: zero online runners. That looks identical whether the fleet is\n'
    printf 'down or this token lost its scope, and GPU runners are REPO-scoped and\n'
    printf 'invisible in the org listing. Blind is a NO-GO, not a pass.\n'
    exit 2
fi

# ── the runs: what actually EXECUTED ────────────────────────────────────────
runs_seen=0; runs_scanned=0; runs_capped=0; runs_old=0
if [ "$MODE" = "live" ]; then
    build_jobs "$JOBS" || { printf 'NO-GO: could not build the job ledger.\n'; exit 2; }
fi
n_jobs="$(grep -cve '^[[:space:]]*$' "$JOBS" || true)"

printf -- '\n-- runs --\n'
if [ "$MODE" = "live" ]; then
    printf 'scheduled+dispatched runs listed: %s (within %sd: %s scanned, %s older, %s over the %s cap)\n' \
        "$runs_seen" "$LOOKBACK_DAYS" "$runs_scanned" "$runs_old" "$runs_capped" "$RUN_CAP"
fi
printf 'concluded jobs in the ledger: %s\n' "${n_jobs:-0}"
# Probe 2b: the positive control on the job ledger. THE DENOMINATOR AGAIN.
# "no axis has run" and "the ledger is empty because nothing was read" are the
# same output, and one of them is a fleet-wide outage while the other is a
# broken guard. Refusing is the only honest answer.
if [ "${n_jobs:-0}" -eq 0 ]; then
    printf 'NO-GO: the job ledger is EMPTY. Every axis would read UNCOVERED, which is\n'
    printf 'indistinguishable from a guard that read nothing at all. Zero over zero is\n'
    printf 'this fleet-s signature defect; refusing rather than reporting an absence\n'
    printf 'this guard never actually looked for.\n'
    exit 2
fi

# ── the axes ────────────────────────────────────────────────────────────────
printf -- '\n-- axes --\n'
now=$(date -u +%s)   # the ONLY wall-clock read; everything else is an API timestamp
axes=0; required=0; covered=0; uncovered=0; stale=0; missing=0
deferred=0; promotable=0; ready=0
fail=0
while IFS= read -r line; do
    line="${line%%#*}"
    case "$line" in ''|[[:space:]]*[[:space:]]) ;; esac
    set -- $line
    [ $# -ge 3 ] || continue
    axis="$1"; status="$2"; selector="$3"; jobcol="${4:-}"
    axes=$((axes + 1))

    # The OPTIONAL 4th column names the job. Fail-safe by construction: a
    # renamed job stops matching and the axis goes red, never silently green.
    # Absent, the axis matches on labels alone — and its verdict line says so,
    # because "some job on this pool ran" is a much weaker claim than coverage.

    jobglob='*'; jobnote=' [labels only]'
    case "$jobcol" in
        job:*) jobglob="${jobcol#job:}"; jobnote="" ;;
        '')    ;;
        *)     printf '  BAD       %-20s 4th field is "%s"; expected job:<glob>\n' "$axis" "$jobcol"
               fail=1; continue ;;
    esac

    # (a) COULD a runner serve it? Weaker than coverage, and no longer confused
    # with it — but still the thing that separates "declared and blocked" from
    # "declared, unblocked, and still not running".
    server=""
    while IFS="$TAB" read -r rname rlabels; do
        [ -n "$rname" ] || continue
        if labels_contain "$rlabels" "$selector"; then server="$rname"; break; fi
    done < "$RUNNERS"

    # (b) DID a job carrying it conclude, and how long ago?
    evidence="$(axis_evidence "$JOBS" "$selector" "$jobglob")"
    n_runs="$(printf '%s' "$evidence" | grep -c . || true)"
    newest_iso=""; newest_concl=""; newest_wf=""; newest_job=""
    age=-1; window="$STALE_DAYS"; fresh=0
    if [ "${n_runs:-0}" -gt 0 ]; then
        IFS="$TAB" read -r newest_iso newest_concl newest_wf newest_job <<<"$evidence"
        newest_epoch=$(iso_epoch "$newest_iso") && age=$(( (now - newest_epoch) / 86400 ))
        interval=1
        if [ "$n_runs" -ge 2 ] && [ "$age" -ge 0 ]; then
            oldest_iso="$(printf '%s\n' "$evidence" | tail -1 | cut -f1)"
            if oldest_epoch=$(iso_epoch "$oldest_iso"); then
                oldest_age=$(( (now - oldest_epoch) / 86400 ))
                interval=$(( (oldest_age - age) / (n_runs - 1) ))
                [ "$interval" -lt 1 ] && interval=1
            fi
        fi
        window=$(( 2 * interval ))
        [ "$window" -lt "$STALE_DAYS" ] && window="$STALE_DAYS"
        [ "$age" -ge 0 ] && [ "$age" -le "$window" ] && fresh=1
    fi

    case "$status" in
        required)
            required=$((required + 1))
            if [ "$fresh" -eq 1 ]; then
                covered=$((covered + 1))
                printf '  ok        %-20s %s %s %sd ago — %s / %s%s\n' \
                    "$axis" "$newest_concl" "$newest_iso" "$age" "$newest_wf" "$newest_job" "$jobnote"
            elif [ "${n_runs:-0}" -gt 0 ]; then
                stale=$((stale + 1)); fail=1
                printf '  STALE     %-20s REQUIRED: last carried %sd ago (window %sd) — %s / %s\n' \
                    "$axis" "$age" "$window" "$newest_wf" "$newest_job"
            elif [ -n "$server" ]; then
                uncovered=$((uncovered + 1)); fail=1
                printf '  UNCOVERED %-20s REQUIRED: %s CAN serve [%s], but no job named %s has carried it\n' \
                    "$axis" "$server" "$selector" "$jobglob"
            else
                missing=$((missing + 1)); fail=1
                printf '  MISSING   %-20s REQUIRED: no online runner carries [%s], and no job named %s has\n' \
                    "$axis" "$selector" "$jobglob"
            fi
            ;;
        pending:*)
            deferred=$((deferred + 1))
            if [ "$fresh" -eq 1 ]; then
                promotable=$((promotable + 1)); fail=1
                printf '  PROMOTE   %-20s marked %s, but a job CARRIED it %sd ago — %s / %s\n' \
                    "$axis" "$status" "$age" "$newest_wf" "$newest_job"
            elif [ -n "$server" ]; then
                ready=$((ready + 1))
                printf '  ready     %-20s %s — %s can serve it; nothing has run on it yet\n' \
                    "$axis" "$status" "$server"
            else
                printf '  deferred  %-20s %s — no runner, no run\n' "$axis" "$status"
            fi
            ;;
        *)
            printf '  BAD       %-20s unknown status "%s"\n' "$axis" "$status"
            fail=1
            ;;
    esac
done < "$POLICY"

printf -- '\n-- denominators --\n'
printf 'axes declared: %s  (required %s: covered %s, STALE %s, UNCOVERED %s, MISSING %s;' \
    "$axes" "$required" "$covered" "$stale" "$uncovered" "$missing"
printf ' deferred %s: PROMOTABLE %s, ready %s)\n' "$deferred" "$promotable" "$ready"
printf 'evidence: %s online runner(s), %s concluded job(s), floor %sd\n' \
    "$n_runners" "$n_jobs" "$STALE_DAYS"

# THE DENOMINATOR. A policy that parsed nothing must not read as clean.
if [ "$axes" -eq 0 ]; then
    printf '\nNO-GO: 0 axes parsed from %s. Nothing was measured.\n' "$POLICY"; exit 2
fi

{
    printf '\n### Silicon coverage\n\n'
    printf -- '- axes: **%s** (required %s, deferred %s)\n' "$axes" "$required" "$deferred"
    printf -- '- required covered by a RUN: **%s of %s**\n' "$covered" "$required"
    # No parentheses in this string, and an `if` rather than `[ ] &&`: bashrs
    # lexes a `(` inside a printf argument on the same line as a `[ ]` test as an
    # unescaped test paren (SC1028), and a `[ ] &&` tail returns non-zero when the
    # test is false.
    if [ "$promotable" -gt 0 ]; then
        printf -- '- **%s deferred axis/axes have RUN and must be promoted**\n' "$promotable"
    fi
    if [ "$ready" -gt 0 ]; then
        printf -- '- %s deferred axis/axes have a runner but no run yet\n' "$ready"
    fi
    printf -- '- runners inspected: %s; concluded jobs inspected: %s\n' "$n_runners" "$n_jobs"
} >> "$SUMMARY"

if [ "$fail" -ne 0 ]; then
    printf '\nFAIL: silicon coverage does not match the policy.\n'
    printf 'MISSING   — no runner and no run: the architecture stopped being tested\n'
    printf '            and nothing said so.\n'
    printf 'UNCOVERED — a runner CAN serve the axis and nothing ever has. This is the\n'
    printf '            state a runner-existence check scored as coverage, which is why\n'
    printf '            it is now its own verdict.\n'
    printf 'STALE     — it ran, then stopped, inside its own measured cadence.\n'
    printf 'PROMOTE   — a deferred axis is running: the blocker is gone and the only\n'
    printf '            thing left is the line in .github/silicon-coverage.txt.\n'
    exit 1
fi
printf '\nOK: every required axis has RUN inside its cadence window, and no deferred\n'
printf 'axis is running behind its own ledger line.\n'
