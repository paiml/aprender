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
#   4. --self-test ALSO runs this script END TO END against a PATH-shimmed `gh`
#      serving canned API pages (PMAT-3337). Every unmodelled call exits 97 and
#      is logged, so a row can never pass on a call the table did not model.
#
# AUTH: the runner's ambient org-scoped gh auth. GPU runners are REPO-scoped and
# invisible in the org listing, so BOTH lists are read and merged — that
# invisibility is exactly why yoga-gpu's absence went unnoticed for months.
#
# EVIDENCE IS READ PER AXIS-CARRYING WORKFLOW (PMAT-3337, paiml/aprender#3337).
#
# Until 2026-09-16 the ledger came from the repo-wide `actions/runs?event=…`
# listings of EVERY workflow, sorted newest-first and truncated at a global
# RUN_CAP of 60. Measured in CI on 2026-09-15, same script, same policy:
#
#     16:27Z #3320  200 listed (within 30d: 25 scanned, 175 older, 0 over the cap)  -> NO-GO
#     19:23Z #3114  200 listed (within 30d: 60 scanned, 75 older, 65 over the cap)  -> UNCOVERED sm89
#     19:31Z local  same listing shape                                              -> GO
#
# The cap made coverage a function of how many OTHER workflows had run since
# the carrying one, and an event page with nothing recent failed every PR's
# guard-tree on a view of the API. Both are the Probe-2b shape: an absence this
# guard never looked for, scored as evidence of absence.
#
# So each axis's workflow(s) are DERIVED from the tree — the policy names a job,
# `.github/workflows/*.yml` says which file declares a job with that id or
# `name:` — and each is asked for exactly its own runs inside the lookback
# (`actions/workflows/<file>/runs?event=<e>&created=>=<date>`), paginated under
# a bounded page cap. Deriving is fail-safe in the same direction as the 4th
# column: rename the job and no workflow is read, so the axis goes red.
#
# A page cap that is reached before the listing is exhausted is NOT a negative.
# An axis not found in a truncated read scores TRUNCATED (unknown, exit 2) with
# the counts it did read — never UNCOVERED, STALE or MISSING.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
POLICY="${SILICON_POLICY:-$REPO_ROOT/.github/silicon-coverage.txt}"
WORKFLOWS_DIR="${SILICON_WORKFLOWS_DIR:-$REPO_ROOT/.github/workflows}"
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
STALE_DAYS="${SILICON_STALE_DAYS:-3}"
LOOKBACK_DAYS="${SILICON_LOOKBACK_DAYS:-30}"
# Pagination of ONE workflow's ONE event listing. There is deliberately no
# repo-wide run cap: a cap shared across workflows is what let other lanes push
# an axis's carrying run out of the read (PMAT-3337).
PAGE_SIZE="${SILICON_PAGE_SIZE:-100}"
PAGE_CAP="${SILICON_PAGE_CAP:-10}"
TAB="$(printf '\t')"
api_calls=0
probe_calls=0

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
    # Deterministic: the instant comes from $1, an API timestamp, not the clock.
    date -u -d "$1" +%s 2>/dev/null  # bashrs disable-line=DET002
}

# epoch_day <epoch> -> YYYY-MM-DD (UTC) of that instant. Deterministic: $1.
epoch_day() {
    date -u -d "@$1" +%Y-%m-%d 2>/dev/null  # bashrs disable-line=DET002
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

# ── WHICH WORKFLOW CARRIES AN AXIS ──────────────────────────────────────────
# workflow_jobs <workflows dir> -> "<file>\t<job id>\t<job name>" per job.
#
# NOT A YAML PARSER, same posture as check_silicon_cuda.sh: a job is a 2-space
# key under the top-level `jobs:`, and its name is the 4-space `name:` under it
# (step names sit deeper, under `- name:`). The API reports a job by `name:`
# when one is set and by its id otherwise, so both are emitted and either may
# match the policy's glob. A templated name is matched by its id only.
workflow_jobs() {
    for _wj_f in "$1"/*.yml "$1"/*.yaml; do
        [ -f "$_wj_f" ] || continue
        awk -v f="${_wj_f##*/}" -v q="'" '
            function emit() { printf "%s\t%s\t%s\n", f, id, name; id = "" }
            /^jobs:[[:space:]]*(#.*)?$/ { injobs = 1; next }
            /^[^[:space:]#]/ { if (id != "") emit(); injobs = 0; next }
            injobs && /^  [A-Za-z0-9_-]+:[[:space:]]*(#.*)?$/ {
                if (id != "") emit()
                id = $1; sub(/:$/, "", id); name = ""; next
            }
            injobs && id != "" && name == "" && /^    name:/ {
                n = $0
                sub(/^    name:[[:space:]]*/, "", n)
                sub(/[[:space:]]+#.*$/, "", n)
                gsub("^[\"" q "]|[\"" q "]$", "", n)
                name = n; next
            }
            END { if (id != "") emit() }
        ' "$_wj_f"
    done
}

# axis_workflows <workflows dir> <job glob> -> workflow file names, one per line.
axis_workflows() {
    _aw_glob="$2"
    workflow_jobs "$1" | while IFS="$TAB" read -r _aw_f _aw_id _aw_name; do
        [ -n "$_aw_f" ] || continue
        # shellcheck disable=SC2254
        case "$_aw_id" in $_aw_glob) printf '%s\n' "$_aw_f"; continue ;; esac
        [ -n "$_aw_name" ] || continue
        case "$_aw_name" in *'${{'*) continue ;; esac
        # shellcheck disable=SC2254
        case "$_aw_name" in $_aw_glob) printf '%s\n' "$_aw_f" ;; esac
    done | sort -u
}

# build_jobs <out-tsv> <listings-tsv> <workflow file>...
# The concluded-job ledger for the AXIS-CARRYING workflows, over the lookback:
#
#     <completed_at>\t<conclusion>\t<runs-on csv>\t<job>\t<workflow>\t<run id>
#
# and, per workflow x event, what the listing held:
#
#     <workflow file>\t<event>\t<runs read>\t<total_count>\t<ok|truncated|unread>
#
# ASK THE API FOR WHAT YOU MEAN, NEVER FILTER A MIXED PAGE. `runs?per_page=100`
# is the newest 100 runs of ALL events; on a repo this busy that is a few hours
# of pull_request traffic. paiml/infra's dead-man's switch reported a weekly
# lane DEAD off exactly that page. The repo-wide `runs?event=schedule` page this
# replaced was the same defect one filter in: its composition depended on every
# other workflow (PMAT-3337). So the listing names the WORKFLOW, the EVENT and
# the WINDOW, and nothing on the page needs discarding except the sub-day part
# of `created>=<date>`.
#
# `success` and `failure` both count as EXERCISED: a red nightly ran the silicon
# and is reported red by its own lane. `cancelled`, `skipped` and a null
# conclusion do not — nothing executed, which is the state this guard exists to
# make visible. A queued job on a purged runner registration is `null` for ever,
# and that is the ~14-day failure this whole lane is built around.
#
# A LISTING THAT ERRORS IS NOT AN EMPTY LISTING: rc 1, and the caller refuses.
build_jobs() {
    _bj_out="$1"; _bj_lst="$2"; shift 2
    : > "$_bj_out"; : > "$_bj_lst"
    _bj_cut=$(( now - LOOKBACK_DAYS * 86400 ))
    _bj_since="$(epoch_day "$_bj_cut")" || return 1
    _bj_runs="$(mktemp)" || return 1
    for _bj_wf in "$@"; do
        for _bj_ev in schedule workflow_dispatch; do
            _bj_page=1; _bj_read=0; _bj_total=0; _bj_state=ok
            while :; do
                if [ "$_bj_page" -gt "$PAGE_CAP" ]; then
                    [ "$_bj_read" -lt "$_bj_total" ] && _bj_state=truncated
                    break
                fi
                _bj_pf="$(mktemp)" || return 1
                api_calls=$((api_calls + 1))
                if ! gh api "repos/$REPO/actions/workflows/${_bj_wf}/runs?event=${_bj_ev}&created=%3E%3D${_bj_since}&per_page=${PAGE_SIZE}&page=${_bj_page}" \
                    --jq '"#total\t\(.total_count)", (.workflow_runs[] | [(.id|tostring), .created_at, .name] | @tsv)' \
                    > "$_bj_pf" 2>/dev/null; then
                    printf 'listing FAILED: %s %s page %s\n' "$_bj_wf" "$_bj_ev" "$_bj_page"
                    rm -f "$_bj_pf" "$_bj_runs"
                    return 1
                fi
                _bj_total="$(awk -F'\t' '$1 == "#total" { print $2; exit }' "$_bj_pf")"
                case "$_bj_total" in ''|*[!0-9]*)
                    printf 'listing UNREADABLE: %s %s page %s has no total_count\n' "$_bj_wf" "$_bj_ev" "$_bj_page"
                    rm -f "$_bj_pf" "$_bj_runs"
                    return 1 ;;
                esac
                _bj_n="$(awk -F'\t' '$1 != "#total" && NF >= 2' "$_bj_pf" | tee -a "$_bj_runs" | grep -c . || true)"
                rm -f "$_bj_pf"
                _bj_read=$((_bj_read + ${_bj_n:-0}))
                if [ "${_bj_n:-0}" -lt "$PAGE_SIZE" ] || [ "$_bj_read" -ge "$_bj_total" ]; then break; fi
                _bj_page=$((_bj_page + 1))
            done
            # CROSS-CHECK AN EMPTY LISTING (PMAT-3337 §8). The recorded failure
            # mode is the API handing back a bad EMPTY page, and an empty page is
            # exactly what would score UNCOVERED. So ask once more WITHOUT the
            # created filter: if the newest run of this workflow and event is
            # inside the lookback, the two listings contradict each other and
            # neither is evidence. Older than the window, or none: a true negative.
            _bj_ptotal="-"; _bj_pnewest="-"
            if [ "$_bj_total" -eq 0 ] && [ "$_bj_state" = ok ]; then
                api_calls=$((api_calls + 1)); probe_calls=$((probe_calls + 1))
                if ! _bj_probe="$(gh api "repos/$REPO/actions/workflows/${_bj_wf}/runs?event=${_bj_ev}&per_page=1" \
                    --jq '"\(.total_count)\t\(.workflow_runs[0].created_at // "")"' 2>/dev/null)"; then
                    printf 'listing FAILED: %s %s unfiltered cross-check\n' "$_bj_wf" "$_bj_ev"
                    rm -f "$_bj_runs"
                    return 1
                fi
                _bj_ptotal="${_bj_probe%%"$TAB"*}"; _bj_pnewest="${_bj_probe#*"$TAB"}"
                if [ -n "$_bj_pnewest" ] && _bj_pe=$(iso_epoch "$_bj_pnewest") && [ "$_bj_pe" -ge "$_bj_cut" ]; then
                    _bj_state=inconsistent
                fi
                [ -n "$_bj_pnewest" ] || _bj_pnewest="none"
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$_bj_wf" "$_bj_ev" "$_bj_read" "$_bj_total" "$_bj_state" \
                "$_bj_ptotal" "$_bj_pnewest" >> "$_bj_lst"
        done
    done
    # One newest-first list across the axis workflows, deduplicated by run id.
    sort -t"$TAB" -u -k1,1 "$_bj_runs" | sort -t"$TAB" -k2,2r > "${_bj_runs}.s" && mv "${_bj_runs}.s" "$_bj_runs"
    runs_seen=0; runs_scanned=0; runs_old=0
    while IFS="$TAB" read -r _bj_id _bj_created _bj_name; do
        [ -n "$_bj_id" ] || continue
        runs_seen=$((runs_seen + 1))
        _bj_ts=$(iso_epoch "$_bj_created") || continue
        if [ "$_bj_ts" -lt "$_bj_cut" ]; then runs_old=$((runs_old + 1)); continue; fi
        runs_scanned=$((runs_scanned + 1))
        _bj_jf="$(mktemp)" || return 1
        api_calls=$((api_calls + 1))
        if gh api "repos/$REPO/actions/runs/${_bj_id}/jobs?per_page=100" \
            --jq '.jobs[] | select(.conclusion=="success" or .conclusion=="failure")
                          | [.completed_at, .conclusion, ([.labels[]] | join(",")), .name] | @tsv' \
            > "$_bj_jf" 2>/dev/null; then
            awk -v wf="$_bj_name" -v rid="$_bj_id" -F'\t' 'NF >= 4 { print $0 "\t" wf "\t" rid }' \
                "$_bj_jf" >> "$_bj_out"
        else
            # An unread run is a hole in the window, not a run without the job.
            printf '%s\tjobs\t%s\t0\tunread\n' "$_bj_name" "$_bj_id" >> "$_bj_lst"
        fi
        rm -f "$_bj_jf"
    done < "$_bj_runs"
    rm -f "$_bj_runs"
    return 0
}

# window_holes <listings-tsv> <workflow file>... -> one line naming every read of
# these workflows that did not reach the lookback edge, or nothing.
# window_contradictions <listings-tsv> <workflow file>... -> one line naming every
# EMPTY filtered listing whose unfiltered newest run is inside the lookback.
window_contradictions() {
    _wc_lst="$1"; shift
    for _wc_wf in "$@"; do
        awk -F'\t' -v wf="$_wc_wf" -v lb="$LOOKBACK_DAYS" '
            $1 == wf && $5 == "inconsistent" { printf "%s %s: created>=lookback listed %s of %s, but the unfiltered listing (total %s) has its newest run at %s, inside the %sd lookback; ", $1, $2, $3, $4, $6, $7, lb }
        ' "$_wc_lst"
    done
}

window_holes() {
    _wh_lst="$1"; shift
    for _wh_wf in "$@"; do
        awk -F'\t' -v wf="$_wh_wf" -v cap="$PAGE_CAP" -v sz="$PAGE_SIZE" '
            $1 == wf && $5 == "truncated" { printf "%s %s: read %s of %s in-window runs (page cap %sx%s); ", $1, $2, $3, $4, cap, sz }
        ' "$_wh_lst"
    done
    # Unread job pages are keyed by the workflow's display name; the listing
    # does not map file -> name, so any unread page is a hole for every axis.
    awk -F'\t' '$5 == "unread" { n++ } END { if (n) printf "%s run(s) whose jobs could not be read; ", n }' "$_wh_lst"
}

# ── SELF-TEST ROWS: the whole script against a PATH-shimmed gh ──────────────
# Hermetic: no network. The shim applies the script's own --jq to canned API
# JSON, so the pages are API-shaped and the filters are exercised. Fixture
# instants are offsets from ST_NOW, a FIXED epoch injected into the subject as
# SILICON_NOW, so no row changes its verdict with the calendar.
# SILICON_SELFTEST_SUBJECT runs the same rows against another copy of this
# guard (e.g. origin/main's); SILICON_SELFTEST_NOW re-anchors the fixture for a
# subject that reads the wall clock instead of SILICON_NOW.
ST=""
ST_NOW="${SILICON_SELFTEST_NOW:-1788998400}"   # 2026-09-10T00:00:00Z
ST_REPO="shim/repo"

st_write_shim() {
    mkdir -p "$ST/bin" || return 1
    cat > "$ST/bin/gh" <<'SHIM'
#!/usr/bin/env bash
# gh shim for check_silicon_coverage.sh --self-test. Unmodelled = exit 97, loud.
set -u
d="${SHIM_DIR:?SHIM_DIR unset}"
printf '%s\n' "$*" >> "$d/calls.log"
unmodelled() {
    printf 'gh-shim: UNMODELLED %s\n' "$*" >> "$d/unmodelled.log"
    printf 'gh-shim: UNMODELLED %s\n' "$*" >&2
    exit 97
}
[ "${1:-}" = api ] || unmodelled "$@"
shift
path=""; expr="."
while [ $# -gt 0 ]; do
    case "$1" in
        --paginate) ;;
        --jq) [ $# -ge 2 ] || unmodelled "--jq without an expression"; expr="$2"; shift ;;
        -*) unmodelled "flag $1" ;;
        *) [ -z "$path" ] || unmodelled "second path $1"; path="$1" ;;
    esac
    shift
done
f="$(awk -F'\t' -v p="$path" '$1 == p { print $2; exit }' "$d/routes.tsv")"
[ -n "$f" ] || unmodelled "route $path"
if [ "$f" = "@ERROR" ]; then
    printf 'gh: Server Error (HTTP 502)\n' >&2
    exit 1
fi
jq -r "$expr" "$d/$f"
SHIM
    chmod +x "$ST/bin/gh"
}

st_iso() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }  # bashrs disable-line=DET002
st_route() { printf '%s\t%s\n' "$1" "$2" >> "$ST/u/routes.tsv"; }

st_reset() {
    rm -rf "${ST:?}/u"
    mkdir -p "$ST/u/pages" "$ST/u/workflows" || return 1
    : > "$ST/u/routes.tsv"; : > "$ST/u/runs.tsv"; : > "$ST/u/calls.log"
    ST_PAGE_SIZE=100; ST_PAGE_CAP=10; ST_RUN_CAP=60; ST_ERROR=0; ST_HIDE_FILTERED=""
    printf '%s\n' \
        '# self-test policy' \
        'cpu-axis   required  self-hosted,shimcpu  job:cpu-leg' \
        'gpu-axis   required  self-hosted,shimgpu  job:gpu-leg*' > "$ST/u/policy.txt"
    printf '%s\n' 'name: Sil' 'on:' '  schedule:' "    - cron: '0 0 * * *'" 'jobs:' \
        '  cpu-leg:' '    runs-on: [self-hosted, shimcpu]' '    steps:' \
        '      - name: gpu-leg a step name is not a job name' '        run: true' \
        > "$ST/u/workflows/sil.yml"
    printf '%s\n' 'name: Gpu' 'on:' '  workflow_dispatch:' 'jobs:' '  gpu:' \
        '    name: gpu-leg (x86, shim)' '    runs-on: [self-hosted, shimgpu]' \
        '    steps:' '      - run: true' > "$ST/u/workflows/gpu.yml"
    printf '%s\n' 'name: Noise' 'on:' '  pull_request:' 'jobs:' '  lint:' \
        '    runs-on: [self-hosted, shimcpu]' '    steps:' '      - run: true' \
        > "$ST/u/workflows/noise.yml"
    printf '%s' '{"runners":[{"name":"shim-runner-cpu","status":"online","labels":[{"name":"self-hosted"},{"name":"shimcpu"}]},{"name":"shim-runner-gpu","status":"online","labels":[{"name":"self-hosted"},{"name":"shimgpu"}]},{"name":"shim-offline","status":"offline","labels":[{"name":"x"}]}]}' \
        > "$ST/u/pages/org-runners.json"
    printf '%s' '{"runners":[]}' > "$ST/u/pages/repo-runners.json"
    st_route "orgs/shimorg/actions/runners?per_page=100" pages/org-runners.json
    st_route "repos/$ST_REPO/actions/runners?per_page=100" pages/repo-runners.json
}

# st_run <id> <workflow file> <event> <hours before ST_NOW> <on the repo-wide
#        event page: 1|0> <job "name|conclusion|labels csv">...
st_run() {
    _sr_id="$1"; _sr_wf="$2"; _sr_ev="$3"; _sr_h="$4"; _sr_g="$5"; shift 5
    _sr_epoch=$(( ST_NOW - _sr_h * 3600 ))
    case "$_sr_wf" in sil.yml) _sr_name="Sil" ;; gpu.yml) _sr_name="Gpu" ;; *) _sr_name="Noise" ;; esac
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$_sr_id" "$_sr_wf" "$_sr_ev" "$_sr_name" "$_sr_epoch" "$_sr_g" \
        >> "$ST/u/runs.tsv"
    _sr_done="$(st_iso $(( _sr_epoch + 1800 )))"
    {
        printf '{"jobs":['
        _sr_sep=""
        for _sr_j in "$@"; do
            _sr_jn="${_sr_j%%|*}"; _sr_r="${_sr_j#*|}"
            _sr_c="${_sr_r%%|*}"; _sr_l="${_sr_r#*|}"
            [ "$_sr_c" = null ] || _sr_c="\"$_sr_c\""
            _sr_lj="$(printf '%s' "$_sr_l" | sed 's/[^,][^,]*/"&"/g')"
            printf '%s{"name":"%s","conclusion":%s,"completed_at":"%s","labels":[%s]}' \
                "$_sr_sep" "$_sr_jn" "$_sr_c" "$_sr_done" "$_sr_lj"
            _sr_sep=","
        done
        printf ']}\n'
    } > "$ST/u/pages/jobs-$_sr_id.json"
    st_route "repos/$ST_REPO/actions/runs/$_sr_id/jobs?per_page=100" "pages/jobs-$_sr_id.json"
}

# st_page <runs subset tsv> <total> <out json>
st_page() {
    {
        printf '{"total_count":%s,"workflow_runs":[' "$2"
        _sp_sep=""
        while IFS="$TAB" read -r _sp_id _sp_wf _sp_ev _sp_name _sp_epoch _; do
            [ -n "$_sp_id" ] || continue
            printf '%s{"id":%s,"created_at":"%s","name":"%s","event":"%s"}' \
                "$_sp_sep" "$_sp_id" "$(st_iso "$_sp_epoch")" "$_sp_name" "$_sp_ev"
            _sp_sep=","
        done < "$1"
        printf ']}\n'
    } > "$3"
}

# st_finish: serve every listing either guard generation asks for.
st_finish() {
    _sf_since="$(epoch_day $(( ST_NOW - 30 * 86400 )))"
    _sf_since_e="$(date -u -d "$_sf_since" +%s)"  # bashrs disable-line=DET002
    for _sf_ev in schedule workflow_dispatch; do
        _sf_key="repos/$ST_REPO/actions/runs?event=${_sf_ev}&per_page=100"
        if [ "$ST_ERROR" = 1 ]; then st_route "$_sf_key" @ERROR; continue; fi
        awk -F'\t' -v ev="$_sf_ev" '$3 == ev && $6 == 1' "$ST/u/runs.tsv" \
            | sort -t"$TAB" -k5,5nr > "$ST/u/l.tsv"
        st_page "$ST/u/l.tsv" "$(grep -c . "$ST/u/l.tsv")" "$ST/u/pages/global-$_sf_ev.json"
        st_route "$_sf_key" "pages/global-$_sf_ev.json"
    done
    for _sf_wf in sil.yml gpu.yml noise.yml; do
        for _sf_ev in schedule workflow_dispatch; do
            awk -F'\t' -v wf="$_sf_wf" -v ev="$_sf_ev" -v s="$_sf_since_e" \
                '$2 == wf && $3 == ev && $5 >= s' "$ST/u/runs.tsv" \
                | sort -t"$TAB" -k5,5nr > "$ST/u/l.tsv"
            # The unfiltered newest run: what the empty-listing cross-check reads.
            awk -F'\t' -v wf="$_sf_wf" -v ev="$_sf_ev" '$2 == wf && $3 == ev' "$ST/u/runs.tsv" \
                | sort -t"$TAB" -k5,5nr > "$ST/u/all.tsv"
            head -n 1 "$ST/u/all.tsv" > "$ST/u/p.tsv"
            st_page "$ST/u/p.tsv" "$(grep -c . "$ST/u/all.tsv")" "$ST/u/pages/wfall-$_sf_wf-$_sf_ev.json"
            _sf_pkey="repos/$ST_REPO/actions/workflows/${_sf_wf}/runs?event=${_sf_ev}&per_page=1"
            if [ "$ST_ERROR" = 1 ]; then st_route "$_sf_pkey" @ERROR
            else st_route "$_sf_pkey" "pages/wfall-$_sf_wf-$_sf_ev.json"; fi
            # A bad EMPTY page: the created-filtered listing of this workflow lies.
            [ "$_sf_wf" = "$ST_HIDE_FILTERED" ] && : > "$ST/u/l.tsv"
            _sf_total="$(grep -c . "$ST/u/l.tsv")"
            _sf_p=1
            while :; do
                _sf_key="repos/$ST_REPO/actions/workflows/${_sf_wf}/runs?event=${_sf_ev}&created=%3E%3D${_sf_since}&per_page=${ST_PAGE_SIZE}&page=${_sf_p}"
                if [ "$ST_ERROR" = 1 ]; then
                    st_route "$_sf_key" @ERROR
                else
                    _sf_from=$(( (_sf_p - 1) * ST_PAGE_SIZE + 1 )); _sf_to=$(( _sf_p * ST_PAGE_SIZE ))
                    sed -n "${_sf_from},${_sf_to}p" "$ST/u/l.tsv" > "$ST/u/p.tsv"
                    st_page "$ST/u/p.tsv" "$_sf_total" "$ST/u/pages/wf-$_sf_wf-$_sf_ev-$_sf_p.json"
                    st_route "$_sf_key" "pages/wf-$_sf_wf-$_sf_ev-$_sf_p.json"
                fi
                [ $(( _sf_p * ST_PAGE_SIZE )) -ge "$_sf_total" ] && break
                _sf_p=$((_sf_p + 1))
            done
        done
    done
}

# st_check <row> <want rc> <want output line, ERE>
st_check() {
    _sc_row="$1"; _sc_rc="$2"; _sc_re="$3"
    _sc_out="$ST/$_sc_row.out"
    env PATH="$ST/bin:$PATH" SHIM_DIR="$ST/u" ORG=shimorg GITHUB_REPOSITORY="$ST_REPO" \
        GITHUB_STEP_SUMMARY=/dev/null SILICON_POLICY="$ST/u/policy.txt" \
        SILICON_WORKFLOWS_DIR="$ST/u/workflows" SILICON_NOW="$ST_NOW" \
        SILICON_LOOKBACK_DAYS=30 SILICON_STALE_DAYS=3 \
        SILICON_PAGE_SIZE="$ST_PAGE_SIZE" SILICON_PAGE_CAP="$ST_PAGE_CAP" \
        SILICON_RUN_CAP="$ST_RUN_CAP" \
        bash "$ST_SUBJECT" > "$_sc_out" 2>&1
    _sc_got=$?
    _sc_why=""
    [ "$_sc_got" = "$_sc_rc" ] || _sc_why="rc=$_sc_got want $_sc_rc; "
    grep -Eq -- "$_sc_re" "$_sc_out" || _sc_why="${_sc_why}no line /$_sc_re/; "
    [ -s "$ST/u/calls.log" ] || _sc_why="${_sc_why}the gh shim was never called; "
    if [ -s "$ST/u/unmodelled.log" ]; then _sc_why="${_sc_why}UNMODELLED gh call(s); "; fi
    if [ -n "$_sc_why" ]; then
        printf '  FAIL %-14s %s\n' "$_sc_row" "$_sc_why"
        sed 's/^/       | /' "$_sc_out"
        [ -s "$ST/u/unmodelled.log" ] && sed 's/^/       ! /' "$ST/u/unmodelled.log"
        return 1
    fi
    printf '  ok   %-14s rc=%s  %s\n' "$_sc_row" "$_sc_got" \
        "$(grep -Em1 -- "$_sc_re" "$_sc_out" | sed 's/^  *//')"
    return 0
}

CPU_JOB="cpu-leg|success|self-hosted,shimcpu"
GPU_JOB="gpu-leg (x86, shim)|success|self-hosted,shimgpu"
NOISE_JOB="lint|success|self-hosted,shimcpu"

selftest_rows() {
    command -v jq >/dev/null 2>&1 || {
        printf 'INSTRUMENT BROKEN — the gh shim needs jq to apply the guard-s --jq filters.\n'
        return 2
    }
    ST_SUBJECT="${SILICON_SELFTEST_SUBJECT:-$SELF}"
    ST="$(mktemp -d)" || return 2
    st_write_shim || { rm -rf "${ST:?}"; return 2; }
    printf 'gh-shim rows: subject %s, fixture now %s\n' "$ST_SUBJECT" "$(st_iso "$ST_NOW")"
    _rows=0; _rows_bad=0

    # shim: the canned fleet is what the guard saw (2 online, 1 offline ignored).
    st_reset; st_run 9001 sil.yml schedule 12 1 "$CPU_JOB"
    st_run 9101 gpu.yml schedule 20 1 "$GPU_JOB"; st_finish
    _rows=$((_rows + 1))
    st_check shim 0 '^online runners visible \(org \+ this repo\): 2$' || _rows_bad=$((_rows_bad + 1))

    # derive: gpu-axis is carried by gpu.yml through its `name:`, not its id,
    # and a STEP named gpu-leg in sil.yml does not make sil.yml a carrier.
    _rows=$((_rows + 1))
    st_check derive 0 '^  gpu-axis +gpu\.yml$' || _rows_bad=$((_rows_bad + 1))

    # (a) more in-window runs than a global cap, carrying run beyond it.
    st_reset; ST_RUN_CAP=3
    for _h in 2 4 6 8; do st_run "900$_h" sil.yml schedule "$_h" 1 "$CPU_JOB"; done
    st_run 9201 noise.yml schedule 1 1 "$NOISE_JOB"; st_run 9203 noise.yml schedule 3 1 "$NOISE_JOB"
    st_run 9101 gpu.yml schedule 36 1 "$GPU_JOB"; st_finish
    _rows=$((_rows + 1))
    st_check a-beyond-cap 0 '^  ok +gpu-axis ' || _rows_bad=$((_rows_bad + 1))

    # (b) repo-wide schedule page holds nothing recent; the workflow listing does.
    st_reset
    st_run 9001 sil.yml schedule 12 0 "$CPU_JOB"; st_run 9101 gpu.yml schedule 20 0 "$GPU_JOB"
    st_run 9301 sil.yml schedule 960 1 "$CPU_JOB"
    st_run 9401 noise.yml workflow_dispatch 5 1 "$NOISE_JOB"; st_finish
    _rows=$((_rows + 1))
    st_check b-empty-event 0 '^  ok +gpu-axis ' || _rows_bad=$((_rows_bad + 1))

    # (c) the axis workflow truly has no in-window run: a true negative.
    st_reset
    st_run 9001 sil.yml schedule 12 1 "$CPU_JOB"; st_run 9201 noise.yml schedule 2 1 "$NOISE_JOB"
    st_run 9102 gpu.yml schedule 960 1 "$GPU_JOB"; st_finish
    _rows=$((_rows + 1))
    st_check c-true-negative 1 '^  UNCOVERED +gpu-axis .*shim-runner-gpu CAN serve' \
        || _rows_bad=$((_rows_bad + 1))

    # (d) page cap reached before the lookback edge; not found in what was read.
    st_reset; ST_PAGE_SIZE=2; ST_PAGE_CAP=1
    st_run 9001 sil.yml schedule 12 1 "$CPU_JOB"
    for _h in 6 30 54; do
        st_run "91$_h" gpu.yml schedule "$_h" 1 "gpu-leg (x86, shim)|cancelled|self-hosted,shimgpu"
    done
    st_finish
    _rows=$((_rows + 1))
    st_check d-truncated 2 '^  TRUNCATED +gpu-axis .*read 2 of 3' || _rows_bad=$((_rows_bad + 1))

    # (f) the filtered listing comes back EMPTY while the unfiltered newest run
    # is inside the window: contradictory listings are NO-GO, never UNCOVERED.
    st_reset; ST_HIDE_FILTERED=gpu.yml
    st_run 9001 sil.yml schedule 12 1 "$CPU_JOB"; st_run 9101 gpu.yml schedule 20 1 "$GPU_JOB"
    st_finish
    _rows=$((_rows + 1))
    st_check f-inconsistent 2 '^  NO-GO +gpu-axis +REQUIRED: inconsistent listing .*listed 0 of 0.*total 1' \
        || _rows_bad=$((_rows_bad + 1))

    # (e) the API errors: refuse, exactly as before.
    st_reset; ST_ERROR=1
    st_run 9001 sil.yml schedule 12 1 "$CPU_JOB"; st_run 9101 gpu.yml schedule 20 1 "$GPU_JOB"
    st_finish
    _rows=$((_rows + 1))
    st_check e-api-error 2 '^NO-GO' || _rows_bad=$((_rows_bad + 1))

    rm -rf "${ST:?}"
    if [ "$_rows_bad" -gt 0 ]; then
        printf 'INSTRUMENT BROKEN — %s of %s gh-shim row(s) did not classify as specified.\n' \
            "$_rows_bad" "$_rows"
        return 2
    fi
    printf 'self-test: %s gh-shim rows, every verdict as specified, no unmodelled call\n' "$_rows"
    return 0
}

MODE=live
FIXTURE=""
case "${1:-}" in
    --self-test)
        printf -- '-- instrument self-test --\n'
        selftest || exit $?
        selftest_rows; exit $? ;;
    --fixture)
        FIXTURE="${2:-}"
        [ -d "$FIXTURE" ] || { printf 'usage: %s --fixture <dir>\n' "$0"; exit 2; }
        MODE=fixture ;;
    '') ;;
    *)
        printf 'usage: %s [--self-test | --fixture <dir>]\n' "$0"; exit 2 ;;
esac

printf '== silicon coverage ==\n'

RUNNERS=""; JOBS=""; LISTINGS=""; AXIS_WF=""
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

# The ONLY wall-clock read in this guard; everything else is an API timestamp.
# Freshness is a distance from now, so this read is the measurement, not an
# impurity to design away. SILICON_NOW pins it for the self-test's rows.
now="${SILICON_NOW:-}"
if [ -z "$now" ]; then
    now=$(date -u +%s)  # bashrs disable-line=DET002
fi

# ── the runners: org-scoped AND repo-scoped ─────────────────────────────────
if [ "$MODE" = "live" ]; then
    RUNNERS="$(mktemp)"; JOBS="$(mktemp)"; LISTINGS="$(mktemp)"; AXIS_WF="$(mktemp)"
    trap 'rm -f "$RUNNERS" "$JOBS" "$LISTINGS" "$AXIS_WF"' EXIT
    : > "$RUNNERS"
    for src in "orgs/$ORG/actions/runners" "repos/$REPO/actions/runners"; do
        api_calls=$((api_calls + 1))
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

# ── which workflows carry the axes (derived from the tree) ──────────────────
if [ "$MODE" = "live" ]; then
    printf -- '\n-- axis workflows (derived from %s) --\n' "$WORKFLOWS_DIR"
    : > "$AXIS_WF"
    while IFS= read -r line; do
        line="${line%%#*}"
        set -- $line
        [ $# -ge 3 ] || continue
        _pw_glob='*'
        case "${4:-}" in job:*) _pw_glob="${4#job:}" ;; '') ;; *) continue ;; esac
        _pw_wfs="$(axis_workflows "$WORKFLOWS_DIR" "$_pw_glob")"
        if [ -z "$_pw_wfs" ]; then
            printf '  %-20s (no workflow declares a job matching %s)\n' "$1" "$_pw_glob"
            continue
        fi
        printf '  %-20s %s\n' "$1" "$(printf '%s' "$_pw_wfs" | tr '\n' ' ' | sed 's/ $//')"
        printf '%s\n' "$_pw_wfs" | while IFS= read -r _pw_wf; do
            printf '%s\t%s\n' "$1" "$_pw_wf"
        done >> "$AXIS_WF"
    done < "$POLICY"
fi

# ── the runs: what actually EXECUTED ────────────────────────────────────────
runs_seen=0; runs_scanned=0; runs_old=0
if [ "$MODE" = "live" ]; then
    # Word-splitting on purpose: workflow file names carry no whitespace.
    # shellcheck disable=SC2046
    build_jobs "$JOBS" "$LISTINGS" $(cut -f2 "$AXIS_WF" | sort -u) \
        || { printf 'NO-GO: could not build the job ledger — a listing errored, and an\n'
             printf 'unread listing is not an empty one.\n'; exit 2; }
fi
n_jobs="$(grep -cve '^[[:space:]]*$' "$JOBS" || true)"

printf -- '\n-- runs --\n'
if [ "$MODE" = "live" ]; then
    while IFS="$TAB" read -r _l_wf _l_ev _l_read _l_total _l_state _; do
        [ -n "$_l_wf" ] || continue
        printf 'listing %-24s %-18s %s of %s in-window run(s) read  %s\n' \
            "$_l_wf" "$_l_ev" "$_l_read" "$_l_total" "$_l_state"
    done < "$LISTINGS"
    printf 'axis-workflow runs listed: %s (within %sd: %s scanned, %s older); page cap %sx%s\n' \
        "$runs_seen" "$LOOKBACK_DAYS" "$runs_scanned" "$runs_old" "$PAGE_CAP" "$PAGE_SIZE"
fi
printf 'concluded jobs in the ledger: %s\n' "${n_jobs:-0}"
# Probe 2b: the positive control on the job ledger. THE DENOMINATOR AGAIN.
# "no axis has run" and "the ledger is empty because nothing was read" are the
# same output, and one of them is a fleet-wide outage while the other is a
# broken guard. Refusing is the only honest answer.
#
# (Probe 2c, "the repo-wide schedule page held nothing recent", is gone with the
# page it guarded: evidence is now one listing per axis workflow, and a listing
# that errors refuses inside build_jobs — PMAT-3337.)
if [ "${n_jobs:-0}" -eq 0 ]; then
    printf 'NO-GO: the job ledger is EMPTY. Every axis would read UNCOVERED, which is\n'
    printf 'indistinguishable from a guard that read nothing at all. Zero over zero is\n'
    printf 'this fleet-s signature defect; refusing rather than reporting an absence\n'
    printf 'this guard never actually looked for.\n'
    exit 2
fi

# ── the axes ────────────────────────────────────────────────────────────────
printf -- '\n-- axes --\n'
axes=0; required=0; covered=0; uncovered=0; stale=0; missing=0
deferred=0; promotable=0; ready=0; unknown=0; inconsistent=0
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

    # (c) Was the window READ to its edge? A non-fresh axis whose workflow
    # listing stopped at the page cap is unknown, not absent.
    holes=""; contradiction=""
    if [ "$MODE" = "live" ] && [ "$fresh" -eq 0 ]; then
        # shellcheck disable=SC2046
        holes="$(window_holes "$LISTINGS" $(awk -F'\t' -v a="$axis" '$1 == a { print $2 }' "$AXIS_WF"))"
        holes="${holes%; }"
        # shellcheck disable=SC2046
        contradiction="$(window_contradictions "$LISTINGS" $(awk -F'\t' -v a="$axis" '$1 == a { print $2 }' "$AXIS_WF"))"
        contradiction="${contradiction%; }"
    fi

    case "$status" in
        required)
            required=$((required + 1))
            if [ "$fresh" -eq 1 ]; then
                covered=$((covered + 1))
                printf '  ok        %-20s %s %s %sd ago — %s / %s%s\n' \
                    "$axis" "$newest_concl" "$newest_iso" "$age" "$newest_wf" "$newest_job" "$jobnote"
            elif [ -n "$contradiction" ]; then
                unknown=$((unknown + 1)); inconsistent=$((inconsistent + 1))
                printf '  NO-GO     %-20s REQUIRED: inconsistent listing — %s\n' "$axis" "$contradiction"
            elif [ -n "$holes" ]; then
                unknown=$((unknown + 1))
                printf '  TRUNCATED %-20s REQUIRED: unknown — no fresh job named %s in what was read: %s\n' \
                    "$axis" "$jobglob" "$holes"
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
            elif [ -n "$contradiction" ]; then
                unknown=$((unknown + 1)); inconsistent=$((inconsistent + 1))
                printf '  NO-GO     %-20s %s: inconsistent listing — %s\n' "$axis" "$status" "$contradiction"
            elif [ -n "$holes" ]; then
                unknown=$((unknown + 1))
                printf '  TRUNCATED %-20s %s: unknown — a PROMOTE could hide in the unread part: %s\n' \
                    "$axis" "$status" "$holes"
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
printf 'axes declared: %s  (required %s: covered %s, STALE %s, UNCOVERED %s, MISSING %s, TRUNCATED %s;' \
    "$axes" "$required" "$covered" "$stale" "$uncovered" "$missing" "$unknown"
printf ' deferred %s: PROMOTABLE %s, ready %s)\n' "$deferred" "$promotable" "$ready"
printf 'evidence: %s online runner(s), %s concluded job(s), floor %sd\n' \
    "$n_runners" "$n_jobs" "$STALE_DAYS"
if [ "$MODE" = "live" ]; then
    printf 'api calls: %s (empty-listing cross-checks: %s)\n' "$api_calls" "$probe_calls"
fi

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
    if [ "$unknown" -gt 0 ]; then
        printf -- '- **%s axis/axes UNKNOWN: the page cap was reached before the lookback edge**\n' "$unknown"
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
if [ "$unknown" -gt 0 ]; then
    printf '\nNO-GO: %s axis/axes could not be judged (%s from an inconsistent listing).\n' "$unknown" "$inconsistent"
    printf 'Either a workflow listing stopped at the page cap before the %sd lookback\n' "$LOOKBACK_DAYS"
    printf 'edge without the job, or an empty created>=lookback listing was contradicted\n'
    printf 'by the unfiltered one. An unread window is Unknown, never UNCOVERED.\n'
    exit 2
fi
printf '\nOK: every required axis has RUN inside its cadence window, and no deferred\n'
printf 'axis is running behind its own ledger line.\n'
