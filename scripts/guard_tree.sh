#!/usr/bin/env bash
#
# guard_tree.sh -- run every scripts/check_*.sh guard and report EVERY
# failure, not just the first (BSE-01, PMAT-1062).
#
# WHY THIS EXISTS
# ---------------
# .github/workflows/ci.yml's `guard-runner-labels` job invokes ~110
# scripts/check_*.sh guards as ~110 separate GitHub Actions steps. GitHub
# Actions steps are fail-fast by construction: the first failing step stops
# the job, so a PR that breaks guard #3 never learns whether guards #4-110
# also broke. This script removes that ordering dependency: it runs every
# guard in the universe, collects every result, prints one row per guard,
# and exits 1 only after every guard has had a chance to run.
#
# THE UNIVERSE IS NEVER A WRITTEN LIST
# -------------------------------------
# `guard_universe()` is exactly `git ls-files 'scripts/check_*.sh'`. A
# hand-maintained list drifts the moment someone adds or removes a guard
# script without also touching the list -- this script cannot go stale that
# way because it has no list to go stale.
#
# THE CARGO-FREE SUBSET
# ----------------------
# `--no-cargo` (and `--list --no-cargo`) restrict the universe to guards
# whose OWN source never contains a bare `cargo ` token:
# `grep -LE '(^|[^a-z_-])cargo '`. This is a purely textual test over the
# guard's source -- it does not execute anything to classify it -- so ci.yml
# can run this subset on a bare runner with no cargo/docker toolchain and
# guard_tree.sh's classification can never independently drift from a
# hand-copied list living in the workflow.
#
# THE INVOCATION FORM IS `bash <guard>`, NOT `<guard>`
# -----------------------------------------------------
# Ten of the 96 tracked guards are mode 100644 (`git ls-files -s
# 'scripts/check_*.sh'`), because ci.yml has always invoked them as
# `bash scripts/check_X.sh` and the executable bit therefore never mattered.
# Exec'ing the path directly turned those ten into `Permission denied` FAIL
# rows -- a red row for a guard that never ran, which is the one failure a
# run-all runner may not produce. Measured before the fix: 3 of the 47
# cargo-free guards (check_assertions_exclude, check_baseline_ratchets,
# check_workflow_path_filters) failed this way while ci.yml ran all three
# green. The `--help` probe uses the same form for the same reason.
#
# SELF-TEST DETECTION
# --------------------
# A guard "advertises --self-test" when its own `--help` output contains the
# literal substring `self-test`. Detection captures `"$g" --help 2>&1` into a
# variable and greps the variable with `grep -c` -- NEVER `<producer> |
# grep -q`, which SIGPIPEs the producer under `pipefail` and can read a real
# match as a false negative (paiml/infra feedback_pipefail_...). A guard that
# advertises self-test gets TWO rows: `[self-test]` (run first) and `[run]`.
# A guard that does not gets one `[run]` row.
#
# FOUR KINDS OF GUARD THIS RUNNER MUST NOT RUN BARE
# ---------------------------------------------------
# "Run every guard" is only true of guards a bare invocation can run at all,
# and there are four populations for which it is false. All four are
# DERIVED from the oracle that already owns the answer -- neither is a name
# list here, because a second hand-maintained list is the defect this whole
# file exists to avoid.
#
#   1. ARGUMENT- OR ENV-WIRED. `check_beat_measurements.sh <beat-log>` in
#      beat-speed-nightly.yml, `check_pr_review_arm4.sh` under a step `env:`
#      carrying PR_NUMBER in pr-review-quorum.yml, `check_receipt_complete.sh
#      --dag <path>` in ci.yml. Run bare these FAIL, and a red row for a guard
#      that was never given its input is the one failure a run-all runner may
#      not produce (the same reason mode-644 guards are run through `bash`).
#      The oracle is the workflows themselves: a guard whose EVERY invocation
#      across .github/workflows/*.yml carries an argument or sits in a step
#      with an `env:` block is `skipped: ... wired-with-args in <workflow>`.
#      One row, never a failure -- it IS wired, just not by this runner.
#
#   2. RELEASE-TIME. scripts/check_no_timing_in_required.sh holds the registry
#      of guards that assert a DURATION and may therefore never reach a
#      required status check (aprender#2671: eleven wall-clock assertions have
#      failed in one, and one ratio rewrite blocked all nine open PRs). This
#      dispatcher runs inside a required job, so running them here would
#      re-create by DISPATCH exactly what that guard forbids by NAME. The
#      registry is read with `check_no_timing_in_required.sh --list`, so the
#      two can never disagree, and that guard now also asserts this run set
#      does not contain its registry -- deleting the skip below turns it RED.
#
#   3. WIRED ELSEWHERE. `check_book_cli_parity.sh` is bare-invoked only in
#      book.yml (PMAT-1062), after that workflow builds `apr` -- a binary
#      this dispatcher's `--no-cargo` step, running inside ci.yml, has no
#      reason to have on PATH. ci.yml never names the guard at all. Running
#      it here duplicates a guard another workflow already owns and fails it
#      for a reason that workflow was written to avoid. The oracle is the
#      same workflow scan as rule 1: a guard with a BARE invocation somewhere
#      and NO invocation of any kind (bare or ARG) in ci.yml is
#      `skipped: ... wired-elsewhere <workflow>` -- one row, still wired,
#      just not by this dispatcher. A guard invoked in ci.yml AND elsewhere
#      still runs; a guard invoked nowhere (dark) still runs -- neither case
#      matches "no invocation in ci.yml AND a bare invocation elsewhere".
#
#   4. UNWIRED-BASELINE. scripts/unwired_guards_baseline.txt (owned by
#      check_guards_are_wired.sh, PMAT-1062) is the shrink-only ledger of
#      guards already accepted as reached by no workflow. A name in it is a
#      known, argued-about gap, not something this run-all dispatcher should
#      turn into a fresh required-check failure. The oracle is the ledger
#      file itself, read the same way check_guards_are_wired.sh's own ratchet
#      reads it: `skipped: ... unwired-baseline (shrink-only ledger)`. The
#      ledger can only shrink (baseline_ratchet_check), so this skip
#      population can only shrink with it; removing an entry from the ledger
#      makes this dispatcher run that guard again.
#
# USAGE
# -----
#   scripts/guard_tree.sh                     run every guard
#   scripts/guard_tree.sh --no-cargo          run only the cargo-free subset
#   scripts/guard_tree.sh --cargo-only        run only the cargo-using subset
#   scripts/guard_tree.sh --dry-run           print run:/skipped: rows, run nothing
#   scripts/guard_tree.sh --list              print the guard universe
#   scripts/guard_tree.sh --list --no-cargo   print the cargo-free subset
#   scripts/guard_tree.sh --list --cargo-only print the cargo-using subset
#
# `--list` prints the raw SUBSET (the cargo classification only); `--dry-run`
# prints the DECISION (subset minus the two skip populations above). Readers
# that want "what does CI actually execute" must ask --dry-run: that is what
# check_guards_are_wired.sh and check_no_timing_in_required.sh both do, so a
# guard this runner skips is never reported as wired by it.

set -uo pipefail

# Resolved BEFORE the cd below, because the pool re-executes this very file as
# its own worker (`--internal-run-one=`) and $0 may be a relative path.
SELF="$(cd "$(dirname "$0")" 2>/dev/null && pwd)/$(basename "$0")"

REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 1
cd "$REPO_ROOT" || exit 1

# HOW MANY GUARDS RUN AT ONCE (PMAT-1098, row 67-E3).
#
# Bounded, not $(nproc): the clean-room pool is SIXTEEN runners on ONE 32-core
# host (feedback_ci_fleet_is_shared_across_repos), so a per-job -P of 48 is a
# fleet-wide denial of service dressed up as a speed-up. 8 is the cap; a
# smaller box gets its own core count. GUARD_TREE_JOBS overrides for a case
# table that needs a known width.
default_jobs() {
    local n
    n="$(nproc 2>/dev/null)" || n=4
    case "$n" in ''|*[!0-9]*) n=4 ;; esac
    [ "$n" -gt 8 ] && n=8
    [ "$n" -lt 1 ] && n=1
    printf '%d\n' "$n"
}
GUARD_TREE_JOBS="${GUARD_TREE_JOBS:-$(default_jobs)}"
case "$GUARD_TREE_JOBS" in ''|*[!0-9]*|0) GUARD_TREE_JOBS=1 ;; esac

# A bare `cargo ` token: not preceded by a lowercase letter, underscore or
# hyphen (so `sccache`, `rustc-sccache`, `cargo-ci` in a path do not count),
# and followed by a space (so `cargo` alone, e.g. a comment fragment, does
# not count either -- it must look like an invocation).
CARGO_RE='(^|[^a-z_-])cargo '

guard_universe() {
    git ls-files 'scripts/check_*.sh'
}

cargo_free_universe() {
    # xargs -r: never invoke grep with zero files (empty universe).
    guard_universe | xargs -r grep -LE "$CARGO_RE"
}

cargo_only_universe() {
    guard_universe | xargs -r grep -lE "$CARGO_RE"
}

usage() {
    printf 'usage: %s [--list | --dry-run] [--no-cargo | --cargo-only]\n' "$0" >&2
    exit 2
}

list_mode=0
dry_run=0
subset=all
run_one_spec=""
for arg in "$@"; do
    case "$arg" in
        --list) list_mode=1 ;;
        --dry-run) dry_run=1 ;;
        --no-cargo) subset=no-cargo ;;
        --cargo-only) subset=cargo-only ;;
        # INTERNAL, and deliberately unadvertised in usage(): this is the
        # script re-entering itself as ONE guard's worker under the xargs
        # pool. `<index>:<guard path>` -- the index fixes the guard's place in
        # the output, which is decided by the plan and never by finish order.
        --internal-run-one=*) run_one_spec="${arg#*=}" ;;
        *) usage ;;
    esac
done

universe_for_subset() {
    case "$subset" in
        no-cargo) cargo_free_universe ;;
        cargo-only) cargo_only_universe ;;
        all) guard_universe ;;
    esac
}

if [ "$list_mode" -eq 1 ]; then
    universe_for_subset
    exit 0
fi

# advertises_self_test G -- does guard G's --help output mention "self-test"?
#
# A guard that does not understand --help at all just runs its normal body
# (most guards fall through an argv `case` to a usage/die message that itself
# echoes the flags it accepts, per this repo's convention -- see
# check_contract_test_binding.sh's `case "${1:-}" in ... *) die "usage: $0
# [--self-test | --update-baseline]" ;; esac`). Either way this call's
# output is captured and never executed a second time for detection alone.
advertises_self_test() {
    g="$1"
    help_out="$(bash "$g" --help 2>&1)"
    n="$(grep -c -- 'self-test' <<<"$help_out")"
    [ "${n:-0}" -gt 0 ]
}

# ---------------------------------------------------------------------------
# THE WORKER -- ONE GUARD, ITS OWN FILES, NO SHARED STDOUT (PMAT-1098, 67-E3)
#
# Every worker writes its rows to `<idx>.rows` and its verdict to `<idx>.meta`
# in the run directory, and NOTHING to the pool's stdout: concurrent writers on
# one stream interleave, and an interleaved FAIL row with someone else's
# captured output underneath it is worse than no row at all. The parent prints
# the rows afterwards, in plan order.
#
# `.meta` is written LAST and moved into place, so it exists only if this
# worker reached the end. A worker that is killed (OOM, a guard that takes its
# process group with it, a pool that is torn down) leaves rows and no verdict,
# and the parent turns a MISSING verdict into a FAILURE -- fail-closed. The
# one thing a run-all dispatcher may never do is drop a guard quietly.
worker_run_one() {
    spec="$1"
    w_idx="${spec%%:*}"
    w_guard="${spec#*:}"
    w_rows="$GUARD_TREE_RUN_DIR/$w_idx.rows"
    w_meta="$GUARD_TREE_RUN_DIR/$w_idx.meta"
    w_cap="$GUARD_TREE_RUN_DIR/$w_idx.cap"
    w_labels="$GUARD_TREE_RUN_DIR/$w_idx.labels"
    : > "$w_rows"
    : > "$w_labels"
    w_total=0
    w_failed=0

    # Same row shape, same capture-and-indent, as the serial runner it
    # replaces -- only the destination changed.
    worker_row() {
        label="$1"
        shift
        w_total=$((w_total + 1))
        if "$@" >"$w_cap" 2>&1; then
            printf 'PASS  %s\n' "$label" >> "$w_rows"
        else
            w_failed=$((w_failed + 1))
            printf 'FAIL  %s\n' "$label" >> "$w_rows"
            sed 's/^/      | /' "$w_cap" >> "$w_rows"
            printf '%s\n' "$label" >> "$w_labels"
        fi
    }

    if advertises_self_test "$w_guard"; then
        worker_row "$w_guard [self-test]" bash "$w_guard" --self-test
        worker_row "$w_guard [run]" bash "$w_guard"
    else
        worker_row "$w_guard [run]" bash "$w_guard"
    fi
    rm -f "$w_cap"
    printf 'total=%d\nfailed=%d\n' "$w_total" "$w_failed" > "$w_meta.tmp"
    mv -f "$w_meta.tmp" "$w_meta"
    return 0
}

if [ -n "$run_one_spec" ]; then
    if [ -z "${GUARD_TREE_RUN_DIR:-}" ] || [ ! -d "${GUARD_TREE_RUN_DIR:-}" ]; then
        printf '%s: --internal-run-one needs GUARD_TREE_RUN_DIR\n' "$0" >&2
        exit 2
    fi
    worker_run_one "$run_one_spec"
    exit 0
fi

# ---------------------------------------------------------------------------
# ORACLE 1 -- how the workflows themselves invoke each guard.
#
# One awk pass over .github/workflows/*.yml emits one row per INVOCATION:
#
#     <basename>\t(ARG|BARE)\t<workflow file>
#
# ARG means the invocation carried an argument, or sat in a step that declares
# an `env:` block (pr-review-quorum.yml passes PR_NUMBER that way, with no
# argument at all -- an env-only invocation is still an invocation this runner
# cannot reproduce). BARE means neither.
#
# MENTION IS NOT INVOCATION, and this is the same distinction
# check_guards_are_wired.sh had to learn the hard way: `sed 's/#.*$//'` strips
# from the FIRST `#`, so a trailing comment cannot mint a fake invocation, and
# the token must sit in command position -- at line start or after
# whitespace/`;`/`&`/`|`/`(`, optionally behind `bash `, `sh ` or `./`.
#
# A trailing `\` (line continuation) and a leading redirect/pipe/terminator in
# the remainder are NOT arguments and are stripped before the emptiness test.
# Reading `bash scripts/check_x.sh >/dev/null` as "takes arguments" would skip
# a guard that runs perfectly well bare, which is coverage lost silently.
workflow_invocations() {
    [ -d .github/workflows ] || return 0
    awk '
        BEGIN { q = sprintf("%c", 39) }
        FNR == 1 {
            step_env = 0
            nf = split(FILENAME, fp, "/")
            wf = fp[nf]
            re = "(^|[[:space:];&|(])((ba)?sh[[:space:]]+|\\./)?[^[:space:]\"" q "`]*check_[A-Za-z0-9_.-]+\\.sh"
        }
        /^[[:space:]]*-[[:space:]]+(name|run|uses|if|shell|env|with|id):/ { step_env = 0 }
        /^[[:space:]]+env:[[:space:]]*$/ { step_env = 1 }
        {
            line = $0
            sub(/#.*$/, "", line)
            s = line
            while (match(s, re)) {
                tok = substr(s, RSTART, RLENGTH)
                rest = substr(s, RSTART + RLENGTH)
                s = rest
                w = tok
                sub(/^[[:space:];&|(]+/, "", w)
                sub(/^(ba)?sh[[:space:]]+/, "", w)
                sub(/^\.\//, "", w)
                np = split(w, pp, "/")
                base = pp[np]
                sub("^[\"" q "]", "", rest)
                sub(/[[:space:]]+$/, "", rest)
                sub(/\\$/, "", rest)
                sub(/^[[:space:]]+/, "", rest)
                if (rest ~ /^([|;&)>]|2>)/) { rest = "" }
                if (rest != "" || step_env == 1)
                    printf "%s\tARG\t%s\n", base, wf
                else
                    printf "%s\tBARE\t%s\n", base, wf
            }
        }
    ' .github/workflows/*.yml 2>/dev/null
}

INVOCATIONS="$(workflow_invocations)"

# arg_wired_in BASE -- the workflow that wires BASE with arguments/env, when
# EVERY invocation of BASE does. Prints nothing when BASE has a bare
# invocation somewhere, or no invocation at all (an unwired guard is exactly
# what this runner exists to reach).
arg_wired_in() {
    base="$1"
    n_bare="$(awk -F'\t' -v b="$base" '$1 == b && $2 == "BARE"' <<<"$INVOCATIONS" | grep -c .)"
    n_arg="$(awk -F'\t' -v b="$base" '$1 == b && $2 == "ARG"' <<<"$INVOCATIONS" | grep -c .)"
    [ "${n_arg:-0}" -gt 0 ] || return 0
    [ "${n_bare:-0}" -eq 0 ] || return 0
    awk -F'\t' -v b="$base" '$1 == b && $2 == "ARG" { print $3; exit }' <<<"$INVOCATIONS"
}

# wired_elsewhere_in BASE -- the (other) workflow that bare-invokes BASE, when
# ci.yml itself never invokes BASE at all (bare or ARG).
#
# THIS RUNNER IS A STEP INSIDE ci.yml, AND ci.yml IS NOT THE ONLY WORKFLOW.
# book.yml bare-invokes check_book_cli_parity.sh at line 94, after building
# `apr` -- a binary this dispatcher's `--no-cargo` step has no reason to have
# on a bare runner, and ci.yml never names the guard at all. Running it here
# duplicates a guard another workflow already owns and fails it for a reason
# that workflow was written to prevent (no `apr` on PATH). The guard is not
# unwired -- book.yml's own step wires it -- it is wired by a DIFFERENT
# workflow than the one this dispatcher runs inside.
#
# Prints nothing when BASE is invoked in ci.yml at all (bare or ARG -- a
# guard ci.yml also names for itself still runs here), or has no BARE
# invocation anywhere (a guard with only ARG invocations is already the
# arg-wired case above; a guard with no invocation at all is the dark case,
# and BSE-01's intent is that dark guards still run).
wired_elsewhere_in() {
    base="$1"
    n_ci="$(awk -F'\t' -v b="$base" '$1 == b && $3 == "ci.yml"' <<<"$INVOCATIONS" | grep -c .)"
    [ "${n_ci:-0}" -eq 0 ] || return 0
    awk -F'\t' -v b="$base" '$1 == b && $2 == "BARE" { print $3; exit }' <<<"$INVOCATIONS"
}

# ---------------------------------------------------------------------------
# ORACLE 2 -- the release-time registry, read from the guard that owns it.
#
# Absent script => empty registry, which is not a silent hole: this runner then
# RUNS those guards, and check_no_timing_in_required.sh is the thing that turns
# red about it. The two halves cannot both go quiet.
RELEASE_TIME_GUARDS=""
if [ -f scripts/check_no_timing_in_required.sh ]; then
    RELEASE_TIME_GUARDS="$(bash scripts/check_no_timing_in_required.sh --list 2>/dev/null)" \
        || RELEASE_TIME_GUARDS=""
fi

is_release_time() {
    base="$1"
    flat=" $(tr '\n' ' ' <<<"$RELEASE_TIME_GUARDS") "
    case "$flat" in
        *" $base "*) return 0 ;;
    esac
    return 1
}

# ---------------------------------------------------------------------------
# ORACLE 3 -- the shrink-only unwired-guards ledger, read from the guard that
# owns it (PMAT-1062).
#
# scripts/unwired_guards_baseline.txt is check_guards_are_wired.sh's own
# accepted-exemption list: a guard named there is ALREADY KNOWN to be reached
# by no workflow, and that file's ratchet (baseline_ratchet_check, compared
# against merge-base(HEAD, origin/main)) is what forbids the list from
# growing -- so a name can only be here because it was already argued about
# and accepted, never because someone wanted a red guard to stop failing.
# Measured on this branch: all three currently ledgered guards exit non-zero
# run bare (check_book_examples_executable.sh, check_mcp_never_path_resolves_apr.sh,
# check_multiplatform_dogfood.sh) -- a run-all dispatcher running them here
# turns three ALREADY-ACCEPTED gaps into three fresh required-check failures,
# which is not what accepting the gap meant.
#
# Absent file => empty ledger, which is not a silent hole: this runner then
# RUNS those guards, and check_guards_are_wired.sh is the thing whose ratchet
# turns red about a guard that is unwired and unledgered.
UNWIRED_BASELINE_GUARDS=""
if [ -f scripts/unwired_guards_baseline.txt ]; then
    UNWIRED_BASELINE_GUARDS="$(grep -vE '^[[:space:]]*(#|$)' scripts/unwired_guards_baseline.txt 2>/dev/null)" \
        || UNWIRED_BASELINE_GUARDS=""
fi

is_unwired_baseline() {
    base="$1"
    flat=" $(tr '\n' ' ' <<<"$UNWIRED_BASELINE_GUARDS") "
    case "$flat" in
        *" $base "*) return 0 ;;
    esac
    return 1
}

# skip_reason GUARD -- why this runner must not run GUARD bare, or empty.
skip_reason() {
    base="$(basename "$1")"
    if is_release_time "$base"; then
        printf 'release-time (check_no_timing_in_required)\n'
        return 0
    fi
    if is_unwired_baseline "$base"; then
        printf 'unwired-baseline (shrink-only ledger)\n'
        return 0
    fi
    wf="$(arg_wired_in "$base")"
    if [ -n "$wf" ]; then
        printf 'wired-with-args in %s\n' "$wf"
        return 0
    fi
    wf="$(wired_elsewhere_in "$base")"
    if [ -n "$wf" ]; then
        printf 'wired-elsewhere %s\n' "$wf"
    fi
}

# advertises_self_test and the worker are defined above, beside the
# --internal-run-one dispatch they belong to.

guards="$(universe_for_subset)"

RUN_DIR="$(mktemp -d)" || exit 1
trap 'rm -rf "${RUN_DIR:?}"' EXIT

total=0
failed=0
fail_rows=""
skipped=0
to_run=0

# ---------------------------------------------------------------------------
# 1. THE PLAN -- built serially, in universe order, and it is what fixes the
#    ORDER OF THE OUTPUT. The pool below finishes in whatever order the host
#    decides; nothing downstream ever learns that order.
#
#    One line per guard: `SKIP<TAB><guard><TAB><reason>` or `RUN<TAB><guard>`.
#    skip_reason reads three oracles and costs nothing worth parallelising.
PLAN="$RUN_DIR/plan"
: > "$PLAN"
while IFS= read -r g; do
    [ -n "$g" ] || continue
    reason="$(skip_reason "$g")"
    if [ -n "$reason" ]; then
        printf 'SKIP\t%s\t%s\n' "$g" "$reason" >> "$PLAN"
    else
        printf 'RUN\t%s\t\n' "$g" >> "$PLAN"
    fi
done <<<"$guards"

TAB="$(printf '\t')"

if [ "$dry_run" -eq 1 ]; then
    while IFS="$TAB" read -r kind g reason; do
        case "$kind" in
            SKIP) skipped=$((skipped + 1)); printf 'skipped: %s -- %s\n' "$g" "$reason" ;;
            RUN)  to_run=$((to_run + 1));   printf 'run: %s\n' "$g" ;;
        esac
    done < "$PLAN"
    printf '%d to run, %d skipped\n' "$to_run" "$skipped"
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. THE POOL. `<index>:<guard>` per line, one worker per line, -P at a time.
#    The index is the guard's LINE NUMBER IN THE PLAN, so a worker's files are
#    addressable by the parent without any communication back from the pool.
WORKLIST="$RUN_DIR/worklist"
: > "$WORKLIST"
idx=0
while IFS="$TAB" read -r kind g reason; do
    idx=$((idx + 1))
    [ "$kind" = RUN ] || continue
    printf '%d:%s\n' "$idx" "$g" >> "$WORKLIST"
done < "$PLAN"

if [ -s "$WORKLIST" ]; then
    # `-I{}` makes xargs split on newlines only (a guard path never contains
    # one) and pass exactly one spec per process. The pool's own exit status is
    # deliberately NOT the verdict: a worker exits 0 whatever its guard did,
    # and a worker that did not exit at all is judged by its missing .meta
    # below. Reading a verdict off xargs would make "an argument failed" and
    # "a guard failed" the same sentence.
    GUARD_TREE_RUN_DIR="$RUN_DIR" xargs -a "$WORKLIST" -r -I{} -P "$GUARD_TREE_JOBS" \
        bash "$SELF" "--internal-run-one={}" || true
fi

# ---------------------------------------------------------------------------
# 3. THE OUTPUT -- plan order, one guard's rows at a time, never interleaved.
idx=0
while IFS="$TAB" read -r kind g reason; do
    idx=$((idx + 1))
    if [ "$kind" = SKIP ]; then
        skipped=$((skipped + 1))
        printf 'skipped: %s -- %s\n' "$g" "$reason"
        continue
    fi
    rows="$RUN_DIR/$idx.rows"
    meta="$RUN_DIR/$idx.meta"
    g_total=""
    g_failed=""
    if [ -f "$meta" ]; then
        g_total="$(sed -n 's/^total=//p' "$meta" | head -1)"
        g_failed="$(sed -n 's/^failed=//p' "$meta" | head -1)"
    fi
    case "${g_total:-}" in ''|*[!0-9]*) g_total="" ;; esac
    case "${g_failed:-}" in ''|*[!0-9]*) g_failed="" ;; esac
    if [ -z "$g_total" ] || [ -z "$g_failed" ]; then
        # FAIL-CLOSED. No verdict means this guard did not run to completion,
        # and a guard that cannot run is a FAILURE -- never a skip, never a
        # row that quietly is not there. (A dispatcher that counted only the
        # verdicts it found would report "N-1 checks, 0 failed" and go green.)
        total=$((total + 1))
        failed=$((failed + 1))
        printf 'FAIL  %s [run]\n' "$g"
        printf '      | guard_tree: no verdict was written for this guard -- its worker did not finish.\n'
        [ -s "$rows" ] && sed 's/^/      | /' "$rows"
        fail_rows="${fail_rows}${g} [run]
"
        continue
    fi
    [ -f "$rows" ] && cat "$rows"
    total=$((total + g_total))
    # ONE INCREMENT PER FAILING CHECK, and deliberately not `failed + g_failed`.
    # scripts/tests/guard_tree_test.sh proves this runner is not fail-fast by
    # MUTATING it into a fail-fast one -- `sed 's/failed=$((failed + 1))/&; exit
    # 1/'` -- and then demanding the mutant miss the second failing guard. A
    # summed accumulator has no such seam, so the mutation would stop
    # discriminating and that row would go green for a runner nobody proved
    # anything about. The seam is load-bearing; keep the shape.
    n_left="$g_failed"
    while [ "$n_left" -gt 0 ]; do
        failed=$((failed + 1))
        n_left=$((n_left - 1))
    done
    if [ -s "$RUN_DIR/$idx.labels" ]; then
        while IFS= read -r lbl; do
            [ -n "$lbl" ] || continue
            fail_rows="${fail_rows}${lbl}
"
        done < "$RUN_DIR/$idx.labels"
    fi
done < "$PLAN"

printf 'dispatch: up to %d guard(s) at a time (GUARD_TREE_JOBS)\n' "$GUARD_TREE_JOBS"
printf '%d guard(s) skipped\n' "$skipped"
printf '%d checks, %d failed\n' "$total" "$failed"
if [ "$failed" -gt 0 ]; then
    printf 'FAILED:\n%s' "$fail_rows" >&2
    exit 1
fi
exit 0
