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
# TWO KINDS OF GUARD THIS RUNNER MUST NOT RUN BARE
# -------------------------------------------------
# "Run every guard" is only true of guards a bare invocation can run at all,
# and there are two populations for which it is false. Both are DERIVED from
# the oracle that already owns the answer -- neither is a name list here,
# because a second hand-maintained list is the defect this whole file exists
# to avoid.
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

REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 1
cd "$REPO_ROOT" || exit 1

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
for arg in "$@"; do
    case "$arg" in
        --list) list_mode=1 ;;
        --dry-run) dry_run=1 ;;
        --no-cargo) subset=no-cargo ;;
        --cargo-only) subset=cargo-only ;;
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

# skip_reason GUARD -- why this runner must not run GUARD bare, or empty.
skip_reason() {
    base="$(basename "$1")"
    if is_release_time "$base"; then
        printf 'release-time (check_no_timing_in_required)\n'
        return 0
    fi
    wf="$(arg_wired_in "$base")"
    if [ -n "$wf" ]; then
        printf 'wired-with-args in %s\n' "$wf"
    fi
}

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

TMP_OUT="$(mktemp)" || exit 1
trap 'rm -f "$TMP_OUT"' EXIT

total=0
failed=0
fail_rows=""

run_row() {
    # $1 = row label, remaining args = the command to run
    label="$1"
    shift
    total=$((total + 1))
    if "$@" >"$TMP_OUT" 2>&1; then
        printf 'PASS  %s\n' "$label"
    else
        failed=$((failed + 1))
        fail_rows="${fail_rows}${label}
"
        printf 'FAIL  %s\n' "$label"
        sed 's/^/      | /' "$TMP_OUT"
    fi
}

guards="$(universe_for_subset)"
skipped=0
to_run=0

while IFS= read -r g; do
    [ -n "$g" ] || continue
    reason="$(skip_reason "$g")"
    if [ -n "$reason" ]; then
        skipped=$((skipped + 1))
        printf 'skipped: %s -- %s\n' "$g" "$reason"
        continue
    fi
    if [ "$dry_run" -eq 1 ]; then
        to_run=$((to_run + 1))
        printf 'run: %s\n' "$g"
        continue
    fi
    if advertises_self_test "$g"; then
        run_row "$g [self-test]" bash "$g" --self-test
        run_row "$g [run]" bash "$g"
    else
        run_row "$g [run]" bash "$g"
    fi
done <<<"$guards"

if [ "$dry_run" -eq 1 ]; then
    printf '%d to run, %d skipped\n' "$to_run" "$skipped"
    exit 0
fi

printf '%d guard(s) skipped\n' "$skipped"
printf '%d checks, %d failed\n' "$total" "$failed"
if [ "$failed" -gt 0 ]; then
    printf 'FAILED:\n%s' "$fail_rows" >&2
    exit 1
fi
exit 0
