#!/usr/bin/env bash
# run_bench_cells.sh - drive the 40 benchmark cells of ONE method, one process per cell.
#
#   bash scripts/run_bench_cells.sh METHOD BENCH_DIR DATA_DIR [MODEL_OR_BASE]
#
#   METHOD          setfit | lora
#   BENCH_DIR       where rows, locks, ledgers and run-manifest.json live
#   DATA_DIR        the attested benchmark directory `apr data tweet-eval-stance` wrote
#   MODEL_OR_BASE   setfit: the pinned all-MiniLM-L6-v2 checkout (--model-dir)
#                   lora:   the base model .apr (--base-model)
#
# ---------------------------------------------------------------------------
# ONE PROCESS PER CELL, AND DELIBERATELY NO --jobs
# ---------------------------------------------------------------------------
#
# Cells run SEQUENTIALLY and there is no parallel-dispatch flag, on purpose.
# Concurrent cell processes would do two bad things at once: race the shared
# run-manifest, and invalidate every EVAL-05 resource number by contending for
# CPU and memory. A peak-RSS figure taken while three siblings were resident is
# not this cell's peak, and a throughput taken under contention is not this
# cell's throughput. If wall-clock is the problem, that is a scheduling decision
# for a human, never a parallelisation of a measurement.
#
# Isolation is per-PROCESS rather than per-thread for the same reason the cold
# probe is a child: a process that has already run a cell is warm, and the next
# cell's train-peak measurement would start from its predecessor's arenas.
#
# ---------------------------------------------------------------------------
# HALT ON THE FIRST EVIDENCE-CLASS FAILURE
# ---------------------------------------------------------------------------
#
# 05-03's coverage claim is deliberately only "membership covers all 40 cells;
# empirical threshold success remains a per-run invariant". The only thing that
# keeps that claim honest is the FIRST empirical failure ending the run instead
# of being buried in an end-of-run tally that a reader skims. So an evidence
# failure - UncalibratedRegime, an evidence-threshold refusal, a lock or
# attestation refusal, a digest failure - stops everything, prints the refusal,
# and exits EXIT_EVIDENCE.
#
# A transient/infrastructure failure (ENOSPC, an ssh drop, an OOM kill) also
# stops, with EXIT_TRANSIENT, because the operator needs to know that resuming
# is SAFE. Conflating the two would make "just re-run it" the advice for a
# finding that no re-run can fix.
#
# ---------------------------------------------------------------------------
# rc IS NEVER READ THROUGH A PIPE
# ---------------------------------------------------------------------------
#
# Every `rc` is captured with `|| rc=$?` on the command it describes, and
# nothing in this script pipes a command whose status it then reads. CLAUDE.md
# Verification Discipline rule 1: that defect has shipped twice in this
# repository (#2336 captured tee's status; #2360 captured grep's), and both
# times the result was a gate that could not fail while printing success.
#
# The `||` is load-bearing under `set -e`. A bare `rc=$?` on the NEXT line never
# runs: `set -e` exits the shell on the failing command itself, taking the whole
# refusal-classification path with it. That form shipped here and made the
# EXIT_EVIDENCE / EXIT_TRANSIENT distinction below unreachable.

set -euo pipefail

# The pin. Sourced, so it stays option-neutral and reports by return status.
. scripts/apr_bin.sh || exit 1

# Incremental compilation regrew target/debug/incremental to 90 GB and filled
# the volume three times in this milestone. A long cell sweep is exactly the
# workload that does it.
export CARGO_INCREMENTAL=0

# ---- Exit codes, distinct so the operator knows whether a resume is safe ----
readonly EXIT_USAGE=2
readonly EXIT_EVIDENCE=3
readonly EXIT_TRANSIENT=4
readonly EXIT_LOCKED=5

# ---- The contracted matrix. Byte-equal to setfit-benchmark-claims-v1. ------
#
# ARRAYS rather than space-separated strings: an unquoted string relies on word
# splitting, which a later `set -f` or a stray IFS change silently turns into a
# single 40-character "shot count". The CLI refuses that, so it would surface as
# a confusing refusal rather than a wrong run — but a matrix definition should
# not depend on a global for its arity.
#
# Declared and marked read-only in two statements: `readonly NAME=(...)` is a
# syntax bashrs refuses (SC2163) and whose behaviour differs across bash
# versions.
SHOTS=(8 16 32 64)
SEEDS=(13 17 23 29 31 37 41 43 47 53)
readonly SHOTS
readonly SEEDS

usage() {
    printf 'usage: %s METHOD BENCH_DIR DATA_DIR [MODEL_OR_BASE]\n' "$0" >&2
    printf '  METHOD is setfit or lora\n' >&2
}

if [ "$#" -lt 3 ]; then
    usage
    exit "$EXIT_USAGE"
fi

METHOD="$1"
BENCH_DIR="$2"
DATA_DIR="$3"
MODEL_OR_BASE="${4:-}"

case "$METHOD" in
    setfit|lora) ;;
    *)
        printf 'unknown method %s (expected setfit or lora)\n' "$METHOD" >&2
        exit "$EXIT_USAGE"
        ;;
esac

if [ ! -d "$DATA_DIR" ]; then
    printf 'DATA_DIR %s is not a directory\n' "$DATA_DIR" >&2
    exit "$EXIT_USAGE"
fi

if [ -z "$MODEL_OR_BASE" ]; then
    printf 'MODEL_OR_BASE is required: the encoder checkout for setfit, the base .apr for lora\n' >&2
    exit "$EXIT_USAGE"
fi

mkdir -p "$BENCH_DIR"

# ---------------------------------------------------------------------------
# CANONICAL PATHS, resolved once
# ---------------------------------------------------------------------------
#
# Every path this script builds is rooted in one of these two, so resolving
# both to an absolute canonical path here means no later write or delete can be
# handed a `..` segment it did not intend. Doing it once also removes the only
# reason a per-use guard would be needed.
#
# (The wording avoids the word "form" on purpose: bashrs 6.66.3's IDEM002 rule
# matches the substring `rm` and fired on "form" inside this comment. Measured
# with a two-sided control - restoring the word reproduces the warning, and
# nothing else in this file changed. Recorded rather than suppressed with a
# flag, because a suppression here would also hide a real `rm`.)
BENCH_DIR=$(cd "$BENCH_DIR" && pwd -P)
DATA_DIR=$(cd "$DATA_DIR" && pwd -P)
readonly BENCH_DIR
readonly DATA_DIR

# ---------------------------------------------------------------------------
# SINGLE WRITER. One driver per bench directory, enforced by a pid lock.
# ---------------------------------------------------------------------------
#
# `set -o noclobber` plus a redirect is the lock primitive: the shell opens with
# O_EXCL, so creation is ATOMIC and a test-then-create window two drivers could
# both pass does not exist. A STALE lock is REPORTED, never silently stolen - a
# driver that broke another's lock because the pid looked dead would be doing
# exactly the concurrent write this lock exists to prevent, on the one occasion
# the pid check was wrong.
#
# The subshell scopes `noclobber` to the one redirect: leaving it set would make
# every later `>` in this script fail, which is how a lock idiom becomes a
# mysterious mid-run failure.
LOCK_FILE="$BENCH_DIR/.driver.lock"
readonly LOCK_FILE
if ! (set -o noclobber; printf '%s\n' "$$" > "$LOCK_FILE") 2>/dev/null; then
    holder="(unknown)"
    if [[ -f "$LOCK_FILE" ]]; then
        read -r holder < "$LOCK_FILE" || holder="(unreadable)"
    fi
    printf 'REFUSED: %s is held by pid %s.\n' "$LOCK_FILE" "$holder" >&2
    printf '  Two drivers writing one bench directory would race run-manifest.json AND\n' >&2
    printf '  invalidate every resource measurement by contending for CPU and memory.\n' >&2
    printf '  If that pid is gone, delete the lock by hand after confirming so.\n' >&2
    exit "$EXIT_LOCKED"
fi

cleanup() {
    unlink "$LOCK_FILE" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Selection manifests: one per (shots, seed), generated once and reused.
# ---------------------------------------------------------------------------
#
# THE SAME FILE FEEDS BOTH METHODS. That is EVAL-02's identical-sampled-ID
# guarantee: it is not that two generators agree, it is that there is one file.
# So a manifest that already exists is REUSED rather than regenerated - a
# regenerated manifest is a new draw even when the seed is the same, and the
# lora run would then be paired against rows the setfit run never saw.
generate_selections() {
    mkdir -p "$BENCH_DIR/selections"
    for shots in "${SHOTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Built by expansion rather than by a helper called in a command
            # substitution: forty subshells to format forty strings is a fork per
            # cell for nothing.
            #
            # `apr data select --output` takes a DIRECTORY and writes
            # `selection-manifest.json` inside it - MEASURED at
            # data_contrastive.rs's `output.unwrap_or(data).join(...)`, not
            # assumed from the flag's name. Passing a file path would have
            # produced `.../s8-seed13.json/selection-manifest.json` and left
            # every `--selection` in this script pointing at nothing.
            target_dir="$BENCH_DIR/selections/s${shots}-seed${seed}"
            target="$target_dir/selection-manifest.json"
            if [[ -f "$target" ]]; then
                continue
            fi
            mkdir -p "$target_dir"
            # `|| rc=$?` for the same reason as run_one_cell: under `set -e` a
            # bare `rc=$?` on the next line never runs, so the EXIT_TRANSIENT
            # branch below was unreachable and the operator got apr's raw status
            # instead of the resume-is-safe signal.
            rc=0
            "$APR" data select --data "$DATA_DIR" --shots "$shots" --seed "$seed" \
                --output "$target_dir" || rc=$?
            if [[ "$rc" -ne 0 ]]; then
                printf 'FAIL selection s%s/seed%s (apr data select exited %s)\n' \
                    "$shots" "$seed" "$rc" >&2
                exit "$EXIT_TRANSIENT"
            fi
            if [[ ! -f "$target" ]]; then
                printf 'FAIL selection s%s/seed%s (apr data select exited 0 but wrote no %s)\n' \
                    "$shots" "$seed" "$target" >&2
                exit "$EXIT_TRANSIENT"
            fi
        done
    done
}

# ---------------------------------------------------------------------------
# Resume: HASH-BASED, never a bare filename check.
# ---------------------------------------------------------------------------
#
# A row file that exists is not evidence that the cell completed: an
# interrupted run could have left one (it cannot, because rows are written by
# atomic rename - but the resume logic must not DEPEND on that, or it becomes
# the one place a future non-atomic write goes unnoticed). So a cell is skipped
# only when the run manifest records a digest for it AND the row file on disk
# carries that same digest.
cell_is_complete() {
    local method="$1"
    local shots="$2"
    local seed="$3"
    local row="$BENCH_DIR/rows/${method}-s${shots}-seed${seed}.json"
    local manifest="$BENCH_DIR/run-manifest.json"
    local line rest row_hash

    if [[ ! -f "$row" ]]; then
        return 1
    fi
    if [[ ! -f "$manifest" ]]; then
        return 1
    fi

    # The row's own envelope digest, read with the shell rather than through a
    # pipeline. Two reasons, and neither is style: a `grep | sed` pipeline is a
    # place where somebody later reads `$?` and gets sed's status (the defect
    # that shipped twice here), and it forks two processes per cell for a value
    # that is on the first few lines of the file.
    row_hash=""
    while IFS= read -r line; do
        case "$line" in
            *'"semantic_hash"'*)
                rest="${line#*: \"}"
                row_hash="${rest%%\"*}"
                break
                ;;
            *) ;;
        esac
    done < "$row"
    if [[ -z "$row_hash" ]]; then
        return 1
    fi

    # And the manifest must record exactly that digest. A manifest that names a
    # DIFFERENT digest for this cell is not a resume - it is the collision
    # `RunManifest::record` refuses, and re-running is how the operator sees it.
    while IFS= read -r line; do
        case "$line" in
            *"$row_hash"*) return 0 ;;
            *) ;;
        esac
    done < "$manifest"
    return 1
}

# Classify a cell's failure so the driver's exit code tells the operator
# whether a resume is safe. The EVIDENCE class is what 05-03's coverage claim
# depends on stopping the run.
is_evidence_failure() {
    log="$1"
    if grep -qE 'UncalibratedRegime|evidence table|threshold|selection lock|attestation|digest mismatch|already recorded|outside the contracted matrix|candidate' "$log"; then
        return 0
    fi
    return 1
}

run_one_cell() {
    # `local` throughout, matching cell_is_complete: these names collide with the
    # caller's sweep loop variables, and an unqualified assignment here would
    # silently rewrite the loop's own `shots`/`seed` mid-iteration.
    local shots seed selection log model_flag rc
    shots="$1"
    seed="$2"
    # The SAME path `generate_selections` wrote. Built by expansion in both
    # places rather than by a shared helper called in a command substitution;
    # the driver-gate test asserts the two spellings agree.
    selection="$BENCH_DIR/selections/s${shots}-seed${seed}/selection-manifest.json"
    log="$BENCH_DIR/logs/${METHOD}-s${shots}-seed${seed}.log"
    mkdir -p "$BENCH_DIR/logs"

    # The two methods differ in exactly one token: the flag naming the model.
    # Spelling the whole invocation twice means every future flag has to be added
    # twice, and a one-sided edit yields a defect the other method never sees.
    if [ "$METHOD" = "setfit" ]; then
        model_flag="--model-dir"
    else
        model_flag="--base-model"
    fi
    # `|| rc=$?`, NOT a bare `rc=$?` on the next line. This script runs under
    # `set -e`, and `set -e` fires on the FAILING invocation itself: the shell
    # exits right there, so a following `rc=$?` is unreachable and every line
    # below it - the refusal dump, is_evidence_failure, EXIT_EVIDENCE vs
    # EXIT_TRANSIENT - is dead code. The operator got apr's raw status and no
    # classification at all. A command on the left of `||` is exempt from
    # `set -e`, and `$?` inside the right-hand side is that command's status.
    rc=0
    "$APR" setfit bench run \
        --method "$METHOD" --shots "$shots" --seed "$seed" \
        --data "$DATA_DIR" --selection "$selection" \
        --bench-dir "$BENCH_DIR" "$model_flag" "$MODEL_OR_BASE" \
        > "$log" 2>&1 || rc=$?

    if [ "$rc" -eq 0 ]; then
        printf 'PASS %s s%s seed%s\n' "$METHOD" "$shots" "$seed"
        return 0
    fi

    printf 'FAIL %s s%s seed%s (exit %s)\n' "$METHOD" "$shots" "$seed" "$rc" >&2
    printf '--- refusal ---\n' >&2
    cat "$log" >&2
    printf '--- end refusal ---\n' >&2

    if is_evidence_failure "$log"; then
        printf 'HALT: evidence-class failure. The remaining cells are NOT run.\n' >&2
        printf '  05-03 covers MEMBERSHIP of all 40 cells; empirical threshold success is a\n' >&2
        printf '  per-run invariant, and the only thing that keeps that honest is this stop.\n' >&2
        printf '  Re-running will not fix it: diagnose the refusal above.\n' >&2
        exit "$EXIT_EVIDENCE"
    fi

    printf 'HALT: transient/infrastructure failure. RESUMING IS SAFE once it is fixed:\n' >&2
    printf '  completed cells are skipped by digest, so re-invoking this script continues.\n' >&2
    exit "$EXIT_TRANSIENT"
}

# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

printf 'bench driver: method=%s bench-dir=%s\n' "$METHOD" "$BENCH_DIR"
printf 'apr: %s\n' "$APR"

generate_selections

# `declare -i` so the counters increment WITHOUT an arithmetic expansion in the
# loop body. Same arithmetic, one less construct a reader has to distinguish
# from a command substitution.
declare -i executed=0
declare -i skipped=0
declare -i total=0
for shots in "${SHOTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if cell_is_complete "$METHOD" "$shots" "$seed"; then
            printf 'SKIP %s s%s seed%s (row digest matches the manifest)\n' \
                "$METHOD" "$shots" "$seed"
            skipped+=1
            continue
        fi
        run_one_cell "$shots" "$seed"
        executed+=1
    done
done

total=executed+skipped
printf 'DONE %s: %s executed, %s skipped, %s of 40 cells covered\n' \
    "$METHOD" "$executed" "$skipped" "$total"

# The vacuity floor. A loop that covered nothing exits 0 and prints a tally of
# zeroes; CR-02 is the precedent (a zero-match libtest filter exits 0 while
# printing "test result: ok"). So the count is asserted, not reported.
if [ "$total" -ne 40 ]; then
    printf 'FAIL: %s cells covered, expected 40. The matrix loop did not cover the\n' "$total" >&2
    printf '      contracted set, so this run proves nothing about completeness.\n' >&2
    exit "$EXIT_EVIDENCE"
fi
