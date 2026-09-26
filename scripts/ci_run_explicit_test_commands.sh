#!/usr/bin/env bash
# ci_run_explicit_test_commands.sh -- run ci/explicit-test-commands.d/ (PMAT-3313).
#
# The workspace-test "Integration tests" step used to be ONE physical ci.yml line,
# `bash -c 'cmd && cmd && ...'` (~4000 chars), and every PR adding a test target
# edited it, so any two such PRs conflicted. A single one-command-per-line file
# only MOVED that lock: two PRs appending at end-of-file still conflict. So the
# commands are FRAGMENTS, the shape of docs/roadmaps/entries/: one file per command,
#
#   ci/explicit-test-commands.d/NNN-<slug>.cmd
#
# NNN is a zero-padded, GAPPED ordinal (010, 020, ...) that fixes execution order;
# a new command takes a free ordinal between two others without renumbering, and
# two PRs adding different files never touch the same path.
#
# Parsing (--list), shared by the CI step and scripts/check_explicit_test_commands.sh:
#   * every entry of the directory must be named ^[0-9]{3}-[a-z0-9-]+\.cmd$
#   * files are read in `LC_ALL=C sort` order
#   * each file holds EXACTLY ONE non-comment, non-blank line (trimmed)
#   * no two files share an ordinal (ambiguous order); no command appears twice
#   * a missing or empty directory is refused -- zero commands is broken wiring
#   Any violation: rc 2, nothing is printed on stdout, nothing runs.
#
# Execution (--run):
#   * commands in order, each as `bash -c "$cmd"` with stdin from /dev/null (the
#     old docker run had no -i; the list is fully read before anything runs)
#   * a ::group:: header per command
#   * NO FAIL-FAST (PMAT-3587). Every command runs. A red run still measures
#     everything else it was going to measure, which is exactly when the data is
#     most wanted. `--fail-fast` restores the old stop-at-first behaviour.
#   * a RESULT line always: `RESULT ran=N skipped=S total=T failed=F`. The skipped
#     count is a NUMBER, never an absence -- an absence reads as "nothing to
#     report", which is indistinguishable from "passed" to every consumer.
#
# EXIT CODE -- the runner's own vocabulary, never a passthrough of a command's rc
# (propagating one would let a test exiting 2 mean "commands were not run"):
#   0  every command ran, none failed
#   1  every command ran, F >= 1 FAILED      -- a measured failure
#   2  UNMEASURED commands exist             -- Unknown, doctrine 4; also every
#      refusal (bad tree, bad shard, vacuity)
#
# Usage:
#   scripts/ci_run_explicit_test_commands.sh --list [DIR]
#   scripts/ci_run_explicit_test_commands.sh --run  [DIR] [--shard N/M] [--fail-fast]
#   scripts/ci_run_explicit_test_commands.sh --run  [DIR] --workers N [--weights F]
#   scripts/ci_run_explicit_test_commands.sh --plan [DIR] --buckets B --weights F
# DIR defaults to ci/explicit-test-commands.d. The case table for this script
# lives in scripts/check_explicit_test_commands.sh --self-test.
set -euo pipefail

DEFAULT_DIR="ci/explicit-test-commands.d"
NAME_RE='^[0-9]{3}-[a-z0-9-]+\.cmd$'

# one_command FILE -> the file's single command on stdout; rc 1 with a message otherwise.
one_command() {
    local file=$1 raw line n=0 found=""
    while IFS= read -r raw || [ -n "$raw" ]; do
        line="${raw#"${raw%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -n "$line" ] || continue
        case "$line" in \#*) continue ;; esac
        n=$((n + 1)); found=$line
    done < "$file"
    if [ "$n" -ne 1 ]; then
        printf 'REFUSE %s holds %s command line(s); a fragment holds exactly 1\n' "$file" "$n" >&2
        return 1
    fi
    printf '%s\n' "$found"
}

# parse DIR -> the ordered commands on stdout; rc 2 on any violation (stdout then empty).
parse() {
    local dir=$1 base cmd bad=0 ord
    local -a names=() cmds=()
    local -A seen_ord=() seen_cmd=()
    if [ ! -d "$dir" ]; then
        printf 'ENV   %s: no such directory -- refusing to run zero commands\n' "$dir" >&2
        return 2
    fi
    mapfile -t names < <(find "$dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
    if [ "${#names[@]}" -eq 0 ]; then
        printf 'ENV   %s is EMPTY -- zero commands is broken wiring, not a pass\n' "$dir" >&2
        return 2
    fi
    for base in "${names[@]}"; do
        if ! grep -qE "$NAME_RE" <<< "$base" || [ ! -f "$dir/$base" ]; then
            printf 'REFUSE %s/%s: not a regular file named NNN-<slug>.cmd (%s)\n' "$dir" "$base" "$NAME_RE" >&2
            bad=1; continue
        fi
        ord=${base%%-*}
        if [ -n "${seen_ord[$ord]:-}" ]; then
            printf 'REFUSE ordinal %s is shared by %s and %s -- ambiguous order; take a free ordinal\n' "$ord" "${seen_ord[$ord]}" "$base" >&2
            bad=1
        fi
        seen_ord[$ord]=$base
        cmd=$(one_command "$dir/$base") || { bad=1; continue; }
        if [ -n "${seen_cmd[$cmd]:-}" ]; then
            printf 'REFUSE the same command is in %s and %s: %s\n' "${seen_cmd[$cmd]}" "$base" "$cmd" >&2
            bad=1
        fi
        seen_cmd[$cmd]=$base
        cmds+=("$cmd")
    done
    [ "$bad" -eq 0 ] || return 2
    printf '%s\n' "${cmds[@]}"
}


# plan DIR B WEIGHTS -> one line per fragment, in list order:
#   <bucket>\t<seconds>\t<basename>
# then one `LOAD <bucket> <seconds>` line per bucket.
# Y2 (#4424, operator via cop 2026-09-25: "over-partition (3N buckets,
# work-steal), not weight-by-host"): the fragments are cut into B = 3N buckets
# for N local workers, and a worker that finishes early STEALS the next
# unclaimed bucket (see pool). The weights only shape the buckets -- heaviest
# first, each to the least-loaded bucket (lowest index on a tie) -- they never
# pin work to a host, so a stale weight costs one bucket's imbalance, not a
# whole shard's. Deterministic: the same tree and weights give the same buckets.
# A fragment with no weight is planned at the median of the weights that do
# apply, so a new fragment needs no edit here. rc 2 on a malformed weights line,
# a duplicate weight, or no applicable weight at all (a plan from zero
# measurements is round-robin in disguise). A weight naming no fragment is
# STALE on stderr, not a refusal: deleting a fragment must not break every PR.
plan() {
    local dir=$1 m=$2 wfile=$3 out base line w j best med n
    local -a names=() load=() sorted=()
    local -A weight=() bucket_of=() present=()
    out=$(parse "$dir") || return $?
    mapfile -t names < <(find "$dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
    for base in "${names[@]}"; do present[$base]=1; done
    if [ ! -f "$wfile" ]; then
        printf 'ENV   weights file %s: no such file\n' "$wfile" >&2; return 2
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|\#*) continue ;; esac
        if ! [[ "$line" =~ ^([0-9]{3}-[a-z0-9-]+\.cmd)$'\t'([0-9]+)$ ]]; then
            printf 'REFUSE %s: malformed weight line (want <NNN-slug.cmd><TAB><seconds>): %s\n' "$wfile" "$line" >&2
            return 2
        fi
        base=${BASH_REMATCH[1]}; w=${BASH_REMATCH[2]}
        if [ -n "${weight[$base]:-}" ]; then
            printf 'REFUSE %s: %s is weighted twice\n' "$wfile" "$base" >&2; return 2
        fi
        weight[$base]=$w
        [ -n "${present[$base]:-}" ] || printf 'STALE %s: %s names no fragment in %s\n' "$wfile" "$base" "$dir" >&2
    done < "$wfile"
    mapfile -t sorted < <(for base in "${names[@]}"; do [ -n "${weight[$base]:-}" ] && printf '%s\n' "${weight[$base]}"; done | sort -n)
    n=${#sorted[@]}
    if [ "$n" -eq 0 ]; then
        printf 'REFUSE %s weights none of the %s fragment(s) -- a plan from zero measurements is not a plan\n' "$wfile" "${#names[@]}" >&2
        return 2
    fi
    med=${sorted[$(( (n - 1) / 2 ))]}
    for ((j = 0; j < m; j++)); do load+=(0); done
    while IFS=$'\t' read -r w base; do
        best=0
        for ((j = 1; j < m; j++)); do
            [ "${load[$j]}" -lt "${load[$best]}" ] && best=$j
        done
        load[best]=$(( load[best] + w ))
        bucket_of[$base]=$(( best + 1 ))
    done < <(for base in "${names[@]}"; do printf '%s\t%s\n' "${weight[$base]:-$med}" "$base"; done | LC_ALL=C sort -t$'\t' -k1,1nr -k2,2)
    for base in "${names[@]}"; do
        printf '%s\t%s\t%s\n' "${bucket_of[$base]}" "${weight[$base]:-$med}" "$base"
    done
    for ((j = 0; j < m; j++)); do printf 'LOAD %s %s\n' "$((j + 1))" "${load[$j]}"; done
}

# pool WORKERS -- runs the global cmds[] (bucket_of[] parallel to it, buckets
# 1..NB) with WORKERS local workers pulling from a claim directory: a bucket is
# claimed by `mkdir`, which is atomic, so it runs on exactly ONE worker, and a
# worker that finishes early takes the next unclaimed bucket -- the work-steal.
# Each command's output goes to its own log, replayed afterwards in list order
# under its ::group:: header, so parallel output never interleaves. Σ belt:
# every bucket must end claimed and every command must leave exactly one rc
# file; anything missing is UNMEASURED (rc 2), never passed.
pool() {
    local workers=$1 cdir k b w rc
    cdir=$(mktemp -d "${TMPDIR:-/tmp}/explicit-pool.XXXXXX")
    for ((w = 1; w <= workers; w++)); do
        (
            for ((b = 1; b <= NB; b++)); do
                mkdir "$cdir/claim.$b" 2> /dev/null || continue
                printf '%s\n' "$w" > "$cdir/claim.$b/by"
                for ((k = 0; k < ${#cmds[@]}; k++)); do
                    [ "${bucket_of[$k]}" -eq "$b" ] || continue
                    rc=0
                    bash -c "${cmds[$k]}" < /dev/null > "$cdir/out.$k" 2>&1 || rc=$?
                    printf '%s\n' "$rc" > "$cdir/rc.$k"
                done
            done
        ) &
    done
    wait
    POOL_DIR=$cdir
}

# run DIR [N/M] [FF] [WEIGHTS] [WORKERS] -- runs the commands in order; with N/M
# only the N-th shard's share (every M-th command from the N-th, round-robin),
# so M shards cover the list exactly once between them (PACK-001). With
# WORKERS > 1 the commands are over-partitioned into 3*WORKERS buckets (by
# WEIGHTS when given, else round-robin) and run by a local work-stealing pool
# (Y2). N and M are validated; a shard that selects zero commands is a refusal.
run() {
    local dir=$1 shard=${2:-1/1} ff=${3:-0} wfile=${4:-} workers=${5:-1} out cmd i=0 total rc n m k=0 failed=0 first_fail=""
    local -a cmds
    if ! [[ "$shard" =~ ^([1-9][0-9]*)/([1-9][0-9]*)$ ]]; then
        printf 'REFUSE --shard must look like N/M, got %s\n' "$shard" >&2; return 2
    fi
    n=${BASH_REMATCH[1]}; m=${BASH_REMATCH[2]}
    if [ "$n" -gt "$m" ]; then printf 'REFUSE --shard %s: N exceeds M\n' "$shard" >&2; return 2; fi
    if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
        printf 'REFUSE --workers must be a positive integer, got %s\n' "$workers" >&2; return 2
    fi
    if [ "$workers" -gt 1 ] && [ "$m" -gt 1 ]; then
        printf 'REFUSE --workers %s with --shard %s: the local pool replaces host sharding, it does not stack on it\n' "$workers" "$shard" >&2; return 2
    fi
    if [ "$workers" -gt 1 ] && [ "$ff" -eq 1 ]; then
        printf 'REFUSE --fail-fast with --workers %s: "the commands after it" has no order in a pool\n' "$workers" >&2; return 2
    fi
    if [ -n "$wfile" ] && [ "$workers" -eq 1 ] && [ "$m" -eq 1 ]; then
        printf 'REFUSE --weights needs --shard N/M (M > 1) or --workers > 1: one runner has nothing to balance\n' >&2; return 2
    fi
    out=$(parse "$dir") || return $?
    mapfile -t cmds <<< "$out"
    total=${#cmds[@]}
    # parse() already refuses an empty directory; this is the belt for a future
    # parse that prints nothing: a here-string of "" still yields one empty element.
    if [ "$total" -eq 0 ] || [ -z "${cmds[0]}" ]; then
        printf 'REFUSE %s parsed to zero commands -- nothing to run is not a pass\n' "$dir" >&2; return 2
    fi
    if [ "$m" -gt 1 ]; then
        local -a mine=() share=()
        if [ -n "$wfile" ]; then
            # Y2 (#4424): the M shards are M runners that cannot steal from each
            # other, so the split itself is balanced -- weighted LPT over M
            # buckets, and shard N takes bucket N. plan() lists one bucket per
            # fragment in the same sorted order parse() reads them; a length
            # mismatch means the two disagree about the list, which is a refusal.
            out=$(plan "$dir" "$m" "$wfile") || return $?
            mapfile -t share < <(grep -v '^LOAD ' <<< "$out" | cut -f1)
            if [ "${#share[@]}" -ne "$total" ]; then
                printf 'REFUSE the weighted plan bucketed %s fragment(s) but parse read %s command(s)\n' "${#share[@]}" "$total" >&2; return 2
            fi
            printf 'weighted plan: %s\n' "$(grep '^LOAD ' <<< "$out" | tr '\n' ' ')"
        else
            for ((k = 0; k < total; k++)); do share+=( $(( k % m + 1 )) ); done
        fi
        for ((k = 0; k < total; k++)); do
            if [ "${share[$k]}" -eq "$n" ]; then mine+=("${cmds[$k]}"); fi
        done
        if [ "${#mine[@]}" -eq 0 ]; then
            printf 'REFUSE shard %s selects 0 of %s command(s) -- more shards than commands is not a pass\n' "$shard" "$total" >&2; return 2
        fi
        printf 'shard %s: %s of %s command(s)\n' "$shard" "${#mine[@]}" "$total"
        cmds=("${mine[@]}"); total=${#cmds[@]}
    fi
    if [ "$workers" -gt 1 ]; then
        local -a bucket_of=()
        local NB=$(( 3 * workers )) POOL_DIR="" b unclaimed=0
        [ "$NB" -le "$total" ] || NB=$total
        if [ -n "$wfile" ]; then
            out=$(plan "$dir" "$NB" "$wfile") || return $?
            mapfile -t bucket_of < <(grep -v '^LOAD ' <<< "$out" | cut -f1)
        else
            for ((k = 0; k < total; k++)); do bucket_of+=( $(( k % NB + 1 )) ); done
        fi
        # Σ belt on the plan: one bucket per command, each in 1..NB.
        if [ "${#bucket_of[@]}" -ne "$total" ] || grep -qvxE "[1-9][0-9]*" < <(printf '%s\n' "${bucket_of[@]}") \
            || [ "$(printf '%s\n' "${bucket_of[@]}" | sort -n | tail -1)" -gt "$NB" ]; then
            printf 'REFUSE the plan bucketed %s of %s command(s) or named a bucket outside 1..%s\n' "${#bucket_of[@]}" "$total" "$NB" >&2
            return 2
        fi
        printf 'pool: %s command(s) in %s bucket(s) over %s worker(s)\n' "$total" "$NB" "$workers"
        pool "$workers"
        for ((b = 1; b <= NB; b++)); do
            if [ -f "$POOL_DIR/claim.$b/by" ]; then
                printf 'bucket %s/%s: worker %s\n' "$b" "$NB" "$(cat "$POOL_DIR/claim.$b/by")"
            else
                unclaimed=$((unclaimed + 1)); printf 'UNCLAIMED bucket %s/%s\n' "$b" "$NB" >&2
            fi
        done
        for ((k = 0; k < total; k++)); do
            printf '::group::[%s/%s] %s\n' "$((k + 1))" "$total" "${cmds[$k]}"
            [ -f "$POOL_DIR/out.$k" ] && cat "$POOL_DIR/out.$k"
            printf '::endgroup::\n'
            [ -f "$POOL_DIR/rc.$k" ] || continue
            i=$((i + 1)); rc=$(cat "$POOL_DIR/rc.$k")
            if [ "$rc" -ne 0 ]; then
                failed=$((failed + 1))
                [ -n "$first_fail" ] || first_fail="[$((k + 1))/$total] exit $rc: ${cmds[$k]}"
                printf 'FAIL  [%s/%s] exit %s: %s\n' "$((k + 1))" "$total" "$rc" "${cmds[$k]}" >&2
            fi
        done
        rm -rf "${POOL_DIR:?}"
        [ "$unclaimed" -eq 0 ] || i=$(( i < total ? i : total - 1 ))
    else
        for cmd in "${cmds[@]}"; do
            i=$((i + 1))
            printf '::group::[%s/%s] %s\n' "$i" "$total" "$cmd"
            rc=0
            bash -c "$cmd" < /dev/null || rc=$?
            printf '::endgroup::\n'
            if [ "$rc" -ne 0 ]; then
                failed=$((failed + 1))
                [ -n "$first_fail" ] || first_fail="[$i/$total] exit $rc: $cmd"
                printf 'FAIL  [%s/%s] exit %s: %s\n' "$i" "$total" "$rc" "$cmd" >&2
                if [ "$ff" -eq 1 ]; then
                    printf '      --fail-fast: the %s command(s) after it were NOT RUN -- unmeasured, not passed\n' "$((total - i))" >&2
                    break
                fi
            fi
        done
    fi
    # Always, on both paths: the counts are numbers a consumer can read.
    printf 'RESULT ran=%s skipped=%s total=%s failed=%s\n' "$i" "$((total - i))" "$total" "$failed"
    [ -n "$first_fail" ] && printf 'RESULT first-failure %s\n' "$first_fail"
    if [ "$i" -lt "$total" ]; then
        printf 'UNKNOWN %s of %s command(s) were not run; unmeasured is neither pass nor fail\n' "$((total - i))" "$total" >&2
        return 2
    fi
    if [ "$failed" -gt 0 ]; then
        printf 'FAIL  %s of %s explicit test command(s) failed; all %s ran\n' "$failed" "$total" "$total" >&2
        return 1
    fi
    printf 'PASS  %s/%s explicit test command(s) from %s\n' "$i" "$total" "$dir"
}

case "${1:-}" in
    --list) parse "${2:-$DEFAULT_DIR}" ;;
    --run|--plan)
        mode=$1; dir=${2:-$DEFAULT_DIR}; shard=1/1; ff=0; wfile=""; workers=1; buckets=""
        shift 2 2>/dev/null || shift $#
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --shard) shard=${2:?--shard needs N/M}; shift 2 ;;
                --buckets) buckets=${2:?--buckets needs B}; shift 2 ;;
                --workers) workers=${2:?--workers needs N}; shift 2 ;;
                --weights) wfile=${2:?--weights needs FILE}; shift 2 ;;
                --fail-fast) ff=1; shift ;;
                *) printf 'usage: %s --run [DIR] [--shard N/M | --workers N [--weights F]] [--fail-fast]\n' "$0" >&2; exit 2 ;;
            esac
        done
        if [ "$mode" = --run ]; then run "$dir" "$shard" "$ff" "$wfile" "$workers"; exit $?; fi
        if ! [[ "$buckets" =~ ^[1-9][0-9]*$ ]] || [ -z "$wfile" ]; then
            printf 'usage: %s --plan [DIR] --buckets B --weights F\n' "$0" >&2; exit 2
        fi
        out=$(plan "$dir" "$buckets" "$wfile") || exit $?
        printf '%s\n' "$out"
        # max/min of the planned bucket loads, x100 (integer).
        awk '/^LOAD /{ if (mx == "" || $3 > mx) mx = $3; if (mn == "" || $3 < mn) mn = $3 }
             END { printf "RATIO max/min x100 = %d (max %d s, min %d s)\n", (mn > 0 ? int(100 * mx / mn + 0.5) : 0), mx, mn }' <<< "$out" ;;
    -h|--help) sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) printf 'usage: %s --list|--run|--plan [DIR] ...\n' "$0" >&2; exit 2 ;;
esac
