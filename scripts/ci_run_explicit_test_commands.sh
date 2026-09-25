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


# plan DIR M WEIGHTS PRELOAD -> one line per fragment, in list order:
#   <shard>\t<seconds>\t<basename>
# then one `LOAD <shard> <seconds>` line per shard on stderr-free stdout tail.
# Longest-processing-time-first over measured seconds (Y2): fragments by weight
# descending (name ascending on a tie), each to the least-loaded shard (lowest
# index on a tie), shards starting at PRELOAD -- the seconds of the once-only
# steps each shard already carries (e.g. "Build every example" on shard 3).
# Deterministic: the same tree, weights and preload give the same plan on every
# shard, which is what makes M independent jobs cover the list exactly once.
# A fragment with no weight is planned at the median of the weights that do
# apply, so a new fragment needs no edit here. rc 2 on a malformed weights line,
# a duplicate weight, a preload that is not M non-negative integers, or no
# applicable weight at all (a plan from zero measurements is round-robin in
# disguise). A weight naming no fragment is STALE on stderr, not a refusal:
# deleting a fragment must not break every other PR.
plan() {
    local dir=$1 m=$2 wfile=$3 preload=${4:-} out base line w i j best med n
    local -a names=() pre=() load=() sorted=()
    local -A weight=() shard_of=() present=()
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
    if [ -z "$preload" ]; then
        for ((i = 0; i < m; i++)); do pre+=(0); done
    else
        IFS=, read -ra pre <<< "$preload"
        if [ "${#pre[@]}" -ne "$m" ]; then
            printf 'REFUSE --preload %s has %s value(s) for %s shard(s)\n' "$preload" "${#pre[@]}" "$m" >&2; return 2
        fi
        for w in "${pre[@]}"; do
            [[ "$w" =~ ^[0-9]+$ ]] || { printf 'REFUSE --preload value %s is not a non-negative integer\n' "$w" >&2; return 2; }
        done
    fi
    load=("${pre[@]}")
    while IFS=$'\t' read -r w base; do
        best=0
        for ((j = 1; j < m; j++)); do
            [ "${load[$j]}" -lt "${load[$best]}" ] && best=$j
        done
        load[best]=$(( load[best] + w ))
        shard_of[$base]=$(( best + 1 ))
    done < <(for base in "${names[@]}"; do printf '%s\t%s\n' "${weight[$base]:-$med}" "$base"; done | LC_ALL=C sort -t$'\t' -k1,1nr -k2,2)
    for base in "${names[@]}"; do
        printf '%s\t%s\t%s\n' "${shard_of[$base]}" "${weight[$base]:-$med}" "$base"
    done
    for ((j = 0; j < m; j++)); do printf 'LOAD %s %s\n' "$((j + 1))" "${load[$j]}"; done
}
# run DIR [N/M] [FF] [WEIGHTS] [PRELOAD] -- runs the commands in order; with N/M
# only the N-th shard's share, so M shards cover the list exactly once between
# them (PACK-001). The share is every M-th command from the N-th (round-robin),
# or with WEIGHTS the shard `plan` assigns (Y2). N and M are validated; a shard
# that selects zero commands is a refusal, not a pass.
run() {
    local dir=$1 shard=${2:-1/1} ff=${3:-0} wfile=${4:-} preload=${5:-} out cmd i=0 total rc n m k=0 sel=0 failed=0 first_fail=""
    local -a cmds
    if ! [[ "$shard" =~ ^([1-9][0-9]*)/([1-9][0-9]*)$ ]]; then
        printf 'REFUSE --shard must look like N/M, got %s\n' "$shard" >&2; return 2
    fi
    n=${BASH_REMATCH[1]}; m=${BASH_REMATCH[2]}
    if [ "$n" -gt "$m" ]; then printf 'REFUSE --shard %s: N exceeds M\n' "$shard" >&2; return 2; fi
    out=$(parse "$dir") || return $?
    mapfile -t cmds <<< "$out"
    total=${#cmds[@]}
    # parse() already refuses an empty directory; this is the belt for a future
    # parse that prints nothing: a here-string of "" still yields one empty element.
    if [ "$total" -eq 0 ] || [ -z "${cmds[0]}" ]; then
        printf 'REFUSE %s parsed to zero commands -- nothing to run is not a pass\n' "$dir" >&2; return 2
    fi
    if [ "$m" -gt 1 ]; then
        local -a mine=() assigned=()
        if [ -n "$wfile" ]; then
            # Y2: the LPT plan, in the same sorted order as cmds. Belt for the
            # Σ invariant: one assignment per command, each to a shard in 1..M.
            out=$(plan "$dir" "$m" "$wfile" "$preload") || return $?
            mapfile -t assigned < <(grep -v '^LOAD ' <<< "$out" | cut -f1)
            if [ "${#assigned[@]}" -ne "$total" ] || grep -qvxE "[1-9][0-9]*" < <(printf '%s\n' "${assigned[@]}") \
                || [ "$(printf '%s\n' "${assigned[@]}" | sort -n | tail -1)" -gt "$m" ]; then
                printf 'REFUSE the plan assigned %s of %s command(s) or named a shard outside 1..%s\n' "${#assigned[@]}" "$total" "$m" >&2
                return 2
            fi
        fi
        for cmd in "${cmds[@]}"; do
            k=$((k + 1))
            if [ -n "$wfile" ]; then sel=${assigned[$((k - 1))]}; else sel=$(( (k - 1) % m + 1 )); fi
            if [ "$sel" -eq "$n" ]; then mine+=("$cmd"); fi
        done
        if [ "${#mine[@]}" -eq 0 ]; then
            printf 'REFUSE shard %s selects 0 of %s command(s) -- more shards than commands is not a pass\n' "$shard" "$total" >&2; return 2
        fi
        printf 'shard %s: %s of %s command(s)\n' "$shard" "${#mine[@]}" "$total"
        cmds=("${mine[@]}"); total=${#cmds[@]}
    fi
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
        mode=$1; dir=${2:-$DEFAULT_DIR}; shard=1/1; ff=0; wfile=""; preload=""; shards=""
        shift 2 2>/dev/null || shift $#
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --shard) shard=${2:?--shard needs N/M}; shift 2 ;;
                --shards) shards=${2:?--shards needs M}; shift 2 ;;
                --weights) wfile=${2:?--weights needs FILE}; shift 2 ;;
                --preload) preload=${2:?--preload needs S1,...,SM}; shift 2 ;;
                --fail-fast) ff=1; shift ;;
                *) printf 'usage: %s --run [DIR] [--shard N/M] [--weights F [--preload S1,..,SM]] [--fail-fast]\n' "$0" >&2; exit 2 ;;
            esac
        done
        if [ -n "$preload" ] && [ -z "$wfile" ]; then
            printf 'REFUSE --preload without --weights: round-robin ignores it\n' >&2; exit 2
        fi
        if [ "$mode" = --run ]; then run "$dir" "$shard" "$ff" "$wfile" "$preload"; exit $?; fi
        if ! [[ "$shards" =~ ^[1-9][0-9]*$ ]] || [ -z "$wfile" ]; then
            printf 'usage: %s --plan [DIR] --shards M --weights F [--preload S1,..,SM]\n' "$0" >&2; exit 2
        fi
        out=$(plan "$dir" "$shards" "$wfile" "$preload") || exit $?
        printf '%s\n' "$out"
        # max/min of the planned loads, x100 (integer): the Y2 target is <= 125.
        awk '/^LOAD /{ if (mx == "" || $3 > mx) mx = $3; if (mn == "" || $3 < mn) mn = $3 }
             END { printf "RATIO max/min x100 = %d (max %d s, min %d s)\n", (mn > 0 ? int(100 * mx / mn + 0.5) : 0), mx, mn }' <<< "$out" ;;
    -h|--help) sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) printf 'usage: %s --list|--run|--plan [DIR] ...\n' "$0" >&2; exit 2 ;;
esac
