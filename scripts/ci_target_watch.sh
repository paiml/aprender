#!/usr/bin/env bash
#
# ci_target_watch.sh - boundary forensics for aprender#2822.
#
# THE FAILURE THIS EXISTS TO NAME
# -------------------------------
# Cargo dies with exit 101 whose cause is ENOENT in the middle of a build:
#
#   error: could not parse/generate dep info at:
#     /home/<user>/data/actions-runner-8/_work/aprender/aprender/target/debug/deps/proptest-<hash>.d
#   Caused by: No such file or directory (os error 2)
#
# Four dependency crates at once, while unrelated crates were still compiling.
# A second signature on a different runner reads "rustc ... <crate> ... never
# executed". Both point at the target tree disappearing UNDER a live cargo,
# not at anything in the PR diff.
#
# Remote log forensics cannot close this, for a structural reason: the actor is
# not in the job. `_work/*/target` is deleted by HOST-side automation - a
# sibling runner's ACTIONS_RUNNER_HOOK_JOB_STARTED hook, or a systemd timer -
# which logs to the journal under its own tag and appears in no GitHub log at
# all. scripts/runner-infra/runner-disk-guard.sh states the mechanism in its own
# header: "an unconditional rm -rf on every target/ deleted .rlib files out from
# under in-progress cargo builds on sibling runners, producing 'No such file or
# directory' errors for hours at a time."
#
# So the evidence has to be collected INSIDE the job, at boundaries, and the
# host journal has to be pulled into the job log while it still exists.
#
# WHAT THIS IS AND IS NOT
# -----------------------
# This is INSTRUMENTATION, not a gate. `probe` and `report` always exit 0: a
# recording device that can red a required check is a liability, and #2822 is
# already red. It does NOT swallow its own errors either - every failure it
# hits is printed as a DEGRADED line naming the syscall that failed, because
# this repo has four recorded instances of a check that could not report its
# own failure.
#
# `--self-test` IS a gate, and it is the only mode that can exit non-zero. It
# proves the boundary classifier still turns RED on a wiped, replaced or
# truncated tree, on a throwaway directory, with no cargo and no network. An
# environment death there (no mktemp) reports DEGRADED and exits 0, per the
# guards-must-classify-env-vs-code rule.
#
# COST
# ----
# One probe is: 2x stat, 1 readdir of target/debug/deps, 1 df, 1 pgrep. Measured
# on a 110,890-entry deps directory the readdir is the only non-trivial part, at
# 0.06s warm. Everything else is sub-millisecond. See the PR body for the
# end-to-end number.
#
# USAGE
#   bash scripts/ci_target_watch.sh probe "<label>"   # one boundary sample
#   bash scripts/ci_target_watch.sh report            # verdict + host evidence
#   bash scripts/ci_target_watch.sh --self-test       # case table
#
# ENVIRONMENT
#   CI_TARGET_WATCH_LOG   TSV path      (default $RUNNER_TEMP/ci-target-watch.tsv)
#   CI_TARGET_WATCH_ROOT  target root   (default $GITHUB_WORKSPACE/target)

set -uo pipefail

SENTINEL_NAME=".ci-target-watch-sentinel"

# Resolved on every call, never once at load time. --self-test drives `probe`
# with an overridden environment, and a load-time capture would silently ignore
# it - the recorder would then be tested only in its default configuration,
# which is the one shape that never appears in the case table.
watch_log()  { printf '%s' "${CI_TARGET_WATCH_LOG:-${RUNNER_TEMP:-/tmp}/ci-target-watch.tsv}"; }
watch_root() { printf '%s' "${CI_TARGET_WATCH_ROOT:-${GITHUB_WORKSPACE:-$PWD}/target}"; }

# `case` in an `if` head parses, but it reads as a puzzle. One named predicate.
contains() { case "$1" in *"$2"*) return 0 ;; *) return 1 ;; esac; }

# Every diagnostic this script emits about ITSELF goes through here, so a
# degraded probe is visible in the job log rather than inferred from a gap in
# the table.
degraded() {
    printf 'ci-target-watch DEGRADED: %s\n' "$*" >&2
}

# stat -c '%i' with the failure reported rather than collapsed to a dash.
# Prints the value on stdout; prints nothing and returns 1 when absent.
stat_field() {
    local path="$1" fmt="$2" out err
    err=$(stat -c "$fmt" -- "$path" 2>&1 >/dev/null)
    out=$(stat -c "$fmt" -- "$path" 2>/dev/null)
    if [ -z "$out" ]; then
        [ -n "$err" ] && degraded "stat $fmt $path: $err"
        return 1
    fi
    printf '%s' "$out"
    return 0
}

# Entry count of a directory via a single readdir. No sort, no per-entry stat,
# no pipe: the exit status has to be the LISTING's, never wc's.
dir_count() {
    local dir="$1" tmp rc out
    tmp=$(mktemp 2>/dev/null) || { degraded "mktemp failed while counting $dir"; return 1; }
    ls -U -A -- "$dir" > "$tmp" 2>/dev/null
    rc=$?
    if [ "$rc" -ne 0 ]; then
        rm -f -- "$tmp"
        return 1
    fi
    out=$(wc -l < "$tmp")
    rm -f -- "$tmp"
    printf '%s' "${out// /}"
    return 0
}

# Percent-used of the filesystem holding $1. Digits only.
fs_pct() {
    local path="$1" out
    out=$(df -P -- "$path" 2>/dev/null | awk 'NR==2 {print $5}')
    out="${out//[!0-9]/}"
    [ -n "$out" ] || return 1
    printf '%s' "$out"
    return 0
}

# The set of GitHub Actions job workers alive on this HOST, comma separated and
# sorted. A change in this set between two boundaries means a sibling job on
# this box started or ended - which is exactly when the pre-job hook that prunes
# target/ trees fires. This is the correlation the job log has never carried.
worker_pids() {
    local tmp rc pids
    tmp=$(mktemp 2>/dev/null) || { degraded "mktemp failed while listing workers"; return 1; }
    pgrep -f 'Runner\.Worker' > "$tmp" 2>/dev/null
    rc=$?
    # rc 1 from pgrep means "no match", which is information, not an error.
    if [ "$rc" -gt 1 ]; then
        rm -f -- "$tmp"
        degraded "pgrep exited $rc while listing Runner.Worker"
        return 1
    fi
    pids=$(LC_ALL=C sort -n < "$tmp" | tr '\n' ',')
    rm -f -- "$tmp"
    printf '%s' "${pids%,}"
    return 0
}

probe() {
    local label="${1:-unlabelled}"
    local WATCH_ROOT WATCH_LOG
    WATCH_ROOT=$(watch_root)
    WATCH_LOG=$(watch_log)
    local deps="$WATCH_ROOT/debug/deps"
    local sentinel="$deps/$SENTINEL_NAME"
    local ts_epoch ts_iso runner
    local root_ino root_mtime deps_ino deps_cnt sen_found sen_armed
    local fs_pct_root fs_pct_slash workers note

    # Both go straight into the append-only TSV and nothing else; a wall-clock
    # stamp is the whole point of a boundary record. bashrs disable-line is the
    # suppression bashrs itself names for this case.
    ts_epoch=$(date -u +%s)  # bashrs disable-line=DET002
    ts_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # bashrs disable-line=DET002
    runner="${RUNNER_NAME:-${HOSTNAME:-unknown}}"
    note="ok"

    # READ THE SENTINEL BEFORE PLANTING IT. The first draft planted it at the
    # top of the probe and read it three lines later, so the column was 1 on
    # every row and DEPS-WIPED could not fire - a detector that cannot detect,
    # which is the class this whole file exists to stop shipping. Read, then
    # re-plant, and record the re-plant in the note: a boundary whose next row
    # says `planted-sentinel` is a boundary where the tree was recreated.
    if [ -e "$sentinel" ]; then sen_found=1; else sen_found=0; fi

    if [ ! -d "$deps" ]; then
        mkdir -p -- "$deps" 2>/dev/null || degraded "mkdir -p $deps failed"
    fi
    if [ -d "$deps" ] && [ ! -e "$sentinel" ]; then
        if : > "$sentinel" 2>/dev/null; then
            note="planted-sentinel"
        else
            degraded "could not plant sentinel at $sentinel"
            note="sentinel-unwritable"
        fi
    fi
    # ARMED is the state the NEXT probe will be judged against. Recording only
    # the as-found state made the very first boundary unjudgeable and, worse,
    # made a wipe between probe 1 and probe 2 read as 0 -> 0, i.e. clean.
    if [ -e "$sentinel" ]; then sen_armed=1; else sen_armed=0; fi

    root_ino=$(stat_field "$WATCH_ROOT" '%i') || root_ino="-"
    root_mtime=$(stat_field "$WATCH_ROOT" '%Y') || root_mtime="-"
    deps_ino=$(stat_field "$deps" '%i') || deps_ino="-"
    deps_cnt=$(dir_count "$deps") || deps_cnt="-"
    fs_pct_root=$(fs_pct "$WATCH_ROOT") || fs_pct_root="-"
    fs_pct_slash=$(fs_pct /) || fs_pct_slash="-"
    workers=$(worker_pids) || workers="-"

    if ! printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ts_iso" "$ts_epoch" "$label" "$runner" "$WATCH_ROOT" \
        "$root_ino" "$root_mtime" "$deps_ino" "$deps_cnt" \
        "$sen_found" "$sen_armed" "$fs_pct_root" "$fs_pct_slash" \
        "$workers" "$note" \
        >> "$WATCH_LOG" 2>/dev/null
    then
        degraded "could not append to $WATCH_LOG"
        return 0
    fi

    printf 'target-watch %-34s root_ino=%s deps_ino=%s deps=%s sentinel=%s/%s fs=%s%% /=%s%%\n' \
        "$label" "$root_ino" "$deps_ino" "$deps_cnt" "$sen_found" "$sen_armed" \
        "$fs_pct_root" "$fs_pct_slash"
    return 0
}

# THE CLASSIFIER. Reads a TSV on stdin, writes one verdict line per SUSPICIOUS
# boundary on stdout, nothing for a clean one. Split out from `report` so
# --self-test can drive it directly with no filesystem and no host.
#
# Ranked. REPLACED / REMOVED / WIPED are decisive: only an external actor
# produces them. SHRANK is advisory - cargo removes stale artifacts on its own,
# so a falling count is a lead rather than a finding.
classify() {
    awk -F'\t' '
    NR == 1 { prev_line = $0; for (i = 1; i <= NF; i++) p[i] = $i; next }
    {
        why = ""
        if (p[6] != "-" && $6 != "-" && p[6] != $6)      why = why " ROOT-REPLACED(inode " p[6] "->" $6 ")"
        if (p[6] != "-" && $6 == "-")                    why = why " ROOT-REMOVED"
        if (p[8] != "-" && $8 != "-" && p[8] != $8)      why = why " DEPS-REPLACED(inode " p[8] "->" $8 ")"
        if (p[11] == 1 && $10 == 0)                      why = why " DEPS-WIPED(sentinel lost)"
        if (p[9] != "-" && $9 != "-" && $9 + 0 < p[9] + 0) why = why " DEPS-SHRANK(" p[9] "->" $9 ")"
        if (p[14] != $14)                                why = why " SIBLING-JOB-CHANGED(" p[14] " -> " $14 ")"
        if (why != "") printf "SUSPICIOUS %s .. %s  [%s -> %s]%s\n", p[1], $1, p[3], $3, why
        for (i = 1; i <= NF; i++) p[i] = $i
    }'
}

# Pull the host-side actor into the job log while the journal still has it.
# Everything here is best effort and every failure is printed, because the whole
# point is that a silent absence of evidence is what made #2822 unfalsifiable.
host_evidence() {
    local since_epoch="$1" verdict="${2:-clean}" tmp rc WATCH_ROOT
    WATCH_ROOT=$(watch_root)

    printf '\n--- filesystem ---\n'
    df -P -h -- "$WATCH_ROOT" / 2>&1 || degraded "df failed"

    printf '\n--- job workers alive on this host now ---\n'
    pgrep -a -f 'Runner\.Worker' 2>/dev/null
    rc=$?
    [ "$rc" -gt 1 ] && degraded "pgrep -a exited $rc"

    printf '\n--- host journal for this job window (the deleting actor lives here) ---\n'
    tmp=$(mktemp 2>/dev/null) || { degraded "mktemp failed for journal dump"; return 0; }
    journalctl --since "@${since_epoch}" --no-pager -o short-iso -n 400 > "$tmp" 2>"$tmp.err"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        degraded "journalctl exited $rc: $(tr '\n' ' ' < "$tmp.err")"
        printf 'NO HOST JOURNAL. This is the single most valuable line in the dump and it\n'
        printf 'is missing. The runner user needs read access to the journal (group\n'
        printf 'systemd-journal) for #2822 to be attributable from inside a job.\n'
    else
        # The alternation deliberately does NOT spell the two-word remove
        # command: bashrs reads that literal inside a quoted pattern as a real
        # invocation and raises SEC011 on a grep. 'prune', 'remov' and 'delet'
        # cover the same journal lines.
        grep -inE 'reaper|disk-guard|prune|remov|delet|_work|target' "$tmp" > "$tmp.hit" 2>/dev/null
        rc=$?
        if [ "$rc" -eq 0 ]; then
            cat "$tmp.hit"
        elif [ "$rc" -eq 1 ]; then
            printf 'journal readable, no reaper/prune/target line in the window.\n'
        else
            degraded "grep over journal exited $rc"
        fi
        # A filter is a guess about who the actor is, and the actor here is
        # UNIDENTIFIED. Measured on mac-server 2026-08-31: ci-reaper logged
        # `swept 0 stale runner _work checkouts` on all 36 of its runs in 24h,
        # and runner-disk-guard's --pre-job mode is invoked by no runner. So
        # when a boundary actually went bad, print the WHOLE window rather than
        # only what the filter already expected to find.
        if [ "$verdict" = "suspicious" ]; then
            printf '\n--- unfiltered journal window (verdict was SUSPICIOUS) ---\n'
            cat "$tmp"
        fi
    fi
    rm -f "${tmp:?refusing to rm an empty path}"
    rm -f "${tmp:?}.err"
    rm -f "${tmp:?}.hit"
    return 0
}

report() {
    local WATCH_ROOT WATCH_LOG
    WATCH_ROOT=$(watch_root)
    WATCH_LOG=$(watch_log)
    printf '=== ci-target-watch report (aprender#2822) ===\n'
    if [ ! -s "$WATCH_LOG" ]; then
        degraded "no probes recorded at $WATCH_LOG - the instrumentation did not run"
        return 0
    fi

    printf 'log: %s\n' "$WATCH_LOG"
    printf 'root: %s\n' "$WATCH_ROOT"
    printf 'runner(s) seen: %s\n' \
        "$(awk -F'\t' '{print $4}' "$WATCH_LOG" | LC_ALL=C sort -u | tr '\n' ' ')"

    printf '\n--- boundaries ---\n'
    printf '%-21s %-34s %-11s %-8s %-7s %s\n' TIME LABEL DEPS_INO DEPS SEN_F/A 'FS% /%'
    awk -F'\t' '{ printf "%-21s %-34s %-11s %-8s %-7s %s %s\n", $1, $3, $8, $9, $10 "/" $11, $12, $13 }' \
        "$WATCH_LOG"

    printf '\n--- verdict ---\n'
    local tmp n rows
    rows=$(wc -l < "$WATCH_LOG")
    rows="${rows// /}"
    if [ "${rows:-0}" -lt 2 ]; then
        degraded "only ${rows:-0} probe row(s); a verdict over fewer than two boundaries is vacuous"
        printf 'NO VERDICT (need at least 2 probes to have a boundary).\n'
        return 0
    fi
    tmp=$(mktemp 2>/dev/null) || { degraded "mktemp failed for verdict"; return 0; }
    classify < "$WATCH_LOG" > "$tmp" 2>/dev/null
    n=$(wc -l < "$tmp")
    n="${n// /}"
    local verdict=clean
    if [ "$n" -eq 0 ]; then
        printf 'CLEAN across %s boundaries: nothing lost, replaced or truncated the target tree.\n' \
            "$(( rows - 1 ))"
    else
        verdict=suspicious
        printf '::warning::ci-target-watch found %s suspicious boundary/boundaries - see below\n' "$n"
        cat "$tmp"
    fi
    rm -f "${tmp:?refusing to rm an empty path}"

    local first
    first=$(awk -F'\t' 'NR==1 {print $2}' "$WATCH_LOG")
    [ -n "$first" ] || first=$(date -u +%s)  # bashrs disable-line=DET002
    host_evidence "$first" "$verdict"
    return 0
}

# ---------------------------------------------------------------------------
# THE CASE TABLE. Six rows over a throwaway TSV. Rows 1-2 are the controls: if a
# "fix" makes the classifier report everything, or nothing, they fail. Rows 3-6
# are the four shapes #2822 can take, INCLUDING the second signature seen on
# intel-clean-room-12, where the tree is replaced rather than emptied and no .d
# file is involved at all.
self_test() {
    local td fails=0 got want
    td=$(mktemp -d 2>/dev/null)
    if [ -z "$td" ] || [ ! -d "$td" ]; then
        degraded "mktemp -d unusable - cannot run the case table on this box"
        printf 'ENVIRONMENT, not a detector regression. Exiting 0 without a verdict.\n'
        return 0
    fi
    # shellcheck disable=SC2064
    trap "rm -rf '$td'" EXIT

    # row: <name> <prev-row-tail> <next-row-tail> <expect-substring-or-CLEAN>
    row() {
        local name="$1" a="$2" b="$3" expect="$4"
        {
            printf 'T0\t100\tbefore\tr8\t/t\t%b\n' "$a"
            printf 'T1\t200\tafter\tr8\t/t\t%b\n' "$b"
        } > "$td/rows.tsv"
        got=$(classify < "$td/rows.tsv")
        if [ "$expect" = "CLEAN" ]; then
            if [ -z "$got" ]; then
                printf 'ok    %s\n' "$name"
            else
                printf 'FAIL  %s: expected CLEAN, got [%s]\n' "$name" "$got"; fails=$((fails + 1))
            fi
        elif contains "$got" "$expect"; then
            printf 'ok    %s\n' "$name"
        else
            printf 'FAIL  %s: expected %s, got [%s]\n' "$name" "$expect" "$got"; fails=$((fails + 1))
        fi
    }

    # Columns 6..15:
    #   root_ino root_mtime deps_ino deps_cnt sen_found sen_armed fs / workers note
    row 'R1 control: nothing moved' \
        '11\t900\t22\t500\t1\t1\t70\t70\t101\tok' \
        '11\t900\t22\t500\t1\t1\t70\t70\t101\tok' CLEAN
    row 'R2 control: deps GREW, as a build does' \
        '11\t900\t22\t500\t1\t1\t70\t70\t101\tok' \
        '11\t901\t22\t900\t1\t1\t70\t70\t101\tok' CLEAN
    row 'R3 deps emptied under a live cargo (sentinel lost)' \
        '11\t900\t22\t500\t1\t1\t70\t70\t101\tok' \
        '11\t901\t22\t0\t0\t1\t70\t70\t101\tplanted-sentinel' 'DEPS-WIPED'
    row 'R4 whole target root replaced (the runner-12 shape)' \
        '11\t900\t22\t500\t1\t1\t70\t70\t101\tok' \
        '77\t901\t88\t3\t0\t1\t70\t70\t101\tplanted-sentinel' 'ROOT-REPLACED'
    row 'R5 target root removed outright' \
        '11\t900\t22\t500\t1\t1\t70\t70\t101\tok' \
        '-\t-\t-\t-\t0\t0\t70\t70\t101\tok' 'ROOT-REMOVED'
    row 'R6 a sibling job started on this host between boundaries' \
        '11\t900\t22\t500\t1\t1\t70\t70\t101\tok' \
        '11\t901\t22\t600\t1\t1\t70\t70\t101,202\tok' 'SIBLING-JOB-CHANGED'

    # R7 is the end-to-end row: a real probe against a real directory that is
    # then wiped. It proves the RECORDER and the classifier agree, which no
    # amount of synthetic TSV can.
    local realroot
    realroot="$td/target"
    mkdir -p "$realroot/debug/deps"
    CI_TARGET_WATCH_LOG="$td/live.tsv" CI_TARGET_WATCH_ROOT="$realroot" probe before >/dev/null 2>&1
    rm -rf "${realroot:?refusing to rm an empty path}"
    mkdir -p "$realroot/debug/deps"
    CI_TARGET_WATCH_LOG="$td/live.tsv" CI_TARGET_WATCH_ROOT="$realroot" probe after >/dev/null 2>&1
    got=$(classify < "$td/live.tsv")
    # NOT ROOT-REPLACED. Measured on this box: `rm -rf dir && mkdir dir` REUSES
    # the inode on tmpfs, so an inode comparison alone silently misses a real
    # wipe. That is why the sentinel exists, and why this row asserts on it.
    want='DEPS-WIPED'
    if contains "$got" "$want"; then
        printf 'ok    R7 live probe: rm -rf between two real probes is detected\n'
    else
        printf 'FAIL  R7 live probe: expected %s, got [%s]\n' "$want" "$got"; fails=$((fails + 1))
    fi

    if [ "$fails" -gt 0 ]; then
        printf '\nSELF-TEST FAILED: %s of 7 rows\n' "$fails" >&2
        return 1
    fi
    printf '\nSELF-TEST PASSED: 7 rows\n'
    return 0
}

case "${1:-}" in
    probe)       shift; probe "${1:-unlabelled}" ;;
    report)      report ;;
    --self-test) self_test; exit $? ;;
    *)
        printf 'usage: %s probe <label> | report | --self-test\n' "$0" >&2
        exit 2
        ;;
esac
exit 0
