#!/bin/bash
# Self-hosted GitHub Actions runner disk guard.
# Two modes:
#   --pre-job        Called from ACTIONS_RUNNER_HOOK_JOB_STARTED. Only acts when
#                    disk usage on / exceeds HIGH_WATER_PCT (default 85).
#   --nightly        Called from systemd timer. Unconditional prune of idle
#                    target/ dirs older than STALE_DAYS (default 7) across all
#                    /home/noah/data/actions-runner*/_work trees.
#
# Safe by design:
#   - only touches target/ under _work trees — never the repo checkout itself
#   - in --pre-job mode, SKIPS target/ dirs on runners that currently have an
#     active Runner.Worker process (prevents clobbering mid-compile jobs on
#     sibling runners when disk crosses the high-water mark)
#   - exits 0 even if nothing to prune
#
# Rationale for skip-active: on a shared host with ≥2 runners, an unconditional
# rm -rf on every target/ deleted .rlib files out from under in-progress cargo
# builds on sibling runners, producing "No such file or directory" errors for
# hours at a time. A skipped active runner keeps its own target/ — it is the
# runner that "owns" that target for the duration of its job.
set -euo pipefail

HIGH_WATER_PCT="${HIGH_WATER_PCT:-85}"
STALE_DAYS="${STALE_DAYS:-7}"
RUNNERS_ROOT="${RUNNERS_ROOT:-/home/noah/data}"
# Space-separated list of "bind-mount target roots" — directories that workflow
# jobs mount into containers as CARGO_TARGET_DIR. These are NOT under a runner's
# _work tree, so the standard runner walk doesn't reach them. Subdirs are named
# by PR number (or "main", or "debug" for pre-isolation orphans). Any subdir
# not modified in STALE_DAYS days is eligible for prune. On 2026-04-23 this
# class held 1.9T (36 closed-PR dirs + a 359G "debug" orphan); without this
# coverage, disk-guard misses the single biggest runner-disk-fill source.
BIND_MOUNT_ROOTS="${BIND_MOUNT_ROOTS:-/mnt/nvme-raid0/targets/aprender-ci}"
LOG_TAG="runner-disk-guard"

log() { logger -t "$LOG_TAG" -- "$*" || true; echo "[$LOG_TAG] $*"; }

disk_pct() { df --output=pcent / | tail -1 | tr -dc '0-9'; }

# Returns 0 if $1 (runner_dir, e.g. /home/noah/data/actions-runner-5) has an
# active Runner.Worker process; 1 if idle. Runner.Listener is always running
# and does NOT count as "active" — only the Worker (spawned per job) does.
runner_has_active_worker() {
    local runner_dir="$1"
    local pid cwd
    # pgrep -f matches the process command line; Runner.Worker is the per-job
    # binary that the Listener spawns and its cwd is under the runner's tree.
    for pid in $(pgrep -f "Runner\.Worker" 2>/dev/null || true); do
        cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || echo "")"
        if [ -n "$cwd" ] && [[ "$cwd" == "$runner_dir"* ]]; then
            return 0
        fi
    done
    return 1
}

# Extract runner dir (/home/noah/data/actions-runner-N) from a target path
# like /home/noah/data/actions-runner-5/_work/aprender/aprender/target
runner_dir_of() {
    local tgt="$1"
    echo "$tgt" | sed -E "s|($RUNNERS_ROOT/actions-runner[^/]+)/.*|\\1|"
}

prune_target_dirs() {
    local mode="$1"  # "aggressive" or "stale"
    local total_freed_kb=0
    local before_kb after_kb

    before_kb="$(df --output=avail / | tail -1 | tr -dc '0-9')"

    while IFS= read -r -d '' tgt; do
        local runner_dir
        runner_dir="$(runner_dir_of "$tgt")"

        # Never prune a target/ belonging to a runner with an active Worker
        # (skipping protects in-progress cargo builds from rlib deletion races).
        if runner_has_active_worker "$runner_dir"; then
            log "skip (runner active): $tgt"
            continue
        fi

        if [ "$mode" = "stale" ]; then
            # Only prune if nothing touched in STALE_DAYS days
            if find "$tgt" -maxdepth 0 -mtime +"$STALE_DAYS" | grep -q .; then
                log "pruning stale target: $tgt"
                rm -rf --one-file-system "$tgt" 2>/dev/null || sudo rm -rf --one-file-system "$tgt" 2>/dev/null || true
            fi
        else
            log "aggressive prune: $tgt"
            rm -rf --one-file-system "$tgt" 2>/dev/null || sudo rm -rf --one-file-system "$tgt" 2>/dev/null || true
        fi
    done < <(find "$RUNNERS_ROOT"/actions-runner*/_work -mindepth 3 -maxdepth 4 -type d -name target -print0 2>/dev/null)

    after_kb="$(df --output=avail / | tail -1 | tr -dc '0-9')"
    total_freed_kb=$(( after_kb - before_kb ))
    log "freed approximately ${total_freed_kb} KiB (mode=$mode)"
}

# A candidate directory is IDLE only if NOTHING anywhere underneath it was
# modified within the window. Its OWN mtime does not answer that question, and
# has not since 2026-05-15, when ci.yml moved the bind mount one level down —
# from `<root>/<PR_OR_REF>` to `<root>/<PR_OR_REF>/run-<RUN_ID>` (the
# cancel-corrupt-state fix, aprender#1693). A directory's mtime changes only
# when an entry is created or removed DIRECTLY inside it, so the parent's mtime
# freezes the instant its `run-<RUN_ID>` child is created and stays frozen while
# cargo writes gigabytes underneath. An in-flight build's parent therefore reads
# as arbitrarily idle, and `rm -rf` on it takes the live build with it — cargo
# then dies ENOENT on its own `target/debug/...` paths mid-compile, which is
# exactly how aprender `workspace-test` failed on main and on the merge-queue
# ref within the same minute on 2026-08-21 (runs 32520690533 / 32520753928).
# The header comment this replaces asserted the opposite ("a container touching
# the dir keeps mtime current") and was true only of the pre-#1693 layout.
#
# `-newermt "-N minutes"` is a half-open comparison with no gap at exactly N,
# and `-print -quit` stops the walk at the FIRST recent entry, so the check
# costs a directory descent rather than a full stat of a 50GB tree. Command
# substitution, not a pipe: `find ... | grep -q` under `set -o pipefail` returns
# 141 when grep exits first and find takes SIGPIPE.
tree_is_idle() {
    local dir="$1" min_idle_min="$2" recent
    recent="$(find "$dir" -newermt "-${min_idle_min} minutes" -print -quit 2>/dev/null)"
    [ -z "$recent" ]
}

# Prune bind-mount target roots (/mnt/nvme-raid0/targets/aprender-ci/*).
# Subdirs are per-PR target dirs isolated per task #134. Remove:
#   - "debug" subdir unconditionally (orphan from pre-isolation era)
#   - PR-named subdirs with NOTHING under them touched in MIN_IDLE_MIN minutes.
# Nightly uses $((STALE_DAYS*24*60)); pre-job uses a 60-min floor so a full-disk
# situation can still reclaim most stale PR dirs while sparing fresh ones.
prune_bind_mount_target_roots() {
    local min_idle_min="$1"  # minutes with no write anywhere below = "stale"
    local root subdir
    for root in $BIND_MOUNT_ROOTS; do
        [ -d "$root" ] || continue
        # Orphan "debug" dir is always safe to prune (not bind-mounted by any
        # current workflow — isolation task #134 replaced it with per-PR dirs).
        if [ -d "$root/debug" ]; then
            log "prune orphan debug: $root/debug"
            rm -rf --one-file-system "$root/debug" 2>/dev/null \
                || sudo rm -rf --one-file-system "$root/debug" 2>/dev/null || true
        fi
        # "main" is explicitly preserved — it hosts push-to-main CI target cache
        # and is legitimately re-used.
        while IFS= read -r -d '' subdir; do
            local name
            name="$(basename "$subdir")"
            [ "$name" = "main" ] && continue
            if ! tree_is_idle "$subdir" "$min_idle_min"; then
                log "skip (write below it within ${min_idle_min}m): $subdir"
                continue
            fi
            log "prune stale bind-mount: $subdir"
            rm -rf --one-file-system "$subdir" 2>/dev/null \
                || sudo rm -rf --one-file-system "$subdir" 2>/dev/null || true
        done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    done
}

# Case table. The only thing separating this guard from `rm -rf` is which
# directories it refuses to touch, and that judgement was wrong in production.
# R3 and R6 are RED against the pre-fix selection
# (`find -mindepth 1 -maxdepth 1 -type d -mmin "+$min_idle_min"`); R1/R5/R7
# fail if a "fix" simply stops reclaiming, which would trade a deleted build
# for a full disk.
self_test() {
    local tmp root fails=0
    tmp="$(mktemp -d)"
    if [ -z "$tmp" ] || [ ! -d "$tmp" ]; then
        echo "self-test: mktemp -d failed" >&2
        return 1
    fi
    # shellcheck disable=SC2064  # $tmp must be expanded now, not at trap time
    trap "rm -rf '$tmp'" EXIT
    root="$tmp/aprender-ci"
    mkdir -p "$root"

    # R1: stale PR dir, nothing below it recent                  -> PRUNE
    mkdir -p "$root/1111/debug"
    touch -d '5 hours ago' "$root/1111/debug" "$root/1111"
    # R2: fresh PR dir, written just now                         -> KEEP
    mkdir -p "$root/2222/debug"
    # R3: THE REGRESSION. Parent mtime frozen 5h ago by the creation of its
    #     run-<ID> child; cargo is writing inside that child RIGHT NOW.
    mkdir -p "$root/3333/run-99/debug/deps"
    : > "$root/3333/run-99/debug/deps/libfoo.rlib"
    touch -d '5 hours ago' "$root/3333/run-99" "$root/3333"   # parents only
    # R4: "main" is exempt even when stale                       -> KEEP
    mkdir -p "$root/main/run-1/debug"
    touch -d '5 hours ago' "$root/main/run-1/debug" "$root/main/run-1" "$root/main"
    # R5: pre-isolation orphan                                   -> PRUNE always
    mkdir -p "$root/debug/deps"
    # R6: merge-queue ref. PR_OR_REF carries slashes, so the depth-1 dir is
    #     `gh-readonly-queue` and the live build is THREE levels below it.
    mkdir -p "$root/gh-readonly-queue/main/pr-2554-abc/run-77/debug"
    : > "$root/gh-readonly-queue/main/pr-2554-abc/run-77/debug/live.rlib"
    touch -d '5 hours ago' \
        "$root/gh-readonly-queue/main/pr-2554-abc/run-77" \
        "$root/gh-readonly-queue/main/pr-2554-abc" \
        "$root/gh-readonly-queue/main" \
        "$root/gh-readonly-queue"
    # R7: stale run-<ID> tree, no live write                     -> PRUNE
    mkdir -p "$root/7777/run-1/debug"
    touch -d '5 hours ago' "$root/7777/run-1/debug" "$root/7777/run-1" "$root/7777"

    BIND_MOUNT_ROOTS="$root" prune_bind_mount_target_roots 60 >/dev/null 2>&1 || true

    check_row() {  # label, path, gone|kept
        if [ "$3" = "gone" ]; then
            if [ -e "$2" ]; then
                echo "FAIL $1: expected PRUNED, survived"
                fails=$((fails + 1))
            else
                echo "ok   $1 (pruned)"
            fi
        elif [ -e "$2" ]; then
            echo "ok   $1 (kept)"
        else
            echo "FAIL $1: expected KEPT, was DELETED"
            fails=$((fails + 1))
        fi
    }
    check_row "R1 stale PR dir"                  "$root/1111"                               gone
    check_row "R2 fresh PR dir"                  "$root/2222"                               kept
    check_row "R3 live build, stale parent"      "$root/3333/run-99/debug/deps/libfoo.rlib" kept
    check_row "R4 main is exempt"                "$root/main"                               kept
    check_row "R5 orphan debug"                  "$root/debug"                              gone
    check_row "R6 live merge-queue build"        \
        "$root/gh-readonly-queue/main/pr-2554-abc/run-77/debug/live.rlib"                   kept
    check_row "R7 stale run-<ID> tree"           "$root/7777"                               gone

    if [ "$fails" -gt 0 ]; then
        echo "SELF-TEST FAILED: $fails of 7 rows" >&2
        return 1
    fi
    echo "SELF-TEST PASSED: 7 rows"
    return 0
}

case "${1:-}" in
    --pre-job)
        pct="$(disk_pct)"
        if [ "$pct" -ge "$HIGH_WATER_PCT" ]; then
            log "pre-job: / at ${pct}% ≥ ${HIGH_WATER_PCT}% — pruning target/ dirs"
            prune_target_dirs aggressive
            # 60-min idle floor spares fresh PR dirs (which would be in-flight
            # right now) but reclaims anything that's been untouched for an hour.
            prune_bind_mount_target_roots 60
        fi
        ;;
    --nightly)
        log "nightly: / at $(disk_pct)% — pruning target/ older than ${STALE_DAYS}d"
        prune_target_dirs stale
        prune_bind_mount_target_roots "$(( STALE_DAYS * 24 * 60 ))"
        ;;
    --self-test)
        self_test
        ;;
    *)
        echo "usage: $0 --pre-job | --nightly | --self-test" >&2
        exit 2
        ;;
esac
