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

# Prune bind-mount target roots (/mnt/nvme-raid0/targets/aprender-ci/*).
# Subdirs are per-PR target dirs isolated per task #134. Remove:
#   - "debug" subdir unconditionally (orphan from pre-isolation era)
#   - numeric PR-named subdirs with mtime older than MIN_IDLE_MIN minutes.
# Minute-resolution idle check protects in-flight CI bind-mounts; a container
# touching the dir keeps mtime current. Nightly uses $((STALE_DAYS*24*60));
# pre-job uses a 60-min floor so a full-disk situation can still reclaim most
# stale PR dirs while sparing fresh ones.
prune_bind_mount_target_roots() {
    local min_idle_min="$1"  # minutes since last mtime that counts as "stale"
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
        # Stale numeric PR dirs (mtime > min_idle_min). "main" is explicitly
        # preserved — it hosts push-to-main CI target cache and is legitimately
        # re-used; stale-days protection still applies via find -mmin below.
        while IFS= read -r -d '' subdir; do
            local name
            name="$(basename "$subdir")"
            [ "$name" = "main" ] && continue
            log "prune stale bind-mount: $subdir"
            rm -rf --one-file-system "$subdir" 2>/dev/null \
                || sudo rm -rf --one-file-system "$subdir" 2>/dev/null || true
        done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -mmin "+$min_idle_min" -print0 2>/dev/null)
    done
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
    *)
        echo "usage: $0 --pre-job | --nightly" >&2
        exit 2
        ;;
esac
