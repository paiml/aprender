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

case "${1:-}" in
    --pre-job)
        pct="$(disk_pct)"
        if [ "$pct" -ge "$HIGH_WATER_PCT" ]; then
            log "pre-job: / at ${pct}% ≥ ${HIGH_WATER_PCT}% — pruning target/ dirs"
            prune_target_dirs aggressive
        fi
        ;;
    --nightly)
        log "nightly: / at $(disk_pct)% — pruning target/ older than ${STALE_DAYS}d"
        prune_target_dirs stale
        ;;
    *)
        echo "usage: $0 --pre-job | --nightly" >&2
        exit 2
        ;;
esac
