#!/bin/bash
# Self-hosted GitHub Actions runner disk guard.
# Two modes:
#   --pre-job        Called from ACTIONS_RUNNER_HOOK_JOB_STARTED. Only acts when
#                    disk usage on / exceeds HIGH_WATER_PCT (default 85).
#   --nightly        Called from systemd timer. Unconditional prune of idle
#                    target/ dirs older than STALE_DAYS (default 7) across all
#                    /home/noah/data/actions-runner*/_work trees.
#
# Safe by design: only touches target/, .rustup/tmp, .cache/cargo/registry/cache,
# and _tool. Never the repo checkout itself. Exits 0 even if nothing to prune.
set -euo pipefail

HIGH_WATER_PCT="${HIGH_WATER_PCT:-85}"
STALE_DAYS="${STALE_DAYS:-7}"
RUNNERS_ROOT="${RUNNERS_ROOT:-/home/noah/data}"
LOG_TAG="runner-disk-guard"

log() { logger -t "$LOG_TAG" -- "$*" || true; echo "[$LOG_TAG] $*"; }

disk_pct() { df --output=pcent / | tail -1 | tr -dc '0-9'; }

prune_target_dirs() {
    local mode="$1"  # "aggressive" or "stale"
    local total_freed_kb=0
    local before_kb after_kb

    before_kb="$(df --output=avail / | tail -1 | tr -dc '0-9')"

    # Find all */target/ under runner _work trees
    while IFS= read -r -d '' tgt; do
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
