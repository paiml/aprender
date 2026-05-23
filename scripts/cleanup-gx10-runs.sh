#!/usr/bin/env bash
# Reclaim disk on gx10 by removing old distill runs.
#
# gx10 was at 98% disk during the PMAT-704 cascade post-mortem (only 21 GB
# free). Stage D 50K writes ~12 GB of checkpoints + final model; the
# dispatch-distill-stage-d.sh preflight requires >= 15 GB free. This
# script identifies + removes old runs older than N days.
#
# Usage:
#   ./scripts/cleanup-gx10-runs.sh                # dry-run, lists candidates
#   ./scripts/cleanup-gx10-runs.sh --apply        # actually delete
#   AGE_DAYS=7 ./scripts/cleanup-gx10-runs.sh     # 7-day cutoff (default 3)
#
# Safety:
#   - Always dry-run first (no --apply flag): prints what WOULD be deleted.
#   - Excludes runs younger than AGE_DAYS (default: 3).
#   - Excludes runs containing the "production" marker file (touch
#     `runs/<name>/.production` to permanently exclude).
#   - Excludes the current most-recent run regardless of age (in case it's
#     still in flight when the operator forgot to set .production).

set -euo pipefail

GX10_HOST="${GX10_HOST:-gx10}"
GX10_USER="${GX10_USER:-noah}"
RUNS_DIR="${RUNS_DIR:-/home/noah/runs}"
AGE_DAYS="${AGE_DAYS:-3}"
APPLY=0
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        --help|-h)
            grep -E '^# ' "$0" | sed 's/^# //;s/^#$//'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

echo "=== gx10 disk-cleanup preview ==="
echo "  target:    ${GX10_USER}@${GX10_HOST}:${RUNS_DIR}"
echo "  age cutoff: ${AGE_DAYS} days"
echo "  mode:       $( [ "$APPLY" = "1" ] && echo APPLY || echo DRY-RUN )"
echo

ssh "${GX10_USER}@${GX10_HOST}" bash <<REMOTE
    set -e
    if [ ! -d "${RUNS_DIR}" ]; then
        echo "ERROR: ${RUNS_DIR} does not exist on remote"
        exit 1
    fi

    echo "--- disk before ---"
    df -h "${RUNS_DIR}" | tail -1

    # Identify candidates: dirs older than AGE_DAYS, without .production marker,
    # and not the most-recent run.
    most_recent=\$(ls -1tdr "${RUNS_DIR}"/*/ 2>/dev/null | tail -1 || true)
    echo "--- exclusions ---"
    echo "  most recent (always kept): \${most_recent}"
    grep -l '.' "${RUNS_DIR}"/*/.production 2>/dev/null | sed 's|/.production||;s|^|  marked .production:  |' || echo "  (no .production-marked runs)"

    echo
    echo "--- candidates (older than ${AGE_DAYS} days, NOT excluded) ---"
    candidates=()
    while IFS= read -r dir; do
        # Skip the most-recent run.
        if [ "\$dir" = "\$most_recent" ]; then continue; fi
        # Skip .production-marked runs.
        if [ -f "\${dir}.production" ] || [ -f "\${dir%/}/.production" ]; then continue; fi
        size=\$(du -sh "\$dir" 2>/dev/null | cut -f1)
        printf '  %s  %s\n' "\$size" "\$dir"
        candidates+=("\$dir")
    done < <(find "${RUNS_DIR}" -maxdepth 1 -mindepth 1 -type d -mtime "+${AGE_DAYS}" 2>/dev/null | sort)

    if [ \${#candidates[@]} -eq 0 ]; then
        echo "  (no candidates)"
        echo
        echo "--- disk unchanged ---"
        df -h "${RUNS_DIR}" | tail -1
        exit 0
    fi

    if [ "${APPLY}" = "1" ]; then
        echo
        echo "--- applying ---"
        for dir in "\${candidates[@]}"; do
            echo "  rm -rf \$dir"
            rm -rf "\$dir"
        done
        echo
        echo "--- disk after ---"
        df -h "${RUNS_DIR}" | tail -1
    else
        echo
        echo "DRY-RUN: pass --apply to actually delete."
    fi
REMOTE
