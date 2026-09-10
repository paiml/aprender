#!/usr/bin/env bash
# run_clean_room.sh — C8: the clean-room p1 gate, run from the infra checkout this repo
# sits beside (SPEC-2.0, driver v5; guide §5.3 hard gate). Requires ../infra; refuses otherwise.
#
#   bash scripts/run_clean_room.sh            # exit 0 iff `make -C ../infra/machines/clean-room clean-room-p1` exits 0
#   bash scripts/run_clean_room.sh --dry-run  # print what would run; exit 2 if ../infra is absent
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=run_clean_room
MAIN="$(dirname "$(cd "$(git -C "$ROOT" rev-parse --git-common-dir)" && pwd)")"   # the main checkout, whose sibling is infra (worktrees live elsewhere)
INFRA="${INFRA_DIR:-$(dirname "$MAIN")/infra}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
if [ ! -d "$INFRA/machines/clean-room" ]; then
    printf '%s: ENV - %s/machines/clean-room is missing; C8 cannot be evaluated here (clone paiml/infra beside this repo). Exit 2, never a pass.\n' "$PROG" "$INFRA" >&2; exit 2
fi
CMD=(make -C "$INFRA/machines/clean-room" clean-room-p1)
if [ "$DRY" = 1 ]; then printf 'would run: %s\n' "${CMD[*]}"; exit 0; fi
printf '=== C8 clean-room p1 (%s) ===\n' "${CMD[*]}"
"${CMD[@]}"
