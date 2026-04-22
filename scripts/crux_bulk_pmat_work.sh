#!/usr/bin/env bash
# Bulk-create pmat work tickets for every CRUX story with status == "missing".
# Idempotent via tag-based existence check (tag: crux-{id_lower}).
#
# Usage:
#   scripts/crux_bulk_pmat_work.sh [--dry-run]
#
# Contract reference: master §12.2, FALSIFY-CRUX-007.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER="$ROOT/contracts/crux-competitive-research-ux-v1.yaml"

if ! command -v yq >/dev/null 2>&1; then
  echo "yq not installed — aborting" >&2
  exit 1
fi

if ! command -v pmat >/dev/null 2>&1; then
  echo "pmat not installed — aborting" >&2
  exit 1
fi

# Cache current ticket list once for O(1) idempotence checks.
EXISTING_LIST=$(mktemp)
trap 'rm -f "$EXISTING_LIST"' EXIT
pmat work list 2>/dev/null > "$EXISTING_LIST" || true

# Extract missing stories: id	title	demand	competitor
rows=$(yq -o json '.stories' "$MASTER" | python3 "$ROOT/scripts/crux_missing_stories.py")

total=0
created=0
skipped=0

while IFS=$'\t' read -r id title score competitor; do
  [[ -z "$id" ]] && continue
  total=$((total + 1))

  cat=$(echo "$id" | cut -d- -f2)
  tag_id=$(echo "$id" | tr 'A-Z' 'a-z')

  case "$score" in
    5) prio=critical ;;
    4) prio=high ;;
    3) prio=medium ;;
    *) prio=low ;;
  esac

  tags="crux,gap,crux-${cat},competitor-${competitor},${tag_id}"
  desc="Contract: contracts/crux-${id#CRUX-}-v1.yaml | Competitor: ${competitor} | Demand: ${score}/5 | Parent: crux-competitive-research-ux-v1"

  # Idempotence: skip if a ticket titled "CRUX gap: <id> —" already exists.
  # pmat work list has no tag filter; grep the full list (cached once below).
  if grep -qF "CRUX gap: $id " "$EXISTING_LIST"; then
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY] pmat work add 'CRUX gap: $id — $title' -p $prio -t '$tags'"
  else
    pmat work add "CRUX gap: $id — $title" \
      -d "$desc" \
      -p "$prio" \
      -t "$tags" >/dev/null
  fi
  created=$((created + 1))
done <<< "$rows"

echo "Total missing: $total, created: $created, skipped (existing): $skipped"
