#!/usr/bin/env bash
# LAB check (nightly only; never a PR, merge-queue or release job): a spec whose
# header says Status PROPOSED must carry an ISO date (YYYY-MM-DD) in its first
# 12 lines, so a stale proposal is measurable. Covers aprender#6604 #6628 #6629
# #6637 #6638 #6639 #6636 (spec-sweep rows "Status PROPOSED", epic #6809).
# Usage: check_spec_proposed_dated.sh [specs-dir]   rc 0 green, 1 violations, 2 scanned 0 files.
set -u
dir="${1:-docs/specifications}"
scanned=0 proposed=0 bad=0
while IFS= read -r f; do
  hdr=$(sed -n '1,12p' "$f")
  printf '%s\n' "$hdr" | grep -qiE '^[*[:space:]-]*status[*:[:space:]]+\**proposed' || { scanned=$((scanned+1)); continue; }
  scanned=$((scanned+1)); proposed=$((proposed+1))
  if ! printf '%s\n' "$hdr" | grep -qE '[0-9]{4}-[0-9]{2}-[0-9]{2}'; then
    echo "VIOLATION $f: Status PROPOSED with no YYYY-MM-DD date in the first 12 lines"; bad=$((bad+1))
  fi
done < <(find "$dir" -name '*.md' | sort)
echo "scanned=$scanned proposed=$proposed violations=$bad"
[ "$scanned" -gt 0 ] || exit 2
[ "$bad" -eq 0 ]
