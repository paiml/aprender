#!/usr/bin/env bash
# story_pmat_hunt.sh - the qwen story's eight pmat hunts, without the story (#4875).
#
# qwen-story-daily starts from the nightly pick, as coverage-nightly does, and
# coverage takes an hour or more while the story job is capped at 30 minutes. So
# the story's hunts had no coverage file for their commit on most nights, and
# their gaps read not_measured. qwen-hunt-nightly starts from Coverage Nightly
# finishing instead, downloads that run's coverage JSON, and runs this script:
# the same eight hunts (STORY_HUNTS in lib_story_pmat.sh) against the night's C.
# It needs pmat and the checkout, not apr, a GPU or a model.
#
# Env: STORY_COVERAGE_FILE / STORY_COVERAGE_SHA as lib_story_pmat.sh reads them;
#      PMAT_HUNT=1 is forced here (this script exists only to hunt).
# Exit: 0 every hunt carried rows with measured gaps; 2 any hunt failed or was
#       not_measured; 1 the library or pmat is missing.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

FAIL=0
FAILED_BEATS=()
emit_fail() { FAIL=$((FAIL + 1)); FAILED_BEATS+=("$1"); printf 'FAIL  %s  -  %s\n' "$1" "$2"; }

# shellcheck source=scripts/lib_story_pmat.sh
. scripts/lib_story_pmat.sh || exit 1

# pmat_hunt returns 0 without a word when pmat is absent; here that would be a
# green run that hunted nothing.
command -v "${PMAT_BIN:-pmat}" >/dev/null 2>&1 || { printf 'story_pmat_hunt: %s not found\n' "${PMAT_BIN:-pmat}"; exit 1; }
export PMAT_HUNT=1

n=0
while IFS= read -r _; do
  n=$((n + 1))
  story_hunt "$n"
done <<< "$STORY_HUNTS"

printf '\n=== pmat hunt: %d beats, %d FAIL ===\n' "$n" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  for b in "${FAILED_BEATS[@]}"; do printf '   - %s\n' "$b"; done
  exit 2
fi
exit 0
