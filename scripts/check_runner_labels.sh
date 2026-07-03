#!/usr/bin/env bash
# Poka-yoke: every inline self-hosted CI job MUST pin a discriminating runner
# label, so a job can never silently land on the wrong self-hosted runner.
#
# Root cause it guards (aprender#2269): GitHub auto-assigns self-hosted/Linux/X64
# as DEFAULT labels that cannot be removed, so a GPU dev runner (lambda-4090)
# matched the bare `[self-hosted, X64, Linux]` selector and the mutants job ran
# there — a box with no sovereign-ci registry — failing non-deterministically.
#
# Rule: any `runs-on:` that names `self-hosted` must ALSO name one of:
#   - clean-room  (the provisioned sovereign-ci pool: registry + cached image)
#   - a GPU label: cuda | gpu | rtx4090 | ada | blackwell | gb10
#   - a macOS label: apple-silicon | m4
# Reusable-workflow jobs (`uses:`) have no `runs-on` and are naturally exempt.
# GitHub-hosted jobs (ubuntu-latest, …) don't name self-hosted and are exempt.
set -euo pipefail
cd "$(dirname "$0")/.."

DISCRIM='clean-room|cuda|gpu|rtx4090|ada|blackwell|gb10|apple-silicon|m4'
fail=0

while IFS=: read -r file line sel; do
  # Only inline self-hosted selectors.
  printf '%s' "$sel" | grep -q 'self-hosted' || continue
  if ! printf '%s' "$sel" | grep -qE "$DISCRIM"; then
    echo "::error file=${file},line=${line}::self-hosted job lacks a discriminating runner label (need clean-room or a GPU/macOS label): ${sel# }"
    fail=1
  fi
done < <(grep -rnE '^[[:space:]]*runs-on:.*self-hosted' .github/workflows/*.yml 2>/dev/null)

if [ "$fail" -ne 0 ]; then
  echo "FAIL: pin each self-hosted job to a provisioned pool (clean-room) or a GPU/macOS label." >&2
  echo "      Bare [self-hosted, X64, Linux] can land on ANY self-hosted runner (incl. GPU dev boxes)." >&2
  exit 1
fi
echo "✓ check_runner_labels: every self-hosted job pins a discriminating runner label"
