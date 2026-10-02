#!/usr/bin/env bash
# check_cb200_head_vs_base.sh -- run scripts/cb200_head_vs_base.sh's case table (PMAT-4641) on every PR.
# The helper needs a pinned pmat and a base ref, so the guard runner (which runs every check_*.sh bare)
# gets the self-test, which drives a fake pmat through the same code the release uses.
set -euo pipefail
case "${1:-}" in -h|--help) echo "usage: bash scripts/check_cb200_head_vs_base.sh"; exit 0 ;; esac
exec bash "$(dirname "$0")/cb200_head_vs_base.sh" --case-table
