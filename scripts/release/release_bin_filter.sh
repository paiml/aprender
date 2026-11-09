#!/usr/bin/env bash
# #6502: drop packages Binary Release must never ship (publish=false, *-ci-tools) from a
# `cargo metadata --format-version 1` document before nightly_manifest.py derives bins.
# Usage: release_bin_filter.sh <metadata.json>   (filtered JSON on stdout)
# Used by .github/workflows/binary-release.yml and scripts/tests/release_bin_filter_test.sh.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 <metadata.json>" >&2; exit 2; }
jq '
  [.packages[] | select(.publish != [] and (.name | endswith("-ci-tools") | not)) | .id] as $keep
  | .packages |= map(select(.id as $i | $keep | index($i)))
  | if .workspace_members then .workspace_members |= map(select(. as $i | $keep | index($i))) else . end
' "$1"
