#!/usr/bin/env bash
# Fixture surface: a case table with one case, for the relative-root test.
set -euo pipefail
case "${1:-}" in
  --selftest) echo "ok alpha" ;;
  *) echo "usage: x.sh --selftest" >&2; exit 2 ;;
esac
