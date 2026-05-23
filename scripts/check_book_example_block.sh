#!/usr/bin/env bash
# FALSIFY-BOOK-EXAMPLE-001 — every CLI chapter under book/src/cli/*.md
# has at least one fenced bash code block. Per BOOK-CLOSEOUT-001 § Phase 4.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Triple-backtick pattern via env var to avoid shell-quote hell
FENCE_BASH=$'^```bash'

fail=0
for f in book/src/cli/*.md; do
  if ! grep -qE "$FENCE_BASH" "$f"; then
    echo "FAIL: $f has no fenced bash example"
    fail=$((fail + 1))
  fi
done

if [ "$fail" -gt 0 ]; then
  echo ""
  echo "FALSIFY-BOOK-EXAMPLE-001: FAIL ($fail chapters without runnable example)"
  exit 1
fi

total=$(ls book/src/cli/*.md | wc -l)
echo "FALSIFY-BOOK-EXAMPLE-001: PASS (all $total CLI chapters have runnable examples)"
