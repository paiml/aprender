#!/usr/bin/env bash
# FALSIFY-BOOK-LIB-EXAMPLE-001 — every lib chapter has at least one
# fenced rust code block. Per BOOK-CLOSEOUT-001 § Phase 3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FENCE_RUST=$'^```rust'

fail=0
for f in book/src/lib/*.md; do
  if ! grep -qE "$FENCE_RUST" "$f"; then
    echo "FAIL: $f has no fenced rust example"
    fail=$((fail + 1))
  fi
done

if [ "$fail" -gt 0 ]; then
  echo ""
  echo "FALSIFY-BOOK-LIB-EXAMPLE-001: FAIL ($fail chapters without runnable example)"
  exit 1
fi

total=$(ls book/src/lib/*.md | wc -l)
echo "FALSIFY-BOOK-LIB-EXAMPLE-001: PASS (all $total lib chapters have runnable examples)"
