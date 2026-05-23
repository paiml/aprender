#!/usr/bin/env bash
# FALSIFY-BOOK-LIB-PARITY-001 — every public module in aprender-core
# has a chapter at book/src/lib/<mod>.md. Per BOOK-CLOSEOUT-001 § Phase 3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Get every `pub mod NAME;` from aprender-core's lib.rs
mods=$(grep -E "^pub mod " crates/aprender-core/src/lib.rs | awk '{print $3}' | tr -d ';')

missing=0
total=0
for m in $mods; do
  total=$((total+1))
  if [ ! -f "book/src/lib/${m}.md" ]; then
    echo "FAIL: book/src/lib/${m}.md does not exist (aprender::${m} has no chapter)"
    missing=$((missing+1))
  fi
done

covered=$((total - missing))
echo ""
echo "Coverage: ${covered}/${total} aprender-core public modules have a chapter (${missing} missing)"

if [ "$missing" -gt 0 ]; then
  echo "FALSIFY-BOOK-LIB-PARITY-001: FAIL"
  exit 1
fi

echo "FALSIFY-BOOK-LIB-PARITY-001: PASS"
