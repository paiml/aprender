#!/usr/bin/env bash
# FALSIFY-BOOK-LINKCHECK-001 — `mdbook-linkcheck --standalone` reports
# zero `File not found:` errors. Per BOOK-CLOSEOUT-001 § Phase 1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/book"

# Install if missing (CI convenience)
if ! command -v mdbook-linkcheck >/dev/null 2>&1; then
  cargo install mdbook-linkcheck --locked --quiet
fi

# Strip ANSI; count
broken=$(mdbook-linkcheck --standalone 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -c "File not found:" || true)

echo "Broken file links: ${broken}"

if [ "$broken" -gt 0 ]; then
  echo ""
  echo "First 5 broken targets:"
  mdbook-linkcheck --standalone 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep "File not found:" | head -5
  echo ""
  echo "FALSIFY-BOOK-LINKCHECK-001: FAIL"
  exit 1
fi

echo "FALSIFY-BOOK-LINKCHECK-001: PASS"
