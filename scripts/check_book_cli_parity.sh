#!/usr/bin/env bash
# FALSIFY-BOOK-CLI-PARITY-001 — every `apr <cmd>` has a chapter at book/src/cli/<cmd>.md.
# Per BOOK-CLOSEOUT-001 § Phase 4.
set -euo pipefail

APR="${APR:-/home/noah/.cargo/bin/apr}"
if ! [ -x "$APR" ]; then
  APR="$(which apr)"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cmds=$("$APR" --help 2>&1 | awk '/^Commands:/{f=1; next} f && /^  [a-z]/{print $1}')

missing=0
for cmd in $cmds; do
  if [ ! -f "book/src/cli/${cmd}.md" ]; then
    echo "FAIL: book/src/cli/${cmd}.md does not exist (apr ${cmd} has no chapter)"
    missing=$((missing+1))
  fi
done

total=$(echo "$cmds" | wc -l)
covered=$((total - missing))
echo ""
echo "Coverage: ${covered}/${total} CLI subcommands have a chapter (${missing} missing)"

if [ "$missing" -gt 0 ]; then
  echo "FALSIFY-BOOK-CLI-PARITY-001: FAIL"
  exit 1
else
  echo "FALSIFY-BOOK-CLI-PARITY-001: PASS"
fi
