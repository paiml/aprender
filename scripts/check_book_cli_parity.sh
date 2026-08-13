#!/usr/bin/env bash
# FALSIFY-BOOK-CLI-PARITY-001 — every `apr <cmd>` has a chapter at book/src/cli/<cmd>.md.
# Per BOOK-CLOSEOUT-001 § Phase 4.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# This gate reads `apr --help` and asserts a chapter exists for every subcommand.
# It used to resolve the binary as `${APR:-/home/noah/.cargo/bin/apr}`, falling
# back to `$(which apr)` — one developer's home directory, then PATH. Both are
# the #2357 defect: the command list it compared the book against came from
# whatever binary happened to be installed, not from this commit. A chapter
# added for a NEW subcommand would have been reported missing, and a chapter for
# a DELETED one reported present, with no way to tell from the output.
. scripts/apr_bin.sh || exit 1

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
