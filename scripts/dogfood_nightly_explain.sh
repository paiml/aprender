#!/usr/bin/env bash
# Usage: dogfood_nightly_explain.sh <dogfood-log> <rc>
# #6798: say WHY a nightly dogfood run exited non-zero, so the step never fails silently.
# Prints the failing gates (the NO-GO block) or, when the run died before any verdict, says so
# and shows the last lines of the log.
set -uo pipefail
log="${1:-}" rc="${2:-?}"
if [ -z "$log" ] || [ ! -r "$log" ]; then
  echo "dogfood nightly rc=$rc: log unreadable (${log:-unset}); no gate can be named"
  exit 0
fi
if grep -q '^VERDICT:' "$log"; then
  echo "dogfood nightly rc=$rc: failing gates:"
  sed -n '/^NO-GO/,/^VERDICT:/p' "$log" | head -n 40
else
  echo "dogfood nightly rc=$rc: exited with no VERDICT line (died before a verdict); last lines:"
  tail -n 20 "$log"
fi
