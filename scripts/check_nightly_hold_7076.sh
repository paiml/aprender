#!/usr/bin/env bash
# #7076: nightly-bench `measure` is on HOLD (`if: false`) while it targets the
# intel,perf-solo lane. Fails if the hold is removed without removing this guard.
set -euo pipefail
f="${1:-.github/workflows/nightly-bench.yml}"
[ -f "$f" ] || { echo "FAIL: $f missing" >&2; exit 1; }
if ! awk '/^  measure:/{m=1;next} m&&/^  [a-z]/{m=0} m&&/^    if: false[[:space:]]*$/{ok=1} END{exit !ok}' "$f"; then
  echo "FAIL: $f job measure lacks 'if: false' hold (#7076)" >&2; exit 1
fi
grep -q '#7076' "$f" || { echo "FAIL: hold comment must name #7076" >&2; exit 1; }
echo "OK: nightly-bench measure held (#7076)"
