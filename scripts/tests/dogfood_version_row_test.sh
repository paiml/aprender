#!/usr/bin/env bash
# Case table for dogfood.sh::version_already_published_verdict (PMAT-1098, 0.67.0 train).
# Sources the function out of dogfood.sh without running the gate: the definition is extracted by
# name, so a rename breaks this test by name, never silently.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
def=$(awk '/^version_already_published_verdict\(\) \{/,/^\}/' scripts/dogfood.sh)
[ -n "$def" ] || { echo "FAIL: version_already_published_verdict not found in scripts/dogfood.sh"; exit 1; }
eval "$def"
fails=0
row() { # row <phase> <want PASS|FAIL>
  got=$(version_already_published_verdict "$1" aprender 0.67.0 | cut -d' ' -f1)
  if [ "$got" = "$2" ]; then printf 'ok    %-13s -> %s\n' "$1" "$got"; else printf 'FAIL  %-13s -> %s (want %s)\n' "$1" "$got" "$2"; fails=$((fails + 1)); fi
}
row post-publish PASS
row pre-publish  FAIL
row full         FAIL
[ "$fails" -eq 0 ] && { echo "self-test OK: 3 case(s)"; exit 0; }
echo "self-test FAILED: $fails case(s)"; exit 1
