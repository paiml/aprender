#!/usr/bin/env bash
# Fail-closed observability for the nightly speed lane.
#
# A speed beat reports by printing a single measurement line, e.g.
#   BEAT-SKLEARN-GAUSSIANNB-SPEED: apr=54.794ms sklearn=116.132ms ratio=0.472 ...
#
# A beat that printed NO such line did not quietly pass - it never ran, or it
# died before it could measure. Both look identical to "green" if you only
# check step exit codes, and that is precisely how the 2026-07-27/28 blackout
# hid a genuinely breaching GaussianNB beat behind a broken LinReg baseline.
#
# So: silence is red. This asserts every expected marker actually appeared.
#
# The expected list is derived from the tests themselves rather than hardcoded
# twice, so adding a beat to the workflow without wiring its marker (or renaming
# a marker) cannot drift out of sync unnoticed.
set -euo pipefail

BEAT_LOG="${1:?usage: check_beat_measurements.sh <beat-log-path>}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT" || exit 2

# Every speed beat the nightly workflow invokes, as `--test <name>` references.
# Same extraction idiom as scripts/check_beats_gated.sh.
INVOKED=$(grep -ohE '\-\-test[[:space:]]+beat_[A-Za-z0-9_]+' \
              .github/workflows/beat-speed-nightly.yml 2>/dev/null \
          | sed -E 's/--test[[:space:]]+//' | sort -u)

n_invoked=$(printf '%s\n' "$INVOKED" | grep -c '[^[:space:]]' || true)

if [ "$n_invoked" -eq 0 ]; then
    echo "✗ check_beat_measurements: found no '--test beat_*' references in the workflow" >&2
    echo "  Either the workflow was restructured or this check has drifted - both need a human." >&2
    exit 2
fi

if [ ! -f "$BEAT_LOG" ]; then
    echo "::error::check_beat_measurements: no measurement log at '$BEAT_LOG'" >&2
    echo "  $n_invoked speed beats were configured and NOT ONE produced output." >&2
    echo "  This is the blackout signature: the lane is dark, not green." >&2
    exit 1
fi

missing=0
found=0
while IFS= read -r test_name; do
    [ -n "$test_name" ] || continue
    # tests/beat_sklearn_gaussiannb_speed.rs -> its BEAT-* marker.
    src=$(find crates -path "*/tests/${test_name}.rs" -type f 2>/dev/null | head -1)
    if [ -z "$src" ]; then
        echo "✗ ${test_name}: workflow invokes it but no crates/*/tests/${test_name}.rs exists" >&2
        missing=$((missing + 1))
        continue
    fi
    marker=$(grep -ohE 'BEAT-[A-Z0-9-]+' "$src" | head -1)
    if [ -z "$marker" ]; then
        echo "✗ ${test_name}: no BEAT-* marker string in ${src} - it cannot report a measurement" >&2
        missing=$((missing + 1))
        continue
    fi
    if grep -q "${marker}:" "$BEAT_LOG"; then
        echo "  ✓ ${marker}: $(grep -m1 "${marker}:" "$BEAT_LOG" | sed "s/.*${marker}: //")"
        found=$((found + 1))
    else
        echo "✗ ${marker}: NO MEASUREMENT REPORTED (test ${test_name} went dark)" >&2
        missing=$((missing + 1))
    fi
done < <(printf '%s\n' "$INVOKED")

if [ "$missing" -gt 0 ]; then
    echo "::error::check_beat_measurements: ${missing}/${n_invoked} speed beats reported no measurement" >&2
    echo "  A beat that reports nothing is not a passing beat. Treating the lane as RED." >&2
    exit 1
fi

echo "✓ check_beat_measurements: all ${found} speed beats reported a measurement"
