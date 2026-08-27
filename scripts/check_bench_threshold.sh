#!/usr/bin/env bash
#
# check_bench_threshold.sh — the threshold rule is DERIVED, and the rule it
# replaced is falsified by execution rather than by argument (PARITY-008,
# aprender#2675).
#
# APR-BENCH-RFC-001 / #2588 BENCH-003 specifies
# `threshold_host = 3 * pooled_relative_stddev`. This guard runs the
# falsification every time, so the correction cannot quietly rot back:
#
#   · 3-sigma POOLED does not catch the one regression this repo has on
#     record (19.306% actual against a 19.74% derived band, reconstructing the
#     19.836% documented). Pooling is what breaks it: two hosts that merely
#     DIFFER -- neither unhealthy -- inflate the dispersion until the band
#     swallows a real regression.
#   · The bootstrap floor, derived per host from that host's own raw samples,
#     does catch it.
#   · And a floor derived from very few samples is both OPTIMISTIC and
#     UNSTABLE, which is why one is never armed below the minimum sample count.
#
# A guard that merely asserted "we use bootstrap" would be a comment. This one
# re-derives both numbers on every run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

DERIVER="scripts/lib/bench_threshold.py"
[ -f "$DERIVER" ] || { printf 'FAIL  %s is missing\n' "$DERIVER"; exit 2; }

printf -- '--- bench threshold derivation --------------------------------------\n'
out=$(python3 "$DERIVER" --falsify 2>&1)
rc=$?
printf '%s\n' "$out" | grep -E '^(ok|FAIL)' | sed 's/^/  /'

if [ "$rc" -ne 0 ]; then
    printf '\nFAIL  the falsification did not hold. Either the derivation changed or\n'
    printf '      the correction was reverted; both need a human. Full output:\n'
    printf '%s\n' "$out" | sed 's/^/      /'
    exit 1
fi

printf '\nPASS  bootstrap catches the documented regression, 3-sigma pooled does\n'
printf '      not, and the derived floor stabilises as n grows.\n'
exit 0
