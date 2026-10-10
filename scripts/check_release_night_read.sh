#!/usr/bin/env bash
# check_release_night_read.sh -- the case table of scripts/release/release_night_read.sh (#4701), the
# release-day read of the nightly train's verdict lanes for H: the deep test lanes, dogfood, and the
# model measure (the CRUX smoke since 0.71.0, or the full ladder).
#
# The reader is not wired into any release script yet, so nothing else runs it. This guard keeps its
# read honest until the switch: every row of its case table must pass (fixtures from nightly_train.sh's
# own bundles plus generated models-t1 artifacts of both measures): 108 rows, 4.2 s on a 32-core host
# at load 7.
#
# The planted mutants are NOT run here: `bash scripts/release/release_night_read.sh --mutants` (49 of
# them) took 3 min 30 s on the same host, every PR would pay it, and it only changes when the reader
# does. Run it after any edit to the reader; it exits 1 unless every mutant changes the file, parses,
# and breaks a row.
#
# Cargo-free: bash, awk, sha256sum, find and GNU date.
# Exit: 0 every row behaved · 1 a row broke · 2 ENV.
set -uo pipefail
# guard_tree.sh probes `--help` to decide whether to run a second mode; answer it before any work.
case "${1:-}" in -h|--help) printf 'usage: bash scripts/check_release_night_read.sh (no arguments: runs the case table)\n'; exit 0 ;; esac

PROG=check_release_night_read
ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
R="$ROOT/scripts/release/release_night_read.sh"
[ -f "$R" ] || { printf '%s: ENV - %s is missing\n' "$PROG" "$R" >&2; exit 2; }
for t in awk sha256sum find sed; do
    command -v "$t" > /dev/null 2>&1 || { printf '%s: ENV - %s is missing\n' "$PROG" "$t" >&2; exit 2; }
done
[ "$(date -u -d '2026-10-01 -1 day' +%F 2> /dev/null)" = 2026-09-30 ] \
    || { printf '%s: ENV - date is not GNU date (the streak steps back one UTC day)\n' "$PROG" >&2; exit 2; }

bash "$R" --self-test || { printf '%s: FAIL - a row of the case table broke\n' "$PROG"; exit 1; }
printf '%s: PASS\n' "$PROG"
