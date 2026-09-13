#!/usr/bin/env bash
# witness_diff.sh — BSE-11 nightly witness byte-compare
# (docs/specifications/build-system-enhancement.md, wave 3, `paiml/infra`).
#
# WHY THIS EXISTS
# ----------------
# The PP-26 batch-invariance witness (`cuda-nightly.yml`, `perf041-witness`,
# `scripts/perf041_batched_parity_probe.sh`) has produced the SAME
# `^PP-26-WITNESS: ` line(s) — a standing INVALID-CORRECTNESS band, not a
# regression — every night since 2026-08-30
# (docs/reports/work-history-delay-optimization-report.md). Until this
# ticket, a human on release day had only the job's GitHub status colour to
# go on, which is red for a known-and-unchanged failure exactly as it is red
# for a brand-new one, so every release day re-diagnosed the same non-news.
#
# This script answers exactly one question — "did the witness's own text
# change, byte for byte, since the file it is compared against?" — and
# nothing else. It does NOT parse JSON, does NOT read a `status` field, and
# does NOT special-case any word (including a colour word like "red" or
# "green" that might appear inside a witness line): two files that differ in
# even one byte are CHANGED, two that are byte-identical are UNCHANGED,
# regardless of what the differing (or non-differing) bytes happen to spell.
# That is what keeps this tool honest against the mutation the spec names —
# "compare status instead of bytes" — which would report UNCHANGED for two
# witnesses that agree on a status word but differ everywhere else.
#
# USAGE
#   witness_diff.sh <expected-line-file> <observed-line-file>
#
# OUTPUT (stdout, exactly one word, nothing else)
#   UNCHANGED   the two files are byte-identical
#   CHANGED     the two files differ in at least one byte
#
# EXIT STATUS
#   0   always, for both UNCHANGED and CHANGED — CHANGED is information for
#       the caller (a release checklist, a job summary) to act on, never a
#       failure this script itself judges.
#   2   usage error (wrong argument count, an argument is not a file, or an
#       argument is an EMPTY file — an empty witness is a broken probe, not
#       a verdict) — this is the caller's mistake, not a witness verdict, so
#       it is kept distinct from 0.
set -uo pipefail

usage() {
  echo "usage: $(basename "$0") EXPECTED_LINE_FILE OBSERVED_LINE_FILE" >&2
  exit 2
}

[ "$#" -eq 2 ] || usage

expected="$1"
observed="$2"

for f in "$expected" "$observed"; do
  if [ ! -f "$f" ]; then
    echo "::error::witness_diff.sh: not a file: $f" >&2
    exit 2
  fi
  # An empty witness file is never a legitimate "no change" - it means the
  # probe that should have produced a `PP-26-WITNESS: ` line produced
  # nothing (a broken grep, a probe crash upstream, a truncated artifact).
  # `cmp -s` on two empty files reports equal, which would silently print
  # UNCHANGED for exactly the failure this tool exists to surface. Reject
  # it here so the caller sees an error, never a false UNCHANGED.
  if [ ! -s "$f" ]; then
    echo "::error::witness_diff.sh: empty file: $f" >&2
    exit 2
  fi
done

# Byte comparison only — `cmp -s` reports equal/unequal without printing
# where, which is exactly the granularity this tool needs and nothing more.
if cmp -s -- "$expected" "$observed"; then
  echo "UNCHANGED"
else
  echo "CHANGED"
fi
exit 0
