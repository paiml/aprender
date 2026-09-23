#!/usr/bin/env bash
# scripts/check_coverage_log_complete.sh -- refuse a coverage verdict when a test binary was KILLED.
#
# `make coverage` runs `cargo llvm-cov test --ignore-run-fail` so one failing TEST cannot
# blank the number (#3839). But --ignore-run-fail also swallows a test BINARY that died by
# signal, and llvm-cov writes a binary's profile only when it exits: a killed binary's crate
# is then (mostly) absent from lcov.info, and the recipe used to print that partial total as a
# verdict. Measured on coverage-nightly run 35868368976 (yoga, 2026-09-23): earlyoom SIGTERMed
# `realizar` (aprender-serve --lib, VmRSS 25.7 GB of 28 GB) after 15,950 of 16,022 tests, and
# the job reported "76% ... REGRESSION" -- a number with the largest crate missing.
#
# Usage:  check_coverage_log_complete.sh <cargo-llvm-cov test log>   exit 0 complete, 1 killed
#         check_coverage_log_complete.sh --self-test                  the case table below
# A sourced-library-free, option-setting script: it is executed, never sourced.
set -euo pipefail

# cargo prints this for a test binary that did not exit normally; `(signal: N, ...)` is the
# killed case. A plain failing test exits with a status (`exit status: 101`), not a signal.
KILLED_RE="process didn't exit successfully: .*\(signal: [0-9]+"

check() {
  local log=$1
  if [ ! -r "$log" ]; then
    echo "check_coverage_log_complete: cannot read $log" >&2
    return 2
  fi
  local killed
  killed=$(grep -E "$KILLED_RE" "$log" | sed -E 's/.*deps\/([^ `]+).*(\(signal: [^)]*\)).*/\1 \2/' || true)
  if [ -n "$killed" ]; then
    echo "coverage DID NOT MEASURE: test binary(ies) were KILLED by a signal, so their crates' profiles"
    echo "are missing from lcov.info and any total would be partial. No coverage verdict."
    printf '   %s\n' "$killed"
    return 1
  fi
  return 0
}

self_test() {
  local dir rc fails=0
  dir=$(mktemp -d)
  trap 'rm -rf "${dir:?}"' RETURN
  # must_flag: the measured SIGTERM line (run 35868368976), and a SIGKILL variant
  printf '%s\n' "warning: process didn't exit successfully: \`/w/target/llvm-cov-target/debug/deps/realizar-5fd3723e2da2d1b3 --exact --skip 'x'\` (signal: 15, SIGTERM: termination signal)" > "$dir/sigterm.log"
  printf '%s\n' "error: process didn't exit successfully: \`/w/target/llvm-cov-target/debug/deps/probador-c099 --exact\` (signal: 9, SIGKILL: kill)" > "$dir/sigkill.log"
  # must_pass: an ordinary failing test (exit status, no signal) and a clean log
  printf '%s\n' "warning: process didn't exit successfully: \`/w/target/llvm-cov-target/debug/deps/probador-c099 --exact\` (exit status: 101)" > "$dir/exit101.log"
  printf '%s\n' "test result: ok. 405 passed; 0 failed" > "$dir/clean.log"
  for c in sigterm sigkill; do
    rc=0; check "$dir/$c.log" > /dev/null || rc=$?
    [ "$rc" -eq 1 ] || { echo "SELF-TEST FAIL: $c must be flagged (rc=$rc)"; fails=$((fails + 1)); }
  done
  for c in exit101 clean; do
    rc=0; check "$dir/$c.log" > /dev/null || rc=$?
    [ "$rc" -eq 0 ] || { echo "SELF-TEST FAIL: $c must pass (rc=$rc)"; fails=$((fails + 1)); }
  done
  if [ "$fails" -eq 0 ]; then echo "check_coverage_log_complete self-test: 4/4 cases OK"; return 0; fi
  return 1
}

case "${1:-}" in
  --self-test) self_test ;;
  "") echo "usage: $0 <log> | --self-test" >&2; exit 2 ;;
  *) check "$1" ;;
esac
