#!/usr/bin/env bash
# coverage_killed_binaries.sh <test.log> — list the test binaries that a SIGNAL ended in a
# `cargo llvm-cov test --ignore-run-fail` run, one "<binary> <SIGNAME>" per line.
#
# Why: a binary killed by a signal writes no .profraw, so its whole crate reads as 0%
# covered. Coverage Nightly 36204739015 (2026-09-26, yoga): earlyoom SIGTERMed the
# aprender-serve lib tests at 15.2 GB RSS, the total fell 90% -> 76%, and the gate said
# "REGRESSION: coverage went DOWN". Nothing went down; a crate was not measured. The
# Makefile uses this list to say so instead.
#
# --self-test runs the must-match / must-not-match case table.
set -uo pipefail

extract() { # extract <file>
  grep -E "process didn't exit successfully: \`[^\`]+\` \(signal: [0-9]+, SIG[A-Z0-9]+" "$1" \
    | sed -E "s/.*successfully: \`([^ \`]+)[^\`]*\` \(signal: [0-9]+, (SIG[A-Z0-9]+).*/\1 \2/" \
    | sort -u
}

self_test() {
  local tmp fails=0 out
  tmp=$(mktemp) || return 2
  # must-match: the exact line from run 36204739015 (skips shortened), and a SIGKILL
  printf '%s\n' \
    "  process didn't exit successfully: \`/w/target/llvm-cov-target/debug/deps/realizar-5c448889829f98ca --exact --skip 'a::b'\` (signal: 15, SIGTERM: termination signal)" \
    "  process didn't exit successfully: \`/w/target/debug/deps/aprender-1a2b --exact\` (signal: 9, SIGKILL: kill)" \
    "  process didn't exit successfully: \`/w/target/debug/deps/trueno-9f\` (exit status: 101)" \
    "test gpu::signal::tests::handles_signal_15 ... ok" \
    "thread panicked: child ended (signal: 15, SIGTERM: termination signal)" \
    > "$tmp"
  out=$(extract "$tmp")
  rm -f "${tmp:?}"
  for want in "/w/target/llvm-cov-target/debug/deps/realizar-5c448889829f98ca SIGTERM" \
              "/w/target/debug/deps/aprender-1a2b SIGKILL"; do
    if grep -qxF "$want" <<< "$out"; then echo "  ok    matches: $want"; else echo "  FAIL  missed: $want"; fails=$((fails + 1)); fi
  done
  for deny in trueno-9f handles_signal_15 "thread panicked"; do
    if grep -qF "$deny" <<< "$out"; then echo "  FAIL  wrongly matched: $deny"; fails=$((fails + 1)); else echo "  ok    ignores: $deny"; fi
  done
  if [ "$(grep -c . <<< "$out")" -ne 2 ]; then echo "  FAIL  expected exactly 2 lines, got: $out"; fails=$((fails + 1)); fi
  [ "$fails" -eq 0 ] && echo "coverage_killed_binaries: self-test PASS" && return 0
  echo "coverage_killed_binaries: self-test FAIL ($fails)"; return 1
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  ""|-h|--help) echo "usage: $0 <test.log> | --self-test"; exit 2 ;;
esac
[ -r "$1" ] || { echo "coverage_killed_binaries: cannot read $1" >&2; exit 2; }
extract "$1"
exit 0
