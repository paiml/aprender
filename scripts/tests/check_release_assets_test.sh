#!/usr/bin/env bash
# check_release_assets_test.sh — falsifier for scripts/check_release_assets.sh
# (row 67-A1, PMAT-1098, issue #3082).
#
# WHY THIS EXISTS
# ---------------
# v0.66.0 shipped EIGHT `pv` tarballs and ZERO `apr` binaries. Nothing was red:
# `verify-cuda-assets` only ever asked about the two CUDA assets, and it was not
# reached because the build lane died at `gh: command not found`. No gate asserted
# the ASSET SET, so a release could be cut with none of the four `apr` binaries the
# operator requires (2026-09-10: CUDA and CPU × ARM and x86, one per tag).
#
# scripts/check_release_assets.sh is that assertion, and this file is its falsifier.
# It is deliberately NOT the script's own opinion of itself:
#
#   1. the script's `--selftest` case table is green and NOT vacuous (>= 5 rows);
#   2. the REGISTERED MUTATION, run here rather than trusted from there: a
#      COMPLETE fixture asset list is GREEN (exit 0), and the same list with
#      `apr-<tag>-aarch64-unknown-linux-gnu-cpu.tar.gz` removed is RED (exit 1)
#      AND NAMES the missing asset. Both polarities, because a checker that
#      exits 1 on everything passes the RED half on its own merits;
#   3. the ENV/NOT-A-PASS polarity: an unreadable asset list is exit 2, never 0.
#      A release checker that cannot read the release must not report `ok`;
#   4. the WIRING: binary-release.yml calls THIS script instead of an inline
#      per-asset loop, so the workflow and the release-day protocol share one
#      checker, and the four apr assets are required by name.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GUARD=scripts/check_release_assets.sh
WF=.github/workflows/binary-release.yml
TAG=v9.9.9

n=0; red=0
t() { # t <want-rc> <label> <cmd...>
  local want=$1 label=$2; shift 2
  local rc=0
  n=$((n + 1))
  "$@" >/dev/null 2>&1 || rc=$?
  if [ "$rc" = "$want" ]; then
    printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
  else
    printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; red=1
  fi
}

[ -f "$GUARD" ] || { printf 'FAIL  %s does not exist — the asset set is asserted by nothing\n' "$GUARD"; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# The COMPLETE set the operator's rule requires, written out by hand here so this
# file is an independent statement of the expectation, not a re-read of the guard's.
{
  for t_arch in x86_64 aarch64; do
    for flavour in cuda cpu; do
      printf 'apr-%s-%s-unknown-linux-gnu-%s.tar.gz\n' "$TAG" "$t_arch" "$flavour"
      printf 'apr-%s-%s-unknown-linux-gnu-%s.tar.gz.sha256\n' "$TAG" "$t_arch" "$flavour"
    done
  done
  for t_arch in x86_64 aarch64; do
    for libc in musl gnu; do
      printf 'pv-%s-%s-unknown-linux-%s.tar.gz\n' "$TAG" "$t_arch" "$libc"
      printf 'pv-%s-%s-unknown-linux-%s.tar.gz.sha256\n' "$TAG" "$t_arch" "$libc"
    done
  done
} > "$WORK/complete.txt"

MUTANT_NAME="apr-$TAG-aarch64-unknown-linux-gnu-cpu.tar.gz"
grep -vx "$MUTANT_NAME" "$WORK/complete.txt" > "$WORK/mutant.txt"
grep -vx "apr-$TAG-x86_64-unknown-linux-gnu-cuda.tar.gz.sha256" "$WORK/complete.txt" > "$WORK/nosha.txt"
grep -v '^pv-' "$WORK/complete.txt" > "$WORK/nopv.txt"

# 1. the guard's own table, and it is not vacuous
t 0 "$GUARD --selftest is green" bash "$GUARD" --selftest
t 0 "--selftest is not vacuous (>= 5 rows)" \
  bash -c "[ \"\$(bash '$GUARD' --selftest | grep -cE '^(ok|FAIL) ')\" -ge 5 ]"

# 2. the registered mutation, both polarities, run HERE
t 0 "complete asset set -> 0" bash "$GUARD" "$TAG" --assets-from "$WORK/complete.txt"
t 1 "MUTATION: aarch64 cpu tarball removed -> 1" bash "$GUARD" "$TAG" --assets-from "$WORK/mutant.txt"
t 0 "the mutation is NAMED, not merely counted" \
  bash -c "bash '$GUARD' '$TAG' --assets-from '$WORK/mutant.txt' 2>&1 | grep -q '$MUTANT_NAME'"
t 1 "a missing .sha256 is as fatal as a missing tarball" bash "$GUARD" "$TAG" --assets-from "$WORK/nosha.txt"
t 1 "the eight pv assets are required too" bash "$GUARD" "$TAG" --assets-from "$WORK/nopv.txt"

# 3. unreadable is ENV (2), never a pass
t 2 "an unreadable asset list is ENV (2), never 0" bash "$GUARD" "$TAG" --assets-from "$WORK/does-not-exist.txt"
t 2 "no tag is a usage error (2), never a pass" bash "$GUARD"

# 4. the wiring: the workflow shares this checker
t 0 "binary-release.yml calls the shared checker" \
  bash -c "grep -q 'check_release_assets.sh' '$WF'"
t 0 "the verify job requires all four apr assets (it is no longer cuda-only)" \
  bash -c "grep -q 'verify-apr-assets:' '$WF'"
t 0 "a build-apr-cpu lane exists" bash -c "grep -q 'build-apr-cpu:' '$WF'"
t 0 "a smoke-cpu lane exists" bash -c "grep -q 'smoke-cpu:' '$WF'"

printf '%s/%s rows\n' "$((n - red))" "$n"
[ "$red" = 0 ] || exit 1
exit 0
