#!/usr/bin/env bash
# FLOW-010 P1: the archive job's wiring must keep four properties (w5404 round 3).
# usage: archive_wiring_check.sh [SECTIONS.yml CI.yml]   |   --self-test
set -euo pipefail

check() {
  local sec="$1" ci="$2" bad=0
  # 1. the smoke refuses to pass having checked zero binaries
  grep -q '\[ "\$n" -gt 0 \]' "$sec" || { echo "wiring: smoke does not require n>0" >&2; bad=1; }
  # 2. the toolchain bind value is captured, then validated, never piped straight into sha256sum
  grep -q 'rustc -vV | sha256sum' "$sec" && { echo "wiring: rustc -vV piped into sha256sum (a failed run hashes to a valid value)" >&2; bad=1; }
  grep -q -- '--hash-toolchain "\$rustc_vv"' "$sec" || { echo "wiring: archive job does not validate the toolchain text" >&2; bad=1; }
  grep -q -- '--toolchain "\$rustc_vv"' "$sec" || { echo "wiring: shard does not validate the toolchain text" >&2; bad=1; }
  # 3. both ws-archive uploads overwrite (a re-run attempt must not hit a name conflict)
  [ "$(grep -c 'overwrite: true' "$sec")" -ge 1 ] || { echo "wiring: section upload lacks overwrite: true" >&2; bad=1; }
  awk '/name: ws-archive/{f=1} f&&/overwrite: true/{ok=1} END{exit !ok}' "$ci" || { echo "wiring: ci.yml ws-archive upload lacks overwrite: true" >&2; bad=1; }
  [ "$bad" = 0 ] && echo "archive_wiring_check: ok"
}

self_test() {
  local d; d="$(mktemp -d)"
  local sec=ci/sections.yml ci=.github/workflows/ci.yml
  check "$sec" "$ci" >/dev/null || { echo "FAIL: real tree rejected" >&2; return 1; }
  mut() { # name file sed-expr
    sed "$3" "$2" > "$d/$1"; cmp -s "$2" "$d/$1" && { echo "FAIL: mutant $1 did not apply" >&2; return 1; }; return 0; }
  mut s1 "$sec" 's/\[ "\$n" -gt 0 \]/true/' && { check "$d/s1" "$ci" >/dev/null 2>&1 && { echo "FAIL: mutant n>0 survived" >&2; return 1; }; }
  mut s2 "$sec" 's/--hash-toolchain "\$rustc_vv"/--hash-toolchain ""/' && { check "$d/s2" "$ci" >/dev/null 2>&1 && { echo "FAIL: mutant hash-toolchain survived" >&2; return 1; }; }
  mut s2b "$sec" 's/--toolchain "\$rustc_vv"/--other "$rustc_vv"/' && { check "$d/s2b" "$ci" >/dev/null 2>&1 && { echo "FAIL: mutant shard toolchain survived" >&2; return 1; }; }
  mut c3 "$ci" 's/overwrite: true/overwrite: false/' && { check "$sec" "$d/c3" >/dev/null 2>&1 && { echo "FAIL: mutant ci overwrite survived" >&2; return 1; }; }
  mut s3 "$sec" 's/overwrite: true/overwrite: false/' && { check "$d/s3" "$ci" >/dev/null 2>&1 && { echo "FAIL: mutant section overwrite survived" >&2; return 1; }; }
  rm -rf "${d:?}"
  echo "archive_wiring_check self-test: ok (5 mutants killed)"
}

if [ "${1:-}" = "--self-test" ]; then self_test; else check "${1:-ci/sections.yml}" "${2:-.github/workflows/ci.yml}"; fi
