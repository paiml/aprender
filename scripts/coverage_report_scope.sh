#!/usr/bin/env bash
# scripts/coverage_report_scope.sh -- print the explicit `-p <crate>` list that scopes a
# `cargo llvm-cov report` (#4023). The bash port of scripts/coverage_report_scope.py, so a
# new caller adds no Python (C301); the .py stays for the callers that predate this file.
#
# `cargo llvm-cov report` takes its package scope from the CURRENT package. This repo's root
# Cargo.toml is also a package (the `apr` facade), so an unscoped report covers only the facade
# and comes out empty (coverage-nightly run 35892421393). 0.9.0 rejects `report --exclude` and
# older versions `report --workspace`; an explicit `-p` list works on both. The list is DERIVED
# from `cargo metadata --no-deps`, never kept by hand.
#
# Usage:  coverage_report_scope.sh [--exclude NAME ...]
#         coverage_report_scope.sh --self-test
# Fails (rc 1, nothing on stdout) on an unknown --exclude name, an empty list, or a metadata
# or jq failure -- a caller's `$(...)` must never get an empty scope.
set -euo pipefail

scope() { # metadata-json, then exclude names
  local meta=$1; shift
  local excl unknown names
  excl=[]; [ $# -eq 0 ] || excl=$(printf "%s\n" "$@" | jq -R . | jq -s .) || return 1
  unknown=$(jq -r --argjson ex "$excl" '[.packages[].name] as $n | $ex | .[] | select(. as $e | $n | index($e) | not)' <<<"$meta") || return 1
  if [ -n "$unknown" ]; then
    echo "coverage_report_scope: --exclude names no workspace member: $(tr '\n' ' ' <<<"$unknown")" >&2
    return 1
  fi
  names=$(jq -r --argjson ex "$excl" '[.packages[].name | select(. as $p | $ex | index($p) | not)] | sort | map("-p " + .) | join(" ")' <<<"$meta") || return 1
  if [ -z "$names" ]; then
    echo "coverage_report_scope: no workspace members left to report" >&2
    return 1
  fi
  printf '%s\n' "$names"
}

self_test() {
  local meta='{"packages":[{"name":"b"},{"name":"a"},{"name":"gpu"}]}' fails=0 out rc
  row() { # name want-rc want-out args...
    local n=$1 wrc=$2 wout=$3; shift 3
    out=$(scope "$meta" "$@" 2>/dev/null) && rc=0 || rc=$?
    if [ "$rc" = "$wrc" ] && [ "$out" = "$wout" ]; then echo "ok    $n"
    else echo "FAIL  $n: rc=$rc out='$out' (want rc=$wrc out='$wout')"; fails=$((fails + 1)); fi
  }
  row "all members, sorted"        0 "-p a -p b -p gpu"
  row "one excluded"               0 "-p a -p b" gpu
  row "unknown exclude fails"      1 "" nosuch
  row "nothing left fails"         1 "" a b gpu
  meta='not json'
  row "bad metadata fails"         1 ""
  [ "$fails" -eq 0 ] && { echo "SELF-TEST PASSED (5 rows)"; return 0; }
  echo "SELF-TEST FAILED ($fails)"; return 1
}

main() {
  local -a ex=()
  if [ "${1:-}" = "--self-test" ]; then self_test; return; fi
  while [ $# -gt 0 ]; do
    case "$1" in
      --exclude) [ $# -ge 2 ] || { echo "usage: $0 [--exclude NAME ...]" >&2; return 2; }; ex+=("$2"); shift 2 ;;
      *) echo "usage: $0 [--exclude NAME ...]" >&2; return 2 ;;
    esac
  done
  local meta
  meta=$(cargo metadata --no-deps --format-version 1) || return 1
  scope "$meta" "${ex[@]}"
}

main "$@"
