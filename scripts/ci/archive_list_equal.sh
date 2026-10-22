#!/usr/bin/env bash
# FLOW-010 P1: the nextest archive must hold exactly the tests the in-job build lists.
# usage: archive_list_equal.sh LIVE.json ARCHIVE.json   (both `nextest list --message-format json`)
#        archive_list_equal.sh --self-test
# Exit 0 = the matching test ids are identical AND non-empty; 1 = differ or empty.
set -euo pipefail

ids() {
  jq -r '."rust-suites" | to_entries[] | .key as $b | .value.testcases | to_entries[]
         | select(.value."filter-match".status == "matches") | "\($b)::\(.key)"' "$1" | LC_ALL=C sort -u
}

check() {
  local a b
  a="$(ids "$1")"; b="$(ids "$2")"
  if [ -z "$a" ] || [ -z "$b" ]; then
    printf 'archive_list_equal: empty id list (live=%s archive=%s) -- a vacuous match is not a match\n' \
      "$(printf '%s' "$a" | grep -c . || true)" "$(printf '%s' "$b" | grep -c . || true)" >&2
    return 1
  fi
  if [ "$a" != "$b" ]; then
    printf 'archive_list_equal: lists differ (live=%s archive=%s)\n' \
      "$(printf '%s\n' "$a" | wc -l)" "$(printf '%s\n' "$b" | wc -l)" >&2
    diff <(printf '%s\n' "$a") <(printf '%s\n' "$b") | head -20 >&2 || true
    return 1
  fi
  printf 'archive_list_equal: %s ids identical\n' "$(printf '%s\n' "$a" | wc -l)"
}

self_test() {
  local d; d="$(mktemp -d)"
  mk() { jq -n --argjson t "$1" '{"rust-suites":{"c::lib":{testcases:($t|map({key:.,value:{"filter-match":{status:"matches"}}})|from_entries)}}}'; }
  mk '["a","b","c"]' > "$d/x.json"; mk '["a","b","c"]' > "$d/y.json"
  mk '["a","b"]' > "$d/drop.json"; mk '["a","b","c","d"]' > "$d/add.json"; mk '[]' > "$d/empty.json"
  check "$d/x.json" "$d/y.json" >/dev/null || { echo "FAIL: equal lists rejected" >&2; return 1; }
  for m in drop add empty; do
    if check "$d/x.json" "$d/$m.json" >/dev/null 2>&1; then echo "FAIL: mutant $m accepted" >&2; return 1; fi
  done
  if check "$d/empty.json" "$d/empty.json" >/dev/null 2>&1; then echo "FAIL: two empty lists accepted" >&2; return 1; fi
  rm -rf "${d:?}"
  echo "archive_list_equal self-test: ok (equal passes; drop/add/empty/both-empty rejected)"
}

if [ "${1:-}" = "--self-test" ]; then self_test; else check "$@"; fi
