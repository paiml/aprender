#!/usr/bin/env bash
# check_shard_count_single_source.sh -- the workspace-test shard count has ONE source.
# The pins in ci/sections.yml `matrix-pins.workspace-test-shard` own it; ci.yml's matrix
# (`shard:` list and `shards:`) must agree, and no literal count may survive in the sharded
# commands (a stale literal silently drops or duplicates tests). Pure bash/awk/grep: the
# explicit-test container has no python3.
# Usage: check_shard_count_single_source.sh [--self-test]
set -euo pipefail
cd "$(dirname "$0")/.."

check() { # $1 = sections.yml  $2 = ci.yml ; prints FAIL lines, returns 1 on any
  local sec="$1" ci="$2" n pins want got bad=0
  pins="$(awk '/^matrix-pins:/{p=1;next} p&&/^  workspace-test-shard:/{w=1;next} w&&/^    - \{shard:/{print;next} w&&!/^    - /{exit}' "$sec")"
  n="$(printf '%s\n' "$pins" | grep -c 'shard:' || true)"
  [ "$n" -ge 1 ] || { echo "FAIL no workspace-test-shard pins in $sec"; return 1; }
  want="$(seq -s, 1 "$n")"
  got="$(printf '%s\n' "$pins" | sed -E 's/^ *- \{shard: ([0-9]+), shards: ([0-9]+)\}.*/\1/' | paste -sd, -)"
  [ "$got" = "$want" ] || { echo "FAIL pins are not shard 1..$n ($got)"; bad=1; }
  [ "$(printf '%s\n' "$pins" | grep -c "shards: $n}")" = "$n" ] || { echo "FAIL a pin's shards != the pin count $n"; bad=1; }
  got="$(awk '/^  workspace-test-shard:/{j=1} j&&/^        shard: \[/{print;exit}' "$ci" | sed -E 's/.*\[(.*)\].*/\1/; s/ //g')"
  [ "$got" = "$want" ] || { echo "FAIL ci.yml matrix shard list ($got) != 1..$n"; bad=1; }
  awk '/^  workspace-test-shard:/{j=1} j&&/^        shards: \[/{print;exit}' "$ci" | grep -q "\[$n\]" || { echo "FAIL ci.yml matrix shards != [$n]"; bad=1; }
  if grep -nE 'hash:\$\{SHARD\}/[0-9]+|quick_tier_shard\.sh "\$SHARD" [0-9]+|--shard "\$\{SHARD\}/[0-9]+"' "$sec"; then echo "FAIL literal shard count in $sec"; bad=1; fi
  lit_loop="n in 1 2 3"
  if grep -nE "\?[0-9]+\?'|${lit_loop}" "$ci"; then echo "FAIL literal shard count in $ci"; bad=1; fi
  [ "$bad" = 0 ] && echo "ok    shard count $n: pins, ci.yml matrix, no literals"
  return "$bad"
}

if [ "${1:-}" = --self-test ]; then
  t="$(mktemp -d)"; trap 'rm -rf "${t:?}"' EXIT
  cp ci/sections.yml "$t/s.yml"; cp .github/workflows/ci.yml "$t/c.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null || { echo "self-test: clean copy must pass"; exit 1; }
  sed 's|hash:${SHARD}/${SHARDS}|hash:${SHARD}/3|' ci/sections.yml > "$t/s.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null && { echo "self-test: literal /3 must be RED"; exit 1; }
  cp ci/sections.yml "$t/s.yml"; sed -E 's/^( +shards: \[)[0-9]+\]/\1999]/' .github/workflows/ci.yml > "$t/c.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null && { echo "self-test: matrix mismatch must be RED"; exit 1; }
  sed 's/for n in \$(seq 1 "\$SHARDS")/for n in 1 2 3/' .github/workflows/ci.yml > "$t/c.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null && { echo "self-test: literal loop must be RED"; exit 1; }
  echo "PASS  check_shard_count_single_source self-test: 4 rows"
  exit 0
fi
check ci/sections.yml .github/workflows/ci.yml
