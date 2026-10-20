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
  # the inert jobs matrix in sections.yml (the pins override it, but it is still a third copy of the count)
  got="$(awk '/^jobs:/{j=1} j&&/^  workspace-test-shard:/{w=1} w&&/^        shard: \[/{print;exit}' "$sec" | sed -E 's/.*\[(.*)\].*/\1/; s/ //g')"
  [ "$got" = "$want" ] || { echo "FAIL sections.yml jobs matrix shard list ($got) != 1..$n"; bad=1; }
  awk '/^jobs:/{j=1} j&&/^  workspace-test-shard:/{w=1} w&&/^        shards: \[/{print;exit}' "$sec" | grep -q "\[$n\]" || { echo "FAIL sections.yml jobs matrix shards != [$n]"; bad=1; }
  # any digit typed after $SHARD / ${SHARD} (hash:${SHARD}/3, "$SHARD/3", "$SHARD" 3) is a literal count; the [ "$SHARD" = 1 ] designations are not
  if grep -nE '\$\{?SHARD\}?"?[/ ] *"?[0-9]+' "$sec"; then echo "FAIL literal shard count in $sec"; bad=1; fi
  lit_loop="n in 1 2 3"
  if grep -nE "\?[0-9]+\?'|${lit_loop}" "$ci"; then echo "FAIL literal shard count in $ci"; bad=1; fi
  if grep -nE 'SHARDS=\"?[0-9]' "$ci"; then echo "FAIL typed SHARDS= in $ci"; bad=1; fi
  grep -qE 'SHARDS="\$\(awk.*ci/sections\.yml' "$ci" || { echo "FAIL the aggregator SHARDS= no longer reads the pins in ci/sections.yml"; bad=1; }
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
  cp ci/sections.yml "$t/s.yml"; sed 's|bash scripts/ci/quick_tier_shard.sh "$SHARD" "$SHARDS"|bash scripts/ci/quick_tier_shard.sh "$SHARD" 3|' ci/sections.yml > "$t/s.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null && { echo "self-test: typed quick_tier_shard count must be RED"; exit 1; }
  sed 's|--shard "${SHARD}/${SHARDS}"|--shard "$SHARD/3"|' ci/sections.yml > "$t/s.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null && { echo "self-test: unbraced $SHARD/3 must be RED"; exit 1; }
  cp ci/sections.yml "$t/s.yml"; sed -E 's/^( +SHARDS=)"\$\(awk[^)]*\)"/\112/' .github/workflows/ci.yml > "$t/c.yml"
  cmp -s .github/workflows/ci.yml "$t/c.yml" && { echo "self-test: SHARDS=12 mutant did not apply"; exit 1; }
  check "$t/s.yml" "$t/c.yml" >/dev/null && { echo "self-test: typed aggregator SHARDS=12 must be RED"; exit 1; }
  sed -E '0,/^        shards: \[12\]/s//        shards: [11]/' ci/sections.yml > "$t/s.yml"
  cmp -s ci/sections.yml "$t/s.yml" && { echo "self-test: inert-matrix mutant did not apply"; exit 1; }
  check "$t/s.yml" .github/workflows/ci.yml >/dev/null && { echo "self-test: sections.yml jobs matrix drift must be RED"; exit 1; }
  echo "PASS  check_shard_count_single_source self-test: 8 rows"
  exit 0
fi
check ci/sections.yml .github/workflows/ci.yml
