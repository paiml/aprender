#!/usr/bin/env bash
# shard_union_nextest.sh -- `--partition hash:i/M` over a probe package: the union over 1..M must equal
# the unpartitioned nextest list (same count, none dropped, none twice). Split out of
# check_shard_union.sh so that guard stays cargo-free and guard-tree dispatches it.
# Usage: shard_union_nextest.sh [pkg]   (M is read from ci/sections.yml, never typed here)
set -euo pipefail
cd "$(dirname "$0")/.."
cd "$(dirname "$0")/.."
M="$(grep -c '^    - {shard:' <(awk '/^matrix-pins:/{p=1;next} p&&/^  workspace-test-shard:/{w=1;next} w&&/^    - \{shard:/{print;next} w&&!/^    - /{exit}' ci/sections.yml))"
[ "$M" -ge 1 ] || { echo "FAIL cannot read the shard count from ci/sections.yml"; exit 1; }

nextest() { # $1 = probe package
  local pkg="$1" full un i
  full="$(cargo nextest list -p "$pkg" --lib --message-format oneline 2>/dev/null | sort)"
  [ -n "$full" ] || { echo "FAIL nextest: empty pre-split list for $pkg"; return 1; }
  un="$(for i in $(seq 1 "$M"); do cargo nextest list -p "$pkg" --lib --message-format oneline --partition "hash:$i/$M" 2>/dev/null; done | sort)"
  [ "$(printf '%s\n' "$full" | wc -l)" = "$(printf '%s\n' "$un" | wc -l)" ] || { echo "FAIL nextest: count pre-split=$(printf '%s\n' "$full" | wc -l) union=$(printf '%s\n' "$un" | wc -l)"; return 1; }
  [ "$full" = "$un" ] || { echo "FAIL nextest: union != pre-split list"; return 1; }
  echo "ok    nextest: union of $M hash shards == pre-split list for $pkg ($(printf '%s\n' "$full" | wc -l) tests)"
}

nextest "${1:-aprender-contracts}"
