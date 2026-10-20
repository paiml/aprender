#!/usr/bin/env bash
# check_shard_count_single_source.sh -- the workspace-test shard count has ONE source.
# The pins in ci/sections.yml `matrix-pins.workspace-test-shard` own it; ci.yml's matrix
# (`shard:` list and `shards:`) must agree, and no `/3`-style literal may survive in the
# sharded commands (a stale literal silently drops or duplicates tests).
# Usage: check_shard_count_single_source.sh [--self-test]
set -euo pipefail
cd "$(dirname "$0")/.."

check() { # $1 = sections.yml  $2 = ci.yml
  python3 - "$1" "$2" <<'PY'
import re, sys, yaml
sec = yaml.safe_load(open(sys.argv[1]))
pins = sec["matrix-pins"]["workspace-test-shard"]
n = len(pins)
bad = []
if [p["shard"] for p in pins] != list(range(1, n + 1)): bad.append("pins are not shard 1..N")
if {p["shards"] for p in pins} != {n}: bad.append("a pin's shards != the pin count %d" % n)
ci = yaml.safe_load(open(sys.argv[2]))["jobs"]["workspace-test-shard"]["strategy"]["matrix"]
if ci.get("shard") != list(range(1, n + 1)): bad.append("ci.yml matrix shard list != 1..%d" % n)
if ci.get("shards") != [n]: bad.append("ci.yml matrix shards != [%d]" % n)
body = open(sys.argv[1]).read()
for m in re.finditer(r'(hash:\$\{SHARD\}/\d+|quick_tier_shard\.sh "\$SHARD" \d+)', body):
    bad.append("literal shard count in sections.yml: " + m.group(1))
if "?3?" in open(sys.argv[2]).read(): bad.append("literal ?3? in ci.yml --sections")
print("\n".join("FAIL " + b for b in bad) if bad else "ok    shard count %d: pins, ci.yml matrix, no literals" % n)
sys.exit(1 if bad else 0)
PY
}

if [ "${1:-}" = --self-test ]; then
  t="$(mktemp -d)"; trap 'rm -rf "${t:?}"' EXIT
  cp ci/sections.yml "$t/s.yml"; cp .github/workflows/ci.yml "$t/c.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null || { echo "self-test: clean copy must pass"; exit 1; }
  sed 's/hash:${SHARD}\/${SHARDS}/hash:${SHARD}\/3/' ci/sections.yml > "$t/s.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null && { echo "self-test: literal /3 must be RED"; exit 1; }
  cp ci/sections.yml "$t/s.yml"; sed 's/shards: \[12\]/shards: [3]/' .github/workflows/ci.yml > "$t/c.yml"
  check "$t/s.yml" "$t/c.yml" >/dev/null && { echo "self-test: matrix mismatch must be RED"; exit 1; }
  echo "PASS  check_shard_count_single_source self-test: 3 rows"
  exit 0
fi
check ci/sections.yml .github/workflows/ci.yml
