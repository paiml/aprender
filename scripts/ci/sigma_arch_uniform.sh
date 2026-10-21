#!/usr/bin/env bash
# sigma_arch_uniform.sh -- the FULL-tier Σ needs every shard on ONE architecture (#5882).
#
# The fan-in compares the union of every shard's executed ids with ONE universe, the list shard 1
# made on ITS machine. Tests gated on target_arch (crates/aprender-zram-core/src/zstd/avx2.rs has 18
# x86_64-only ones, 141 files carry a cfg) exist in one arch's list and not the other's, so a mixed
# x86 + aarch64 shard set is red or green by PLACEMENT, and a test owed on x86 whose hash slice drew
# an aarch64 shard runs nowhere. A per-arch compare cannot fix that: each arch's shards run only
# their own slices, so each arch would be owed tests it was never given. The sound rule is
# placement: ci.yml job workspace-test-arch decides the tier first and pins the full tier's shards to
# X64. This script is the backstop that keeps the rule true after the fact: at tier full, a shard
# that staged no arch, or two arches in one run, is RED. Any other agreed tier owes no Σ: not checked.
#
#   sigma_arch_uniform.sh DIR SHARDS     DIR/sigma-shard-N/{tier,arch}, N = 1..SHARDS
#   sigma_arch_uniform.sh --self-test    the case table (incl. the mixed placement that went red)
set -euo pipefail

check() { # check DIR SHARDS
  local d=$1 n=$2 i t tier arch first="" tiers="" arches=""
  for i in $(seq 1 "$n"); do
    [ -f "$d/sigma-shard-$i/tier" ] || { echo "::error::Σ arch: shard $i staged no tier file"; return 1; }
    tier=$(cat "$d/sigma-shard-$i/tier"); tiers="$tiers $tier"
  done
  for t in $tiers; do [ "$t" = full ] || { echo "Σ arch: tier '$t' in this run, the lib/explicit Σ is not owed, arch not checked"; return 0; }; done
  for i in $(seq 1 "$n"); do
    [ -s "$d/sigma-shard-$i/arch" ] || { echo "::error::Σ arch: shard $i staged no arch at the full tier (a missing arch is RED, never uniform)"; return 1; }
    arch=$(tr -d '[:space:]' < "$d/sigma-shard-$i/arch")
    arches="$arches $i=$arch"
    first=${first:-$arch}
    if [ "$arch" != "$first" ]; then
      echo "::error::Σ arch: full tier on MIXED architectures (shard $i is $arch, shard 1 is $first):$arches"
      echo "::error::Σ arch: arch-gated tests are owed on one arch and run on another; workspace-test-arch must pin the full tier to X64"
      return 1
    fi
  done
  echo "Σ arch: every shard of the full tier ran on $first"
}

self_test() {
  local td rows=0 bad=0 want got
  td=$(mktemp -d)
  mk() { # mk TIER ARCH... -> one shard dir per arch argument ("-" = no arch file)
    local t=$1 i=0 a; shift; rm -rf "${td:?}/s"; mkdir -p "$td/s"
    for a in "$@"; do
      i=$((i + 1)); mkdir -p "$td/s/sigma-shard-$i"; printf '%s\n' "$t" > "$td/s/sigma-shard-$i/tier"
      [ "$a" = - ] || printf '%s\n' "$a" > "$td/s/sigma-shard-$i/arch"
    done
  }
  row() { # row WANT LABEL TIER ARCH...
    want=$1; local label=$2; shift 2; mk "$@"
    set +e; check "$td/s" "$(($# - 1))" > /dev/null 2>&1; got=$?; set -e
    rows=$((rows + 1)); [ "$got" = "$want" ] || bad=$((bad + 1))
    printf '%s want_rc=%s got_rc=%s  %s\n' "$([ "$got" = "$want" ] && echo 'ok ' || echo 'BAD')" "$want" "$got" "$label"
  }
  row 0 "full, 12 shards all x86_64" full x86_64 x86_64 x86_64 x86_64 x86_64 x86_64 x86_64 x86_64 x86_64 x86_64 x86_64 x86_64
  row 0 "full, 12 shards all aarch64" full aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64
  row 1 "full, MIXED: shard 3 x86_64, the rest aarch64 (run 39620701544's placement)" full aarch64 aarch64 x86_64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64 aarch64
  row 1 "full, MIXED: shard 1 x86_64, the rest aarch64" full x86_64 aarch64 aarch64 aarch64
  row 1 "full, one shard staged no arch" full x86_64 - x86_64
  row 0 "quick, MIXED: no Σ owed, not checked (the PR tier stays arch-neutral)" quick aarch64 x86_64 aarch64
  mk full x86_64; rm -f "${td:?}/s/sigma-shard-1/tier"
  set +e; check "$td/s" 1 > /dev/null 2>&1; got=$?; set -e
  rows=$((rows + 1)); [ "$got" = 1 ] || bad=$((bad + 1))
  printf '%s want_rc=1 got_rc=%s  a missing tier file refuses\n' "$([ "$got" = 1 ] && echo 'ok ' || echo 'BAD')" "$got"
  # Why a per-arch compare is not the fix, on the real comparator: shard 1's list is aarch64 (no
  # avx2 test), the x86 shard executed one -> "executed but not in the universe".
  local j="$td/j.xml" l="$td/l.json"
  printf '<testsuites><testsuite><testcase classname="z" name="a::neon"/><testcase classname="z" name="avx2::t"/></testsuite></testsuites>' > "$j"
  printf '{"rust-suites":{"z":{"testcases":{"a::neon":{"filter-match":{"status":"matches"}}}}}}' > "$l"
  set +e; python3 "$(dirname "$0")/sigma_executed.py" nextest --junit "$j" --list-json "$l" > /dev/null 2>&1; got=$?; set -e
  rows=$((rows + 1)); [ "$got" = 1 ] || bad=$((bad + 1))
  printf '%s want_rc=1 got_rc=%s  sigma_executed over a mixed-arch pair (x86 test, aarch64 universe) is RED\n' "$([ "$got" = 1 ] && echo 'ok ' || echo 'BAD')" "$got"
  rm -rf "${td:?}"
  echo "Σ arch self-test: $((rows - bad))/$rows rows as expected"
  [ "$bad" = 0 ]
}

case "${1:-}" in
  --self-test) self_test ;;
  "" | -h | --help) sed -n 2,16p "$0" ;;
  *) check "$1" "${2:?SHARDS}" ;;
esac
