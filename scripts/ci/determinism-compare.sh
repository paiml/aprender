#!/usr/bin/env bash
# APEX-001 EV-2a: decide cross-architecture render identity from two per-host receipts.
#
# Each `determinism (<arch>)` job writes `determinism-<arch>.json` from
# crates/aprender-viz/tests/render_determinism.rs. This script is the only place that sees both,
# so it is the only place allowed to set `svg_identical`.
#
# The row ASSERTS SVG identity and only RECORDS PNG identity, so the two are separate digests
# and are decided separately. An earlier revision decided `svg_identical` from `manifest_root`,
# a manifest that held only `fixture.png`: the field named SVG, measured PNG, and compared no
# SVG byte at all. The self-test below plants exactly that shape and the pairs that tell the two
# designs apart, and runs a mutant that decides from `manifest_root` to show the table catches it.
#
# usage:
#   determinism-compare.sh <receipts-dir> <out-receipt.json>
#   determinism-compare.sh --self-test
#
# exit 0: both receipts well-formed, from two architectures, SVG identical (PNG recorded either way)
# exit 1: anything else, with the reason on stderr
set -euo pipefail

fail() {
  printf 'determinism-compare: FAIL — %s\n' "$1" >&2
  return 1
}

# The field `svg_identical` is decided from. DETERMINISM_COMPARE_MUTATE=manifest_root exists only
# so --self-test can prove its own case table would catch the old design.
svg_key() {
  if [ "${DETERMINISM_COMPARE_MUTATE:-}" = "manifest_root" ]; then
    printf 'manifest_root'
  else
    printf 'svg_sha256'
  fi
}

compare() {
  local dir=$1 out=$2 n key
  n=$(find "$dir" -maxdepth 1 -name 'determinism-*.json' | wc -l)
  printf 'receipts found: %s\n' "$n"
  # The denominator, stated: a comparison over one file agrees with itself.
  [ "$n" -eq 2 ] || { fail "expected 2 host receipts, found $n"; return 1; }

  # A receipt without an SVG digest is refused, never compared: null == null is "identical".
  jq -se 'all(.[]; (.svg_sha256|type=="string" and test("^[0-9a-f]{64}$"))
                 and (.png_sha256|type=="string" and test("^[0-9a-f]{64}$"))
                 and (.host_arch|type=="string" and length>0))' "$dir"/determinism-*.json >/dev/null \
    || { fail "a receipt lacks a 64-hex svg_sha256, png_sha256 or host_arch"; return 1; }

  # Two receipts from one architecture prove nothing about the platform libm.
  jq -se '[.[].host_arch]|unique|length==2' "$dir"/determinism-*.json >/dev/null \
    || { fail "both receipts come from one architecture"; return 1; }

  key=$(svg_key)
  jq -s --arg key "$key" '
    {
      row: "EV-2a",
      hosts: [.[] | {arch: .host_arch, os: .os, svg_sha256: .svg_sha256, svg_bytes: .svg_bytes,
                     png_sha256: .png_sha256, png_bytes: .png_bytes, manifest_root: .manifest_root}],
      coord_grid: (.[0].coord_grid),
      libm: (.[0].libm),
      svg_identical: ([.[][$key]] | unique | length == 1),
      png_identical: ([.[].png_sha256] | unique | length == 1)
    }' "$dir"/determinism-*.json > "$out"
  cat "$out"

  jq -e '(.hosts | length) == 2' "$out" >/dev/null || { fail "merged receipt does not name two hosts"; return 1; }
  jq -e '.svg_identical == true' "$out" >/dev/null \
    || { fail "the two architectures rendered different SVG bytes (APEX-001 EV-2a rule 5)"; return 1; }

  # Recorded, deliberately NOT asserted: this verdict decides EV-15's manifest shape.
  if jq -e '.png_identical == true' "$out" >/dev/null; then
    printf 'PNG is byte-identical across architectures: EV-15 may use one manifest\n'
  else
    printf 'PNG DIFFERS across architectures: EV-15 must key its PNG manifest by target triple\n'
  fi
}

# receipt <dir> <arch> <svg-hex-char> <png-hex-char> — a planted per-host receipt.
receipt() {
  local svg png root
  svg=$(printf "%064d" 0 | tr 0 "$3")
  png=$(printf "%064d" 0 | tr 0 "$4")
  root=$(printf '%s%s' "$3" "$4" | sha256sum | cut -d' ' -f1)
  jq -n --arg a "$2" --arg s "$svg" --arg p "$png" --arg r "$root" \
    '{host_arch:$a, os:"linux", svg_sha256:$s, svg_bytes:16754, png_sha256:$p, png_bytes:98390,
      manifest_root:$r, coord_grid:0.001, libm:"pure-rust"}' > "$1/determinism-$2.json"
}

# The old writer's shape: manifest over the PNG only, no svg_sha256 field at all.
old_receipt() {
  local png root
  png=$(printf "%064d" 0 | tr 0 b)
  root=$(printf "%064d" 0 | tr 0 c)
  jq -n --arg a "$2" --arg p "$png" --arg r "$root" \
    '{host_arch:$a, os:"linux", png_sha256:$p, png_bytes:98390, manifest_root:$r,
      coord_grid:0.001, libm:"pure-rust"}' > "$1/determinism-$2.json"
}

plant() {
  local d=$1 case=$2
  case "$case" in
    identical)      receipt "$d" x86_64 a b; receipt "$d" aarch64 a b ;;
    png-only-diff)  receipt "$d" x86_64 a b; receipt "$d" aarch64 a e ;;
    svg-only-diff)  receipt "$d" x86_64 a b; receipt "$d" aarch64 d b ;;
    one-host)       receipt "$d" x86_64 a b ;;
    one-arch)       receipt "$d" x86_64 a b; receipt "$d" x86_64-again a b
                    jq '.host_arch="x86_64"' "$d/determinism-x86_64-again.json" > "$d/t" \
                      && mv "$d/t" "$d/determinism-x86_64-again.json" ;;
    old-shape)      old_receipt "$d" x86_64; old_receipt "$d" aarch64 ;;
  esac
}

# case  expected-rc  expected svg_identical  expected png_identical  ("-" = no receipt written)
CASES="identical 0 true true
png-only-diff 0 true false
svg-only-diff 1 false true
one-host 1 - -
one-arch 1 - -
old-shape 1 - -"

run_case() {
  local name=$1 want_rc=$2 want_svg=$3 want_png=$4 d rc got_svg got_png
  d=$(mktemp -d)
  plant "$d" "$name"
  rc=0
  compare "$d" "$d/out.json" >/dev/null 2>&1 || rc=$?
  got_svg=-; got_png=-
  if [ -s "$d/out.json" ]; then
    got_svg=$(jq -r '.svg_identical' "$d/out.json")
    got_png=$(jq -r '.png_identical' "$d/out.json")
  fi
  if [ -n "$d" ] && [ "$d" != "/" ] && [ -d "$d" ]; then
    rm -rf "$d"
  fi
  [ "$rc" = "$want_rc" ] && [ "$got_svg" = "$want_svg" ] && [ "$got_png" = "$want_png" ] && return 0
  printf '  case %s: rc=%s svg=%s png=%s, wanted rc=%s svg=%s png=%s\n' \
    "$name" "$rc" "$got_svg" "$got_png" "$want_rc" "$want_svg" "$want_png" >&2
  return 1
}

run_table() {
  local failed=0 name rc svg png
  while read -r name rc svg png; do
    run_case "$name" "$rc" "$svg" "$png" || failed=$((failed + 1))
  done <<<"$CASES"
  printf '%s' "$failed"
}

self_test() {
  local total failed mutant_failed
  total=$(wc -l <<<"$CASES")
  failed=$(run_table)
  printf 'determinism-compare self-test: cases=%s failed=%s\n' "$total" "$failed"
  [ "$failed" -eq 0 ] || return 1
  # The table must be able to tell the two designs apart, or it is agreeing with itself.
  mutant_failed=$(DETERMINISM_COMPARE_MUTATE=manifest_root run_table 2>/dev/null)
  printf 'mutant (svg_identical from manifest_root): cases failed=%s\n' "$mutant_failed"
  [ "$mutant_failed" -ge 1 ] || { fail "the case table does not catch svg_identical decided from manifest_root"; return 1; }
}

case "${1:-}" in
  --self-test) self_test ;;
  "") printf 'usage: %s <receipts-dir> <out.json> | --self-test\n' "$0" >&2; exit 2 ;;
  *) [ $# -eq 2 ] || { printf 'usage: %s <receipts-dir> <out.json>\n' "$0" >&2; exit 2; }
     compare "$1" "$2" ;;
esac
