#!/usr/bin/env bash
# APEX-001 EV-2d: record cross-architecture PNG raster identity from two per-host receipts.
#
# Each `raster (<arch>)` job writes `raster-<arch>.json` from
# crates/aprender-viz/tests/raster_determinism.rs. This script is the only place that sees both,
# so it is the only place allowed to set `png_identical`.
#
# This is a SIBLING of determinism-compare.sh, not a mode of it: that script ASSERTS SVG
# identity; this one asserts NOTHING about the PNG. A PNG difference across architectures is
# exit 0 with png_identical=false — recorded for EV-15 to key its manifest on, never a failure
# here. The receipts directory also holds determinism-*.json files from the sibling script; this
# script must look only at raster-*.json and ignore the rest, and the self-test below proves it.
#
# usage:
#   raster-compare.sh <receipts-dir> <out-receipt.json>
#   raster-compare.sh --self-test
#
# exit 0: both receipts well-formed, from two architectures, same SVG input, same geometry
#         (PNG identity is recorded in the output, never asserted)
# exit 1: anything else, with the reason on stderr — and nothing is written to <out-receipt.json>
set -euo pipefail

fail() {
  printf 'raster-compare: FAIL — %s\n' "$1" >&2
  return 1
}

# compare <receipts-dir> <out.json>
#
# RASTER_COMPARE_MUTATE exists only so --self-test can prove its own case table would catch two
# wrong designs:
#   assert_png     — treats PNG identity as a hard requirement (the row explicitly does not).
#   skip_svg_check — skips the "same SVG input" check, so two hosts rasterising different SVG
#                    bytes would be compared as if they were the same input.
compare() {
  local dir=$1 out=$2 n tmp
  n=$(find "$dir" -maxdepth 1 -name 'raster-*.json' | wc -l)
  printf 'receipts found: %s\n' "$n"
  # The denominator, stated: a comparison over one file agrees with itself.
  [ "$n" -eq 2 ] || { fail "expected 2 host receipts, found $n"; return 1; }

  jq -se 'all(.[]; (.svg_sha256|type=="string" and test("^[0-9a-f]{64}$"))
                 and (.png_sha256|type=="string" and test("^[0-9a-f]{64}$"))
                 and (.host_arch|type=="string" and length>0)
                 and (.png_bytes|type=="number")
                 and (.width|type=="number")
                 and (.height|type=="number")
                 and (.scale|type=="number"))' "$dir"/raster-*.json >/dev/null \
    || { fail "a receipt lacks a 64-hex svg_sha256/png_sha256, a non-empty host_arch, or numeric png_bytes/width/height/scale"; return 1; }

  # Two receipts from one architecture prove nothing about the platform rasteriser.
  jq -se '[.[].host_arch]|unique|length==2' "$dir"/raster-*.json >/dev/null \
    || { fail "both receipts come from one architecture"; return 1; }

  # Both hosts must have rasterised the SAME SVG, or the PNG comparison measures nothing.
  if [ "${RASTER_COMPARE_MUTATE:-}" != "skip_svg_check" ]; then
    jq -se '[.[].svg_sha256]|unique|length==1' "$dir"/raster-*.json >/dev/null \
      || { fail "the two hosts rasterised different SVG input (svg_sha256 differs)"; return 1; }
  fi

  jq -se '[.[].width]|unique|length==1' "$dir"/raster-*.json >/dev/null \
    || { fail "width differs across hosts"; return 1; }
  jq -se '[.[].height]|unique|length==1' "$dir"/raster-*.json >/dev/null \
    || { fail "height differs across hosts"; return 1; }
  jq -se '[.[].scale]|unique|length==1' "$dir"/raster-*.json >/dev/null \
    || { fail "scale differs across hosts"; return 1; }

  # Every assertion above this line has passed. Write to a temp file first, so a mid-write
  # failure (including the diagnostic mutant check below) leaves the caller's <out> untouched.
  tmp=$(mktemp "$dir/.raster-compare.XXXXXX")
  jq -s '{
    row: "EV-2d",
    hosts: [.[] | {arch: .host_arch, os: .os, svg_sha256: .svg_sha256, png_sha256: .png_sha256,
                   png_bytes: .png_bytes, width: .width, height: .height, scale: .scale,
                   resvg: .resvg}],
    png_identical: ([.[].png_sha256] | unique | length == 1),
    svg_sha256: .[0].svg_sha256
  }' "$dir"/raster-*.json > "$tmp"

  # Diagnostic only: the real rule never asserts this. Exists so --self-test can show the case
  # table catches a design that wrongly treats PNG identity as required.
  if [ "${RASTER_COMPARE_MUTATE:-}" = "assert_png" ] \
     && ! jq -e '.png_identical == true' "$tmp" >/dev/null; then
    rm -f "$tmp"
    fail "PNG bytes differ across architectures (assert_png mutant — the row never requires this)"
    return 1
  fi

  mv "$tmp" "$out"
  cat "$out"
  if jq -e '.png_identical == true' "$out" >/dev/null; then
    printf 'png_identical: true\n'
  else
    printf 'png_identical: false\n'
  fi
}

# receipt <dir> <arch> <svg-hex-char> <png-hex-char> [width] [height] [scale] — a planted receipt.
receipt() {
  local d=$1 arch=$2 svgc=$3 pngc=$4 w=${5:-640} h=${6:-400} sc=${7:-2.0}
  local svg png
  svg=$(printf '%064d' 0 | tr 0 "$svgc")
  png=$(printf '%064d' 0 | tr 0 "$pngc")
  jq -n --arg a "$arch" --arg s "$svg" --arg p "$png" --argjson w "$w" --argjson h "$h" \
        --argjson sc "$sc" \
    '{row:"EV-2d", host_arch:$a, os:"linux", svg_sha256:$s, png_sha256:$p, png_bytes:98765,
      width:$w, height:$h, scale:$sc, resvg:"0.45.1"}' > "$d/raster-$arch.json"
}

# receipt_no_png <dir> <arch> <svg-hex-char> — a receipt missing png_sha256 entirely.
receipt_no_png() {
  local d=$1 arch=$2 svgc=$3
  local svg
  svg=$(printf '%064d' 0 | tr 0 "$svgc")
  jq -n --arg a "$arch" --arg s "$svg" \
    '{row:"EV-2d", host_arch:$a, os:"linux", svg_sha256:$s, png_bytes:98765,
      width:640, height:400, scale:2.0, resvg:"0.45.1"}' > "$d/raster-$arch.json"
}

# plant_determinism_noise <dir> — two well-formed determinism-*.json siblings that must be
# ignored by name (compare() only globs raster-*.json).
plant_determinism_noise() {
  local d=$1 svg png
  svg=$(printf '%064d' 0 | tr 0 f)
  png=$(printf '%064d' 0 | tr 0 f)
  jq -n --arg s "$svg" --arg p "$png" '{host_arch:"x86_64", os:"linux", svg_sha256:$s, png_sha256:$p}' \
    > "$d/determinism-x86_64.json"
  jq -n --arg s "$svg" --arg p "$png" '{host_arch:"aarch64", os:"linux", svg_sha256:$s, png_sha256:$p}' \
    > "$d/determinism-aarch64.json"
}

plant() {
  local d=$1 case=$2
  case "$case" in
    identical)        receipt "$d" x86_64 a b; receipt "$d" aarch64 a b ;;
    png-differs)       receipt "$d" x86_64 a b; receipt "$d" aarch64 a c ;;
    one-host)          receipt "$d" x86_64 a b ;;
    three-hosts)       receipt "$d" x86_64 a b; receipt "$d" aarch64 a b; receipt "$d" armv7 a b ;;
    same-arch-twice)   receipt "$d" x86_64 a b
                       receipt "$d" x86_64-again a b
                       jq '.host_arch="x86_64"' "$d/raster-x86_64-again.json" > "$d/t" \
                         && mv "$d/t" "$d/raster-x86_64-again.json" ;;
    png-missing)       receipt_no_png "$d" x86_64 a; receipt "$d" aarch64 a b ;;
    svg-differs)       receipt "$d" x86_64 a b; receipt "$d" aarch64 d b ;;
    width-differs)     receipt "$d" x86_64 a b 640 400 2.0; receipt "$d" aarch64 a b 800 400 2.0 ;;
    noise-ignored)     receipt "$d" x86_64 a b; receipt "$d" aarch64 a b; plant_determinism_noise "$d" ;;
  esac
}

# All self-test scratch directories are registered here and swept by an EXIT trap, so a case
# that dies before its own rm -rf still never leaks into /tmp, and never touches a caller's
# receipts dir (each case gets its own fresh mktemp -d).
SELF_TEST_TMP_DIRS=()

cleanup_self_test_tmp_dirs() {
  local status=$? d
  for d in "${SELF_TEST_TMP_DIRS[@]:-}"; do
    if [ -n "$d" ] && [ -d "$d" ]; then
      rm -rf "$d"
    fi
  done
  return "$status"
}

# run_case <name> <want-rc> <want-png> [mutate-env] — plant a case, run compare(), grade it.
run_case() {
  local name=$1 want_rc=$2 want_png=$3 mutate=${4:-} d rc got_png out_exists status
  d=$(mktemp -d)
  SELF_TEST_TMP_DIRS+=("$d")
  plant "$d" "$name"
  rc=0
  if [ -n "$mutate" ]; then
    RASTER_COMPARE_MUTATE=$mutate compare "$d" "$d/out.json" >/dev/null 2>&1 || rc=$?
  else
    compare "$d" "$d/out.json" >/dev/null 2>&1 || rc=$?
  fi
  got_png=-
  [ -s "$d/out.json" ] && got_png=$(jq -r '.png_identical' "$d/out.json")
  out_exists=no
  [ -e "$d/out.json" ] && out_exists=yes
  rm -rf "$d"
  status=FAIL
  [ "$rc" = "$want_rc" ] && [ "$got_png" = "$want_png" ] && status=ok
  printf '  case %-24s %-4s rc=%s(want %s) png=%s(want %s) out=%s\n' \
    "$name${mutate:+ [$mutate]}" "$status" "$rc" "$want_rc" "$got_png" "$want_png" "$out_exists" >&2
  [ "$status" = ok ]
}

# run_case_no_output <name> — a failing case must leave <out> entirely absent.
run_case_no_output() {
  local name=$1 d rc out_exists status
  d=$(mktemp -d)
  SELF_TEST_TMP_DIRS+=("$d")
  plant "$d" "$name"
  rc=0
  compare "$d" "$d/out.json" >/dev/null 2>&1 || rc=$?
  out_exists=no
  [ -e "$d/out.json" ] && out_exists=yes
  rm -rf "$d"
  status=FAIL
  [ "$rc" != 0 ] && [ "$out_exists" = no ] && status=ok
  printf '  case %-24s %-4s rc=%s out=%s (want rc!=0 out=no)\n' "$name-no-output" "$status" "$rc" "$out_exists" >&2
  [ "$status" = ok ]
}

self_test() {
  local failed=0
  trap cleanup_self_test_tmp_dirs EXIT

  run_case identical      0 true  || failed=$((failed + 1))
  run_case png-differs    0 false || failed=$((failed + 1))
  run_case one-host       1 -     || failed=$((failed + 1))
  run_case three-hosts    1 -     || failed=$((failed + 1))
  run_case same-arch-twice 1 -    || failed=$((failed + 1))
  run_case png-missing    1 -     || failed=$((failed + 1))
  run_case svg-differs    1 -     || failed=$((failed + 1))
  run_case width-differs  1 -     || failed=$((failed + 1))
  run_case noise-ignored  0 true  || failed=$((failed + 1))

  # Case 10: a failing case leaves NO output file — assert absence, not just a non-zero exit.
  run_case_no_output one-host || failed=$((failed + 1))

  # Mutant 1: asserting PNG identity is the wrong design (rule 3 of the header). It must turn
  # case png-differs — normally exit 0 with png_identical=false — into a hard failure that also
  # writes nothing.
  run_case png-differs 1 - assert_png || failed=$((failed + 1))

  # Mutant 2: skipping the SVG-input-equality check must silently accept case svg-differs —
  # normally exit 1 — as if it were valid, proving the table would catch that regression.
  run_case svg-differs 0 true skip_svg_check || failed=$((failed + 1))

  printf 'raster-compare self-test: %s cases, %s mutants, failed: %s\n' 10 2 "$failed"
  [ "$failed" -eq 0 ]
}

case "${1:-}" in
  --self-test) self_test ;;
  "") printf 'usage: %s <receipts-dir> <out.json> | --self-test\n' "$0" >&2; exit 2 ;;
  *) [ $# -eq 2 ] || { printf 'usage: %s <receipts-dir> <out.json>\n' "$0" >&2; exit 2; }
     compare "$1" "$2" ;;
esac
