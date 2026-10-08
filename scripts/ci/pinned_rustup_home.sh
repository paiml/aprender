#!/usr/bin/env bash
# scripts/ci/pinned_rustup_home.sh -- install the rust-toolchain.toml toolchain ONCE per job,
# into a host directory every later container step mounts read-only (0.71 row 23).
#
# Why: the sovereign-ci image is rust:1.95-slim with its rustup home baked in, and
# rust-toolchain.toml pins 1.93.0. Every `docker run` of the image therefore re-downloaded
# 1.93.0 into a throwaway container: up to 7 syncs per x86-main job, and one connect timeout
# on static.rust-lang.org made the GATE red (guard-cargo#m60, run 37718520217, #4870).
#
# seed DIR IMAGE [TOOLCHAIN_FILE]
#   1. copies the image's own rustup home into DIR (so the image's default toolchain and
#      settings are unchanged for anything that runs outside the pinned workspace),
#   2. installs the pinned channel + components into it, at most RETRY_MAX (4) attempts
#      with a 10/20/30 s backoff, and nothing else is retried,
#   3. proves the result: in a directory holding only the toolchain file, with DIR mounted
#      READ-ONLY, rustc/cargo/rustfmt/clippy report the pinned version and rustup prints no
#      "syncing channel updates" -- a later step that still tried to download would fail
#      on the read-only mount instead of reaching the network,
#   4. writes PINNED_RUSTUP_MOUNT=DIR:<image RUSTUP_HOME>:ro to $GITHUB_ENV.
#   Any failure is rc 1 with an ::error line. There is no fallback to the old per-step sync.
#
# --self-test   the case table (retry bound, backoff, pin parsing, no-sync proof); no docker.
set -euo pipefail

RETRY_MAX=${RETRY_MAX:-4}

# retry_bounded MAX CMD... -- run CMD until it succeeds, at most MAX times in total.
# Waits 10*n s after failed attempt n (n < MAX). rc 0 on success, 1 when every attempt
# failed, 2 when MAX is not a positive integer (nothing is run).
retry_bounded() {
  local max=$1 n=0 rc
  shift
  case "$max" in ''|*[!0-9]*|0) echo "::error::retry_bounded: '$max' is not a positive attempt count" >&2; return 2 ;; esac
  while :; do
    n=$((n + 1))
    if "$@"; then echo "attempt $n/$max: ok"; return 0; else rc=$?; fi
    if [ "$n" -ge "$max" ]; then
      echo "::error::$1 failed $n/$max attempts (last rc $rc)" >&2
      return 1
    fi
    echo "::warning::attempt $n/$max of $1 failed (rc $rc); retrying in $((10 * n)) s" >&2
    ${RETRY_SLEEP:-sleep} "$((10 * n))"
  done
}

# read_pin FILE -- prints "CHANNEL COMPONENT..." from a rust-toolchain.toml. The channel must
# be an exact pin (1.93.0 or nightly-YYYY-MM-DD): a floating channel is not a pin.
read_pin() {
  local f=$1 ch comps
  [ -f "$f" ] || { echo "::error::no toolchain file at $f" >&2; return 1; }
  ch=$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$f" | head -n 1)
  if ! [[ "$ch" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ || "$ch" =~ ^nightly-[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "::error::$f: channel '${ch}' is not an exact toolchain pin" >&2
    return 1
  fi
  comps=$(sed -n 's/^[[:space:]]*components[[:space:]]*=[[:space:]]*\[\(.*\)\].*/\1/p' "$f" | head -n 1 | tr -d '" ' | tr ',' ' ')
  printf '%s %s\n' "$ch" "$comps"
}

# assert_pinned_no_sync CHANNEL OUTPUT -- the proof that a step reused the seeded home:
# the pinned version is what answered, and rustup did not go to the network.
assert_pinned_no_sync() {
  local ch=$1 out=$2
  if grep -q 'syncing channel updates' <<< "$out"; then
    echo "::error::the seeded rustup home still synced a channel:" >&2
    grep 'syncing channel updates' <<< "$out" >&2
    return 1
  fi
  grep -q "^rustc ${ch//./\\.} " <<< "$out" || { echo "::error::rustc is not $ch in the seeded home" >&2; return 1; }
  return 0
}

seed() {
  local dir=$1 image=$2 pinfile=${3:-rust-toolchain.toml} pin ch comps rh c out probe
  local -a cargs=()
  pin=$(read_pin "$pinfile") || return 1
  ch=${pin%% *}; comps=${pin#* }
  for c in $comps; do cargs+=(--component "$c"); done
  if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "::error::$dir is not empty; the seed only fills a fresh directory" >&2
    return 1
  fi
  mkdir -p "$dir"
  rh=$(docker run --rm "$image" printenv RUSTUP_HOME) || { echo "::error::cannot read RUSTUP_HOME from $image" >&2; return 1; }
  [ -n "$rh" ] || { echo "::error::$image sets no RUSTUP_HOME" >&2; return 1; }
  echo "seed: $image rustup home $rh -> $dir; pin $ch [${comps}]"
  docker run --rm --user "$(id -u):$(id -g)" -v "$dir:/seed" "$image" \
    sh -c 'cp -a "$RUSTUP_HOME"/. /seed/' || { echo "::error::copying $rh out of $image failed" >&2; return 1; }
  retry_bounded "$RETRY_MAX" docker run --rm --user "$(id -u):$(id -g)" -v "$dir:$rh" "$image" \
    rustup toolchain install "$ch" --profile minimal --no-self-update "${cargs[@]}" || return 1
  probe=$(mktemp -d)
  cp "$pinfile" "$probe/rust-toolchain.toml"
  out=$(docker run --rm --user "$(id -u):$(id -g)" -v "$dir:$rh:ro" -v "$probe:/probe:ro" -w /probe "$image" \
    sh -c 'rustc --version && cargo --version && for c in "$@"; do case $c in rustfmt) cargo fmt --version ;; clippy) cargo clippy --version ;; esac; done' _ $comps 2>&1) \
    || { echo "::error::the read-only seeded home cannot run the pinned toolchain:" >&2; printf '%s\n' "$out" >&2; return 1; }
  rm -rf "${probe:?}"
  printf '%s\n' "$out"
  assert_pinned_no_sync "$ch" "$out" || return 1
  echo "PINNED_RUSTUP_MOUNT=$dir:$rh:ro"
  if [ -n "${GITHUB_ENV:-}" ]; then echo "PINNED_RUSTUP_MOUNT=$dir:$rh:ro" >> "$GITHUB_ENV"; fi
}

self_test() {
  local t fails=0 out rc
  t=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '${t:?}'" RETURN
  flaky() { # flaky K: fails on the first K calls, then succeeds; counts calls in $t/n
    local k=$1 n
    n=$(( $(cat "$t/n" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$t/n"
    [ "$n" -gt "$k" ]
  }
  rec_sleep() { printf '%s ' "$1" >> "$t/slept"; }
  row() { # name want-rc want-calls want-slept max k
    local name=$1 wrc=$2 wcalls=$3 wslept=$4 max=$5 k=$6 calls slept
    : > "$t/slept"; rm -f "$t/n"
    RETRY_SLEEP=rec_sleep retry_bounded "$max" flaky "$k" > /dev/null 2>&1 && rc=0 || rc=$?
    calls=$(cat "$t/n" 2>/dev/null || echo 0); slept=$(cat "$t/slept")
    if [ "$rc" = "$wrc" ] && [ "$calls" = "$wcalls" ] && [ "$slept" = "$wslept" ]; then echo "ok    $name"
    else echo "FAIL  $name: rc=$rc calls=$calls slept='$slept' (want rc=$wrc calls=$wcalls slept='$wslept')"; fails=$((fails + 1)); fi
  }
  row "first attempt ok: one call, no wait"            0 1 ""          4 0
  row "two failures then ok: three calls"              0 3 "10 20 "    4 2
  row "always failing stops at exactly MAX"            1 4 "10 20 30 " 4 99
  row "success after MAX is never reached"             1 3 "10 20 "    3 3
  row "MAX 1 = one attempt, no retry"                  1 1 ""          1 5
  row "MAX 0 runs nothing"                             2 0 ""          0 0
  row "non-numeric MAX runs nothing"                   2 0 ""          x 0

  pin() { # name want-rc want-out file-content
    local name=$1 wrc=$2 wout=$3
    printf '%s\n' "$4" > "$t/tc.toml"
    out=$(read_pin "$t/tc.toml" 2>/dev/null) && rc=0 || rc=$?
    if [ "$rc" = "$wrc" ] && [ "$out" = "$wout" ]; then echo "ok    $name"
    else echo "FAIL  $name: rc=$rc out='$out' (want rc=$wrc out='$wout')"; fails=$((fails + 1)); fi
  }
  pin "repo pin with components"   0 "1.93.0 rustfmt clippy" $'[toolchain]\nchannel = "1.93.0"\ncomponents = ["rustfmt", "clippy"]'
  pin "dated nightly is a pin"     0 "nightly-2026-01-02 "   $'[toolchain]\nchannel = "nightly-2026-01-02"'
  pin "floating stable refused"    1 ""                      $'[toolchain]\nchannel = "stable"'
  pin "no channel refused"         1 ""                      $'[toolchain]\ncomponents = ["clippy"]'
  out=$(read_pin "$t/absent.toml" 2>/dev/null) && rc=0 || rc=$?
  if [ "$rc" = 1 ]; then echo "ok    missing toolchain file refused"; else echo "FAIL  missing toolchain file: rc=$rc"; fails=$((fails + 1)); fi

  nosync() { # name want-rc output
    assert_pinned_no_sync 1.93.0 "$3" 2>/dev/null && rc=0 || rc=$?
    if [ "$rc" = "$2" ]; then echo "ok    $1"; else echo "FAIL  $1: rc=$rc (want $2)"; fails=$((fails + 1)); fi
  }
  nosync "reused home: pinned rustc, no sync"   0 $'rustc 1.93.0 (254b59607 2026-01-19)\ncargo 1.93.0 (x 2026-01-19)'
  nosync "a sync line is RED"                   1 $'info: syncing channel updates for 1.93.0-x86_64-unknown-linux-gnu\nrustc 1.93.0 (254b59607 2026-01-19)'
  nosync "image default answered is RED"        1 'rustc 1.95.0 (abc 2026-05-01)'
  nosync "1.93.01 is not 1.93.0"                1 'rustc 1.93.01 (abc 2026-05-01)'

  [ "$fails" -eq 0 ] && { echo "SELF-TEST PASSED (16 rows)"; return 0; }
  echo "SELF-TEST FAILED ($fails)"; return 1
}

case "${1:-}" in
  --self-test) self_test ;;
  seed) shift; [ $# -ge 2 ] || { echo "usage: $0 seed DIR IMAGE [TOOLCHAIN_FILE]" >&2; exit 2; }; seed "$@" ;;
  *) echo "usage: $0 seed DIR IMAGE [TOOLCHAIN_FILE] | --self-test" >&2; exit 2 ;;
esac
