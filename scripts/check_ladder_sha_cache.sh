#!/usr/bin/env bash
# check_ladder_sha_cache.sh — the ladder's sha256 cache (#4131) saves the hash only for a file that
# is provably the SAME file: a changed file is always re-hashed, and the cache is never trusted blindly.
#
# WHY. model_ladder.sh sha256'd every held model on every run: 26 files, 109.1 GB on lambda, at a
# measured 0.23 GB/s, ≈473 s, ~60% of a --only 9B rung run. ladder_sha256 now caches by the file's
# IDENTITY (st_dev, st_ino, size, mtime_ns, ctime_ns via `stat -L`). The danger of any cache is a
# stale attestation, where the receipt certifies a file that is no longer the one on disk. Each case
# below is a way a file can change under the same PATH, and each must re-hash.
#
# It lifts the SHIPPED ladder_sha256 and runs it against real files:
#   miss-then-hit      first call hashes (source `hashed`), a fresh process reads it back (`cache`),
#                      and a second call in the same process is `memo`; all three hashes are equal
#   rewrite-same-size  new content, same size, normal write -> re-hashed, the NEW sha
#   mtime-restored     new content, same size, mtime put back with `touch -d` -> re-hashed (ctime
#                      moved); this is the cop's must-RED
#   inode-swap         a different file with the same size and mtime moved over the path -> re-hashed
#   symlink            a symlink is keyed and hashed through its target; retargeting it re-hashes
#   cache-consulted    a planted entry for the CURRENT identity is returned (the cache is really read,
#                      so the cases above prove re-hashing, not merely a cache that is never used)
# --self-test plants a PATH-keyed cache (rewrite/mtime/inode RED) and a key without ctime
# (mtime-restored RED).
#
# Exit: 0 all as expected · 1 a case landed wrong · 2 could not check.
set -uo pipefail
SCRIPT="scripts/model_ladder.sh"; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ladder_sha_cache: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -f "$SCRIPT" ] || { echo "  cannot check: $SCRIPT not found" >&2; exit 2; }
T=$(mktemp -d -t check_ladder_sha_cache.XXXXXXXX) || { echo "  cannot check: mktemp failed" >&2; exit 2; }
cleanup() { case "${T:-}" in /tmp/?*) if [ -n "$T" ] && [ -d "$T" ]; then rm -rf -- "$T" || :; fi ;; esac; }
trap cleanup EXIT

lift() { awk '/^declare -A LADDER_SHA_MEMO=/{print} /^ladder_sha256\(\) \{/{f=1} f{print} f && /^\}$/{exit}' "$1"; }

run_cases() { # <script> -> 0 when every case lands
  local src="$1" rc=0 body d out a b c
  ok() { printf '  ok    %s\n' "$1"; }
  bad() { printf '  FAIL  %s: %s\n' "$1" "$2"; rc=1; }
  body=$(lift "$src")
  grep -q '^ladder_sha256() {' <<< "$body" || { bad lift "$src defines no ladder_sha256()"; return 1; }
  d=$(mktemp -d "$T/case.XXXX")
  sh1() { LADDER_SHA_CACHE="$d/cache.tsv" bash -c "$body"$'\n''ladder_sha256 "$1"' _ "$1"; }
  real() { sha256sum "$1" | cut -d' ' -f1; }

  printf 'aaaaaaaaaaaaaaaa' > "$d/m.gguf"
  a=$(sh1 "$d/m.gguf"); b=$(sh1 "$d/m.gguf")
  c=$(LADDER_SHA_CACHE="$d/cache.tsv" bash -c "$body"$'\n''ladder_sha256 "$1" > /dev/null; ladder_sha256 "$1"' _ "$d/m.gguf")
  if [ "$a" = "$(real "$d/m.gguf") hashed" ] && [ "$b" = "$(real "$d/m.gguf") cache" ] && [ "$c" = "$(real "$d/m.gguf") memo" ]; then ok miss-then-hit
  else bad miss-then-hit "got '$a' / '$b' / '$c'"; fi

  sleep 0.05; printf 'bbbbbbbbbbbbbbbb' > "$d/m.gguf"
  out=$(sh1 "$d/m.gguf")
  [ "$out" = "$(real "$d/m.gguf") hashed" ] && ok rewrite-same-size || bad rewrite-same-size "got '$out', want the new sha, hashed"

  local mt; mt=$(stat -c '%.9Y' "$d/m.gguf"); sh1 "$d/m.gguf" > /dev/null
  sleep 0.05; printf 'cccccccccccccccc' > "$d/m.gguf"; touch -d "@$mt" "$d/m.gguf"
  out=$(sh1 "$d/m.gguf")
  if [ "$(stat -c '%.9Y' "$d/m.gguf")" = "$mt" ] && [ "$out" = "$(real "$d/m.gguf") hashed" ]; then ok mtime-restored
  else bad mtime-restored "got '$out' (mtime restored: $([ "$(stat -c '%.9Y' "$d/m.gguf")" = "$mt" ] && echo yes || echo no)), want the new sha, hashed"; fi

  sh1 "$d/m.gguf" > /dev/null; mt=$(stat -c '%.9Y' "$d/m.gguf")
  printf 'dddddddddddddddd' > "$d/other"; touch -d "@$mt" "$d/other"; mv -f "$d/other" "$d/m.gguf"
  out=$(sh1 "$d/m.gguf")
  [ "$out" = "$(real "$d/m.gguf") hashed" ] && ok inode-swap || bad inode-swap "got '$out', want the new sha, hashed"

  printf 'eeeeeeeeeeeeeeee' > "$d/t1"; printf 'ffffffffffffffff' > "$d/t2"; ln -sfn "$d/t1" "$d/link.gguf"
  a=$(sh1 "$d/link.gguf"); ln -sfn "$d/t2" "$d/link.gguf"; b=$(sh1 "$d/link.gguf")
  if [ "$a" = "$(real "$d/t1") hashed" ] && [ "$b" = "$(real "$d/t2") hashed" ]; then ok symlink
  else bad symlink "got '$a' then '$b'"; fi

  printf 'gggggggggggggggg' > "$d/p.gguf"
  printf '%s\t%s\n' "$(stat -Lc '%d %i %s %.9Y %.9Z' "$d/p.gguf")" "$(printf '0%.0s' $(seq 64))" >> "$d/cache.tsv"
  out=$(sh1 "$d/p.gguf")
  [ "$out" = "$(printf '0%.0s' $(seq 64)) cache" ] && ok cache-consulted || bad cache-consulted "got '$out', want the planted entry, cache"
  return "$rc"
}

if [ "$SELF_TEST" = 1 ]; then
  bad=0
  mutant() { # <label> <sed> <case...>
    local label="$1" expr="$2" m="$T/m-$1.sh" o c; shift 2
    sed "$expr" "$SCRIPT" > "$m"
    cmp -s "$SCRIPT" "$m" && { echo "  FAIL  mutant $label did not apply"; bad=1; return; }
    o=$(run_cases "$m" 2>&1) || true
    for c in "$@"; do
      grep -q "FAIL  $c:" <<< "$o" && printf '  ok    mutant %-12s killed by %s\n' "$label" "$c" || { echo "  FAIL  mutant $label SURVIVED case $c"; bad=1; }
    done
  }
  mutant path-keyed "s/stat -Lc '%d %i %s %.9Y %.9Z' \"\$path\"/printf '%s' \"\$path\"/g" rewrite-same-size mtime-restored inode-swap
  mutant no-ctime "s/'%d %i %s %.9Y %.9Z'/'%d %i %s %.9Y'/g" mtime-restored
  [ "$bad" = 0 ] && { echo "SELF-TEST OK"; exit 0; }
  echo "SELF-TEST FAIL"; exit 1
fi
echo "ladder sha256 cache: identity-keyed, a changed file is always re-hashed ($SCRIPT)"
if run_cases "$SCRIPT"; then echo "PASS"; exit 0; fi
echo "FAIL"; exit 1
