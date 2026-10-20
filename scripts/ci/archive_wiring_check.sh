#!/usr/bin/env bash
# FLOW-010 P1: the archive job's wiring must keep its properties. Comments are stripped
# first and every property is scoped to the step that owns it, so a commented-out line or a
# property moved to another step does not satisfy it.
# usage: archive_wiring_check.sh [SECTIONS.yml CI.yml]   |   --self-test
set -euo pipefail

# code(FILE): the file without whole-line comments
code() { sed -E 's/^[[:space:]]*#.*$//' "$1"; }

# step_with(FILE, REGEX...): the steps (blocks starting at a `      - ` line) that match EVERY regex
step_with() {
  local f="$1"; shift
  code "$f" | awk -v n="$#" -v pats="$(printf '%s\x1f' "$@")" '
    BEGIN { split(pats, P, "\x1f") }
    function flush(   i, ok) { if (blk == "") return; ok = 1; for (i = 1; i <= n; i++) if (blk !~ P[i]) ok = 0; if (ok) printf "%s\x1e", blk; blk = "" }
    /^      - / { flush() }
    { blk = blk $0 "\n" }
    END { flush() }'
}

check() {
  local sec="$1" ci="$2" bad=0 s
  # 1. the smoke step refuses to pass having checked zero binaries
  s="$(step_with "$sec" 'ws-smoke-target')"
  [ -n "$s" ] && printf '%s' "$s" | grep -qF '[ "$n" -gt 0 ]' || { echo "wiring: smoke step does not require n>0" >&2; bad=1; }
  # 2. the toolchain text is captured by a plain command substitution (set -e sees its failure),
  #    never piped, never given a fallback, and validated by the CLI on both sides
  if step_with "$sec" 'rustc_vv=' | tr '\036' '\n' | grep 'rustc -vV' | grep -vE '^[[:space:]]*(rustc_vv|rustc_text)="\$\(docker run --rm "\$IMAGE" rustc -vV\)"[[:space:]]*$' | grep -q .; then
    echo "wiring: a rustc -vV use is not the bare captured form (pipe or || fallback?)" >&2; bad=1; fi
  s="$(step_with "$sec" 'rustc_vv=' 'jq -n')"
  printf '%s' "$s" | grep -qF -- '--hash-toolchain "$rustc_vv"' || { echo "wiring: archive step does not validate the toolchain text" >&2; bad=1; }
  s="$(step_with "$sec" 'rustc_vv=' 'archive_bind_check.sh')"
  printf '%s' "$s" | grep -qF -- '--toolchain "$rustc_vv"' || { echo "wiring: shard step does not bind the validated toolchain" >&2; bad=1; }
  # 3. each ws-archive UPLOAD step overwrites
  for f in "$sec" "$ci"; do
    s="$(step_with "$f" 'upload-artifact' 'name: ws-archive')"
    [ -n "$s" ] && [ "$(printf '%s' "$s" | tr '\036' '\n' | grep -c 'name: ws-archive$')" -ge 1 ] || { echo "wiring: no ws-archive upload step in $f" >&2; bad=1; continue; }
    [ "$(step_with "$f" 'upload-artifact' 'name: ws-archive' 'overwrite: true' | tr '\036' '\n' | grep -c 'name: ws-archive$')" = \
      "$(printf '%s' "$s" | tr '\036' '\n' | grep -c 'name: ws-archive$')" ] || { echo "wiring: a ws-archive upload in $f lacks overwrite: true" >&2; bad=1; }
  done
  [ "$bad" = 0 ] && echo "archive_wiring_check: ok"
}

self_test() {
  local d; d="$(mktemp -d)" sec=ci/sections.yml ci=.github/workflows/ci.yml
  check "$sec" "$ci" >/dev/null || { echo "FAIL: real tree rejected" >&2; check "$sec" "$ci" >&2; return 1; }
  kill_sec() { # name sed-expr : mutate sections.yml, expect rejection
    sed -E "$2" "$sec" > "$d/$1"; cmp -s "$sec" "$d/$1" && { echo "FAIL: mutant $1 did not apply" >&2; return 1; }
    if check "$d/$1" "$ci" >/dev/null 2>&1; then echo "FAIL: mutant $1 survived" >&2; return 1; fi; }
  kill_ci() { sed -E "$2" "$ci" > "$d/$1"; cmp -s "$ci" "$d/$1" && { echo "FAIL: mutant $1 did not apply" >&2; return 1; }
    if check "$sec" "$d/$1" >/dev/null 2>&1; then echo "FAIL: mutant $1 survived" >&2; return 1; fi; }
  kill_sec n0       's/\[ "\$n" -gt 0 \]/true/'
  kill_sec n0c      's/^( *)\[ "\$n" -gt 0 \]/\1# [ "$n" -gt 0 ]/'
  kill_sec hash     's/--hash-toolchain "\$rustc_vv"/--hash-toolchain ""/'
  kill_sec shardtc  's/--toolchain "\$rustc_vv"/--other "$rustc_vv"/'
  kill_sec fallback 's/(rustc_vv="\$\(docker run --rm "\$IMAGE" rustc -vV)\)"/\1 || echo x)"/'
  kill_sec pipe     's/(rustc_vv="\$\(docker run --rm "\$IMAGE" rustc -vV)\)"/\1 | head -1)"/'
  kill_sec ovw      's/overwrite: true/overwrite: false/'
  kill_sec ovwmove  's/^( *)overwrite: true/\1if-no-files-found: error/'
  kill_ci  ciovw    's/overwrite: true/overwrite: false/'
  rm -rf "${d:?}"
  echo "archive_wiring_check self-test: ok (9 mutants killed)"
}

if [ "${1:-}" = "--self-test" ]; then self_test; else check "${1:-ci/sections.yml}" "${2:-.github/workflows/ci.yml}"; fi
