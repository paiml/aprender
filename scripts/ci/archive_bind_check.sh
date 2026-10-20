#!/usr/bin/env bash
# FLOW-010 P1: a nextest archive is only used by the run, commit, filterset, build flags
# and image it was built for. Fails (rc 1) on any mismatch, a missing bind.json, a tier
# that is not quick, or a missing key -- the quick tier has no second path.
# usage: archive_bind_check.sh [--toolchain "$(rustc -vV)"] BIND.json sha=… expr=… flags=…
#        archive_bind_check.sh --hash-toolchain "$(rustc -vV)"   (prints the 16-hex bind value)
#        archive_bind_check.sh --self-test
set -euo pipefail

# A failed `docker run … rustc -vV` prints nothing, and sha256 of nothing is a perfectly
# good hash that two failed runs share: empty must be refused BEFORE hashing.
hash_toolchain() {
  case "${1:-}" in *"rustc "*) ;; *) echo "archive_bind_check: no 'rustc ' line in the toolchain text -- refusing to hash nothing" >&2; return 1;; esac
  printf '%s' "$1" | sha256sum | cut -c1-16
}

check() {
  if [ "${1:-}" = --toolchain ]; then
    local h; h="$(hash_toolchain "${2:-}")" || return 1
    shift 2
    set -- "$@" "rustc=$h"
  fi
  local bind="$1"; shift
  [ "$#" -ge 1 ] || { echo "archive_bind_check: no keys to bind on -- a bind of zero keys binds nothing" >&2; return 1; }
  [ -s "$bind" ] || { echo "archive_bind_check: $bind missing or empty -- the quick tier needs the archive" >&2; return 1; }
  [ "$(jq -r '.tier // empty' "$bind")" = quick ] || { echo "archive_bind_check: archive job decided tier '$(jq -r '.tier // "?"' "$bind")', this shard is quick" >&2; return 1; }
  local kv k v got bad=0
  for kv in "$@"; do
    k="${kv%%=*}"; v="${kv#*=}"
    got="$(jq -r --arg k "$k" '.[$k] // empty' "$bind")"
    [ -n "$v" ] && [ "$got" = "$v" ] || { printf 'archive_bind_check: %s differs (archive=%s shard=%s)\n' "$k" "$got" "$v" >&2; bad=1; }
  done
  [ "$bad" = 0 ] || return 1
  echo "archive_bind_check: archive is bound to this run ($#) keys match"
}

self_test() {
  local d; d="$(mktemp -d)"
  jq -n '{tier:"quick",sha:"abc",expr:"a | b",flags:"-p x",rustc:"r1"}' > "$d/b.json"
  check "$d/b.json" sha=abc "expr=a | b" "flags=-p x" rustc=r1 >/dev/null || { echo "FAIL: matching bind rejected" >&2; return 1; }
  local m
  for m in sha=zzz "expr=a" "flags=-p y" rustc=r2 sha= ; do
    if check "$d/b.json" sha=abc "expr=a | b" "flags=-p x" rustc=r1 "$m" >/dev/null 2>&1; then echo "FAIL: mutant $m accepted" >&2; return 1; fi
  done
  if check "$d/b.json" >/dev/null 2>&1; then echo "FAIL: zero keys accepted" >&2; return 1; fi
  jq -n '{tier:"reuse",sha:"abc"}' > "$d/r.json"
  if check "$d/r.json" sha=abc >/dev/null 2>&1; then echo "FAIL: non-quick tier accepted" >&2; return 1; fi
  if check "$d/none.json" sha=abc >/dev/null 2>&1; then echo "FAIL: missing bind accepted" >&2; return 1; fi
  # empty VALUE in the bind and on the shard: equal, and still a refusal
  jq -n '{tier:"quick",sha:""}' > "$d/e.json"
  if check "$d/e.json" sha= >/dev/null 2>&1; then echo "FAIL: empty==empty accepted" >&2; return 1; fi
  # empty / rustc-less toolchain text (a failed docker run) is refused, a real one hashes
  if hash_toolchain "" >/dev/null 2>&1; then echo "FAIL: empty toolchain hashed" >&2; return 1; fi
  if hash_toolchain "error: nope" >/dev/null 2>&1; then echo "FAIL: non-rustc text hashed" >&2; return 1; fi
  [ "$(hash_toolchain "rustc 1.99.0 (abc 2026-01-01)")" != "" ] || { echo "FAIL: real toolchain refused" >&2; return 1; }
  # the CLI entry points the workflow calls (round 4: --hash-toolchain was never dispatched)
  local h; h="$(bash "$0" --hash-toolchain "rustc 1.99.0 (abc 2026-01-01)")" || { echo "FAIL: --hash-toolchain CLI failed on real text" >&2; return 1; }
  [ "${#h}" = 16 ] || { echo "FAIL: --hash-toolchain printed '$h'" >&2; return 1; }
  if bash "$0" --hash-toolchain "" >/dev/null 2>&1; then echo "FAIL: --hash-toolchain CLI accepted empty" >&2; return 1; fi
  jq -n --arg r "$h" '{tier:"quick",sha:"abc",rustc:$r}' > "$d/cli.json"
  bash "$0" --toolchain "rustc 1.99.0 (abc 2026-01-01)" "$d/cli.json" sha=abc >/dev/null || { echo "FAIL: --toolchain CLI did not bind the hash" >&2; return 1; }
  if bash "$0" --toolchain "rustc 2.0.0" "$d/cli.json" sha=abc >/dev/null 2>&1; then echo "FAIL: --toolchain accepted a different toolchain" >&2; return 1; fi
  if bash "$0" --toolchain "" "$d/cli.json" sha=abc >/dev/null 2>&1; then echo "FAIL: --toolchain accepted empty text" >&2; return 1; fi
  # mutants of this script itself must each be caught by the cases above
  local mut
  for mut in 's/--hash-toolchain) hash_toolchain/--hash-toolchain) true/' 's/\[ -n "\$v" \] && //' 's/\*"rustc "\*) ;;/*) ;;/'; do
    sed "$mut" "$0" > "$d/mut.sh"
    if cmp -s "$0" "$d/mut.sh"; then echo "FAIL: mutant did not apply: $mut" >&2; return 1; fi
    if bash "$d/mut.sh" --self-test >/dev/null 2>&1; then echo "FAIL: surviving mutant: $mut" >&2; return 1; fi
  done
  rm -rf "${d:?}"
  echo "archive_bind_check self-test: ok (match passes; sha/expr/flags/rustc/empty/tier/missing rejected)"
}

case "${1:-}" in
  --self-test) self_test ;;
  --hash-toolchain) hash_toolchain "${2:-}" ;;
  *) check "$@" ;;
esac
