#!/usr/bin/env bash
# FLOW-010 P1: a nextest archive is only used by the run, commit, filterset, build flags
# and image it was built for. Fails (rc 1) on any mismatch, a missing bind.json, a tier
# that is not quick, or a missing key -- the quick tier has no second path.
# usage: archive_bind_check.sh BIND.json sha=… expr=… flags=… rustc=…
#        archive_bind_check.sh --self-test
set -euo pipefail

check() {
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
  rm -rf "${d:?}"
  echo "archive_bind_check self-test: ok (match passes; sha/expr/flags/rustc/empty/tier/missing rejected)"
}

if [ "${1:-}" = "--self-test" ]; then self_test; else check "$@"; fi
