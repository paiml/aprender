#!/usr/bin/env bash
# sccache_probe.sh DIR: print the directory a container should mount as /sccache.
#
# Prints DIR when this user can create and delete a file in it (a real write test, not `test -w`,
# which misses read-only mounts and ACL/ownership cases). Otherwise prints a fresh writable
# directory under ${RUNNER_TEMP:-${TMPDIR:-/tmp}} and logs ONE line on stderr saying so, so the
# build runs with a cold but WORKING sccache instead of failing a build script with
# "sccache: error: ... Permission denied" (main run 44583965968, guard-cargo#m60, burst runner).
#
#   usage: sccache_probe.sh DIR      prints the chosen dir on stdout
#          sccache_probe.sh --self-test
set -euo pipefail

probe_writable() { # dir -> 0 when a file can be created and removed in it
  local f
  mkdir -p "${1:?}" 2>/dev/null || return 1
  f=$(mktemp "${1:?}/.sccache-probe.XXXXXX" 2>/dev/null) || return 1
  rm -f "${f:?}" 2>/dev/null || return 1
  return 0
}

choose() { # dir -> chosen dir on stdout, one log line on stderr when it fell back
  local want=$1 alt
  if probe_writable "$want"; then
    printf '%s\n' "$want"
    return 0
  fi
  alt=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/sccache.XXXXXX")
  printf 'sccache_probe: %s is not writable by uid %s; using %s (cold cache, sccache still works)\n' "$want" "$(id -u)" "$alt" >&2
  printf '%s\n' "$alt"
}

if [ "${1:-}" = "--self-test" ]; then
  fails=0
  t=$(mktemp -d)
  trap 'rm -rf "${t:?}"' EXIT
  # row 1: a writable dir is returned as is, with no log line
  got=$(choose "$t/ok" 2>"$t/err1")
  if [ "$got" = "$t/ok" ] && [ ! -s "$t/err1" ]; then echo "ok    row 1 writable dir is kept, no log"; else echo "FAIL  row 1 got=$got err=$(cat "$t/err1")"; fails=1; fi
  # row 2: a dir that cannot be created (parent is a regular file) falls back, with exactly one log line
  : > "$t/file"
  got=$(RUNNER_TEMP="$t" choose "$t/file/sub" 2>"$t/err2")
  if [ "$got" != "$t/file/sub" ] && [ -d "$got" ] && probe_writable "$got" && [ "$(wc -l < "$t/err2")" = 1 ]; then
    echo "ok    row 2 unusable dir falls back to a writable temp dir, one log line"
  else echo "FAIL  row 2 got=$got err=$(cat "$t/err2")"; fails=1; fi
  # row 3: an existing dir without write permission falls back (skipped when uid 0 ignores modes)
  mkdir "$t/ro"; chmod 555 "$t/ro"
  if [ "$(id -u)" = 0 ]; then echo "ok    row 3 skipped: uid 0 ignores directory modes"
  else
    got=$(RUNNER_TEMP="$t" choose "$t/ro" 2>/dev/null)
    if [ "$got" != "$t/ro" ] && probe_writable "$got"; then echo "ok    row 3 read-only dir falls back"; else echo "FAIL  row 3 got=$got"; fails=1; fi
  fi
  chmod 755 "$t/ro"
  exit "$fails"
fi

[ "$#" -eq 1 ] || { echo "usage: sccache_probe.sh DIR | --self-test" >&2; exit 2; }
choose "$1"
