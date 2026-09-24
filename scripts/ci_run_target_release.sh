#!/usr/bin/env bash
# ci_run_target_release.sh — last-one-out release of a CI job's per-RUN target dir (#4102).
#
# WHY. One #4046 shard left 64G of target on gx10: nothing deletes a run's target dir when its job ends, only the
# NEXT run's start-of-job reclaim (scripts/ci_reclaim_target_dirs.sh), and only for runs it can prove dead. An
# `if: always()` step that simply `rm -rf`s the run dir is the obvious fix, and it is WRONG as a general rule: a
# run dir is keyed per RUN, and ci_reclaim_target_dirs.sh measured two jobs of one run, on two runners of one host,
# sharing `run-<RUN_ID>` with windows overlapping 21-29 min. Today each job's dir is its own (shards are keyed
# `<PR>-s<N>`, guard-cargo `run-<id>-guards`), but a re-run reuses the RUN_ID and a future re-key can share again.
# So the dir is deleted by the LAST job out, never by the first.
#
# HOW. Every job REGISTERS a marker `<dir>/.live/<token>` before its first cargo step, and RELEASES it in an
# `if: always()` step. Register, release and the delete all run under ONE flock on `<dir>.lock`, a sibling of the
# dir on the SAME host-local filesystem, so:
#   - a release that finds another live marker keeps the tree;
#   - two releases racing serialize; exactly one of them sees the last marker go and deletes, never zero, never both;
#   - a sibling cannot register between the empty-check and the delete (the delete is inside the lock).
# A job killed without running its release leaves its marker: the tree is kept, and the start-of-job reclaim,
# which checks run liveness through the API, stays the backstop. Nothing live is ever deleted by this script.
#
#   ci_run_target_release.sh register <dir> <token>
#   ci_run_target_release.sh release  <dir> <token> [-- <delete command…>]   (default delete: rm -rf -- <dir>)
#   ci_run_target_release.sh --self-test
#
# The delete command gets the dir as its last argument. CI passes a `docker run … rm -rf` wrapper, because the
# tree holds root-owned files from the build container.
#
# Exit: 0 done (release prints `released: deleted <dir>` or `released: kept <dir> (live: …)`) · 1 the delete failed
#       · 2 usage, or the lock could not be taken. A cleanup step must never fail the build: CI treats non-zero as a
#       warning.
set -uo pipefail
PROG=ci_run_target_release
LOCK_WAIT=${CI_RELEASE_LOCK_WAIT:-120}

die() { echo "$PROG: $1" >&2; exit 2; }

valid_token() { case "$1" in ''|*/*|.*) return 1 ;; *) return 0 ;; esac; }

do_register() { # <dir> <token>
  local dir=$1 token=$2
  mkdir -p "$dir/.live" || die "cannot create $dir/.live"
  : > "$dir/.live/$token" || die "cannot write the marker $dir/.live/$token"
  echo "registered: $token on $dir ($(live_list "$dir"))"
}

live_list() { # <dir> -> comma list of live markers, or "none"
  local names=() f
  if [ -d "$1/.live" ]; then
    for f in "$1"/.live/*; do [ -e "$f" ] && names+=("$(basename "$f")"); done
  fi
  if [ "${#names[@]}" -eq 0 ]; then echo none; else local IFS=,; echo "${names[*]}"; fi
}

do_release() { # <dir> <token> <delete cmd…>
  local dir=$1 token=$2; shift 2
  [ -d "$dir" ] || { echo "released: $dir is already gone"; return 0; }
  rm -f -- "$dir/.live/$token"
  local live; live=$(live_list "$dir")
  if [ "$live" != none ]; then
    echo "released: kept $dir (live: $live)"
    return 0
  fi
  if [ "$#" -eq 0 ]; then set -- rm -rf --; fi
  if "$@" "$dir"; then
    # Still holding the lock: unlink it too. A waiter on this inode re-opens the path (see locked()).
    rm -f -- "${dir%/}.lock"
    echo "released: deleted $dir"
    return 0
  fi
  echo "released: FAILED to delete $dir" >&2
  return 1
}

locked() { # <dir> <fn> <args…>: run fn under the dir's flock (a sibling path, same filesystem)
  # The last release UNLINKS the lock file (else one 0-byte file per job-run leaks, #4102 round 2). A waiter that
  # was blocked on the old inode would then hold a lock nobody else can see, so after every acquire the fd's inode
  # must still be the path's; if not, re-open the path and lock again. Standard lock-file-with-unlink protocol.
  local dir=$1 lock tries=0; shift
  lock="${dir%/}.lock"
  while :; do
    exec 9>> "$lock" || die "cannot open the lock $lock"
    flock -w "$LOCK_WAIT" 9 || die "the lock $lock was not free within ${LOCK_WAIT}s"
    [ "$(stat -L -c %i /proc/self/fd/9 2> /dev/null)" = "$(stat -c %i "$lock" 2> /dev/null)" ] && break
    exec 9>&-
    tries=$((tries + 1))
    [ "$tries" -lt 50 ] || die "the lock $lock was replaced 50 times while waiting"
  done
  "$@"
  local rc=$?
  exec 9>&-
  return "$rc"
}

self_test() {
  local T fails=0 n
  T=$(mktemp -d -t ci-release.XXXXXX) || exit 2
  # shellcheck disable=SC2064
  trap "rm -rf -- '$T'" RETURN
  case_line() { printf '  %-4s %s\n' "$1" "$2"; [ "$1" = ok ] || fails=$((fails + 1)); }
  me=$0

  d="$T/a/run-1"; mkdir -p "$d"; echo x > "$d/f"
  bash "$me" register "$d" job-1 > /dev/null
  out=$(bash "$me" release "$d" job-1); rc=$?
  { [ "$rc" = 0 ] && [ ! -e "$d" ] && grep -q "released: deleted" <<< "$out"; } \
    && case_line ok "the only job out deletes its tree" || case_line FAIL "the only job out did not delete ($out)"

  d="$T/b/run-1"; mkdir -p "$d"; echo x > "$d/f"
  bash "$me" register "$d" shard-1 > /dev/null; bash "$me" register "$d" shard-2 > /dev/null
  out=$(bash "$me" release "$d" shard-1)
  { [ -f "$d/f" ] && grep -q "kept .*(live: shard-2)" <<< "$out"; } \
    && case_line ok "a live sibling keeps the tree, and is named" || case_line FAIL "a live sibling's tree was touched ($out)"
  out=$(bash "$me" release "$d" shard-2)
  [ ! -e "$d" ] && case_line ok "the last sibling out deletes" || case_line FAIL "the last sibling out did not delete ($out)"
  [ ! -e "$d.lock" ] && case_line ok "the last release unlinks the lock file (no 0-byte leak per run)" \
    || case_line FAIL "the lock file $d.lock was left behind"

  # must-RED: a waiter blocked on a lock whose file is then unlinked must NOT proceed on the stale inode while a
  # newcomer holds the fresh one. A holds the old inode; W (register) waits on it; A unlinks and a newcomer N takes
  # the fresh file for 2 s; A releases. W must finish only AFTER N releases. N closes the inherited fd 8 first:
  # a lock is held until EVERY fd on its open file is closed, so an inherited fd 8 would keep W waiting anyway and
  # make this case pass without the inode re-check (measured: it did, before this line).
  d="$T/g/run-1"; mkdir -p "$d"; lk="$d.lock"; : > "$lk"; ord="$T/g.order"
  ( exec 8>> "$lk"; flock 8; sleep 1; rm -f "$lk"; ( exec 8>&-; exec 7>> "$lk"; flock 7; echo "N-holds" >> "$ord"; sleep 2; echo "N-releases" >> "$ord" ) & sleep 0.3; exit 0 ) &
  sleep 0.2
  ( bash "$me" register "$d" w > /dev/null 2>&1; echo "W-registered" >> "$ord" ) &
  wait
  if [ "$(tr '\n' ' ' < "$ord")" = "N-holds N-releases W-registered " ]; then
    case_line ok "a waiter on an unlinked lock re-opens and waits for the fresh one (no two holders)"
  else
    case_line FAIL "two holders: order was '$(tr '\n' ' ' < "$ord")'"
  fi

  # must-RED (cop ruling, 2026-09-24): two jobs finishing TOGETHER. Exactly one deletes: never zero, never both.
  for i in 1 2 3 4 5 6 7 8; do
    d="$T/race-$i/run-1"; mkdir -p "$d"; echo x > "$d/f"; log="$T/race-$i.log"
    bash "$me" register "$d" j1 > /dev/null; bash "$me" register "$d" j2 > /dev/null
    counter() { echo del >> "${1:?}.dels"; rm -rf -- "${1:?}"; }
    export -f counter
    ( bash "$me" release "$d" j1 -- bash -c 'counter "$0"' >> "$log" 2>&1 ) &
    ( bash "$me" release "$d" j2 -- bash -c 'counter "$0"' >> "$log" 2>&1 ) &
    wait
    n=$(grep -c . "$d.dels" 2>/dev/null || echo 0)
    if [ "$n" != 1 ] || [ -e "$d" ]; then
      case_line FAIL "race $i: $n delete(s), tree $( [ -e "$d" ] && echo left || echo gone) — want exactly one, gone"
    fi
  done
  [ "$fails" = 0 ] && case_line ok "8 races of two jobs finishing together: exactly one delete each, tree gone"

  d="$T/c/run-1"; mkdir -p "$d"
  bash "$me" register "$d" crashed > /dev/null; bash "$me" register "$d" me > /dev/null
  out=$(bash "$me" release "$d" me)
  [ -d "$d" ] && case_line ok "a job that died without releasing keeps the tree (the start-of-job reclaim is the backstop)" \
    || case_line FAIL "a crashed job's marker was ignored ($out)"

  d="$T/e/run-1"; mkdir -p "$d"; bash "$me" register "$d" j > /dev/null
  out=$(bash "$me" release "$d" j -- false 2>&1); rc=$?
  { [ "$rc" = 1 ] && grep -q FAILED <<< "$out"; } && case_line ok "a failed delete exits 1 and says so" \
    || case_line FAIL "a failed delete was reported as success (rc $rc)"

  out=$(bash "$me" release "$T/never-existed/run-1" j); rc=$?
  [ "$rc" = 0 ] && case_line ok "releasing a tree that is already gone is not an error" || case_line FAIL "rc $rc on a gone tree"

  for bad in "" "../x" ".hidden" "a/b"; do
    bash "$me" register "$T/f/run-1" "$bad" > /dev/null 2>&1; rc=$?
    [ "$rc" = 2 ] || case_line FAIL "token '$bad' accepted (rc $rc)"
  done
  [ ! -e "$T/f/run-1/.live/x" ] && case_line ok "empty, '..', dotted and slashed tokens are refused (rc 2)"

  if [ "$fails" = 0 ]; then echo "$PROG --self-test: PASS"; return 0; fi
  echo "$PROG --self-test: FAIL ($fails)"; return 1
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  register|release)
    cmd=$1; dir=${2:-}; token=${3:-}
    [ -n "$dir" ] || die "usage: $cmd <dir> <token>"
    valid_token "$token" || die "bad token '$token' (non-empty; no '/', no leading '.')"
    shift 3
    if [ "$cmd" = register ]; then
      mkdir -p "$(dirname "$dir")" || die "cannot create $(dirname "$dir")"
      locked "$dir" do_register "$dir" "$token"
    else
      [ "${1:-}" = "--" ] && shift
      [ -d "$(dirname "$dir")" ] || { echo "released: $dir is already gone"; exit 0; }
      locked "$dir" do_release "$dir" "$token" "$@"
    fi
    exit $?
    ;;
  *) die "usage: register|release <dir> <token> [-- <delete cmd>] | --self-test" ;;
esac
