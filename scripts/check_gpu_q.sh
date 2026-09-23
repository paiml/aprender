#!/usr/bin/env bash
# check_gpu_q.sh — the case table for scripts/gpu-q (#3966). Hermetic: GPUQ_DIR and GPUQ_LOCK point every
# case at a private queue and lock, so it never touches /tmp/apr-gpu.lock or a real job.
#
# WHY. Two gpu-q waiters on gx10 spun for 34 hours after their tickets vanished (aprender-3a, 2026-09-23):
# the wait loop only ever asked "is the head ticket mine?", so a waiter with NO ticket never matched,
# never re-enqueued and never failed. They also ignored SIGTERM, because a signal ignored at shell entry
# cannot be trapped by bash, so only SIGKILL removed them.
#
#   free          lock free                                       -> runs the command, rc 0
#   vanished      a live waiter's ticket is deleted               -> exits NON-ZERO (76) within one poll
#   term-ignored  waiter started with TERM ignored, then TERMed   -> exits, and its ticket is gone
#   long-wait     waiting past GPUQ_REPORT_S                      -> says so on stderr, naming the holder
#   prune         --prune with one dead and one live ticket       -> the dead one removed by name, the live kept
#
# Usage: check_gpu_q.sh [--gpu-q PATH]   (default: scripts/gpu-q beside this script)
# Exit: 0 every case as expected · 1 a case broke · 2 could not check.
set -uo pipefail

GQ="$(cd "$(dirname "$0")" && pwd)/gpu-q"
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-q) GQ="$2"; shift 2 ;;
    *) echo "check_gpu_q: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -x "$GQ" ] || { echo "check_gpu_q: $GQ is not executable" >&2; exit 2; }
for tool in flock pgrep; do command -v "$tool" >/dev/null 2>&1 || { echo "check_gpu_q: $tool missing" >&2; exit 2; }; done

TMP=$(mktemp -d) || exit 2
HOLDERS=()
cleanup() {
  local h
  for h in "${HOLDERS[@]}"; do pkill -P "$h" 2>/dev/null; kill "$h" 2>/dev/null; done
  rm -rf "$TMP"
}
trap cleanup EXIT
fails=0
ok()    { echo "  ok    $1"; }
broke() { echo "  BROKE $1"; fails=1; }

setup() { # setup <case>: private queue + lock, lock HELD by a sleeper until the case ends
  export GPUQ_DIR="$TMP/$1/q" GPUQ_LOCK="$TMP/$1/lock"
  mkdir -p "$TMP/$1"
  if [ "${2:-hold}" = hold ]; then
    # The holder is itself a gpu-q job at prio 0, so its ticket stays at the HEAD: the waiter under test
    # then sits in the ticket loop, which is where the 34 h waiters were. A bare flock holder leaves the
    # waiter alone at the head, where it skips the loop and blocks in flock.
    "$GQ" --prio 0 -- sleep 120 2>/dev/null & HOLDERS+=("$!")
    sleep 0.5
  fi
}
my_ticket() { grep -ls "^$1 " "$GPUQ_DIR"/* 2>/dev/null | head -1; }
wait_ticket() { local i; for i in $(seq 1 30); do [ -n "$(my_ticket "$1")" ] && return 0; sleep 0.1; done; return 1; }

# free
setup free nohold
"$GQ" -- true 2>/dev/null; rc=$?
[ "$rc" -eq 0 ] && ok "free: ran the command (rc 0)" || broke "free: rc $rc, want 0"

# vanished (#3966's must-RED)
setup vanished
"$GQ" -- true 2> "$TMP/vanished.err" & w=$!
if wait_ticket "$w"; then
  rm -f "$(my_ticket "$w")"
  t0=$(date +%s)
  for _ in $(seq 1 80); do kill -0 "$w" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$w" 2>/dev/null; then
    broke "vanished: the waiter is still spinning 8 s after its ticket was deleted (the 34 h state)"
    kill -9 "$w" 2>/dev/null
  else
    wait "$w"; rc=$?
    if [ "$rc" -eq 76 ] && grep -q 'ticket' "$TMP/vanished.err"; then
      ok "vanished: exited rc 76 within $(( $(date +%s) - t0 ))s, naming its ticket"
    else
      broke "vanished: exited rc $rc (want 76 with a stderr line naming the ticket): $(tail -1 "$TMP/vanished.err")"
    fi
  fi
else
  broke "vanished: the waiter never wrote a ticket"; kill -9 "$w" 2>/dev/null
fi

# term-ignored
setup termign
( trap '' TERM; exec "$GQ" -- true ) 2>/dev/null & w=$!
if wait_ticket "$w"; then
  kill -TERM "$w"
  for _ in $(seq 1 30); do kill -0 "$w" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$w" 2>/dev/null; then
    broke "term-ignored: survived SIGTERM (only SIGKILL would remove it)"; kill -9 "$w" 2>/dev/null
  elif [ -n "$(my_ticket "$w")" ]; then
    broke "term-ignored: exited but left its ticket behind"
  else
    ok "term-ignored: SIGTERM ended it and its ticket is gone"
  fi
else
  broke "term-ignored: the waiter never wrote a ticket"; kill -9 "$w" 2>/dev/null
fi

# long-wait
setup longwait
GPUQ_REPORT_S=2 "$GQ" -- true 2> "$TMP/longwait.err" & w=$!
sleep 7
kill -TERM "$w" 2>/dev/null; wait "$w" 2>/dev/null
if grep -q 'still waiting' "$TMP/longwait.err"; then
  ok "long-wait: reported itself ($(grep -m1 'still waiting' "$TMP/longwait.err" | cut -c1-90))"
else
  broke "long-wait: no 'still waiting' report after GPUQ_REPORT_S=2 and 7 s"
fi

# prune: a dead holder's ticket (its pid exited) is removed; a live waiter's ticket is kept
setup prune
"$GQ" -- true 2>/dev/null & live=$!
if wait_ticket "$live"; then
  sh -c 'exit 0' & dead=$!; wait "$dead"
  printf '%s %s\n' "$dead" 12345 > "$GPUQ_DIR/5-1-$dead"
  "$GQ" --prune > "$TMP/prune.out" 2>&1; rc=$?
  if [ "$rc" -eq 0 ] && [ ! -e "$GPUQ_DIR/5-1-$dead" ] && [ -n "$(my_ticket "$live")" ] && grep -q "pruned dead ticket 5-1-$dead" "$TMP/prune.out"; then
    ok "prune: removed the dead ticket by name, kept the live one"
  else
    broke "prune: rc $rc; dead ticket $( [ -e "$GPUQ_DIR/5-1-$dead" ] && echo KEPT || echo removed ), live ticket $( [ -n "$(my_ticket "$live")" ] && echo kept || echo REMOVED ): $(tail -1 "$TMP/prune.out")"
  fi
else
  broke "prune: the live waiter never wrote a ticket"
fi
kill "$live" 2>/dev/null

[ "$fails" -eq 0 ] && { echo "check_gpu_q: PASS"; exit 0; }
echo "check_gpu_q: FAIL"; exit 1
