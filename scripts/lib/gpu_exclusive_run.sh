#!/usr/bin/env bash
# gpu_exclusive_run.sh -- run one command on this host's GPU with PROOF it had the card
# to itself, and refuse the result when it did not (#3964).
#
# WHY THE FLEET LOCK IS NOT ENOUGH. /tmp/apr-gpu.lock coordinates only the processes that
# take it. Ollama's llama-server is a system daemon that loads a model on any request and
# holds it for its keep_alive; it never takes the lock. On 2026-09-23 it arrived MID-RUN
# (1328 MiB) during a device A/B whose "card clear" check had passed at the start -- the
# same process and footprint that poisoned a cuda suite into 34 failures a week earlier.
# A clear card at the START proves nothing about the run.
#
# SO THIS SCRIPT:
#   1. waits for an EMPTY card WITHOUT holding the lock -- holding the fleet lock while
#      idle-waiting on a process that ignores it blocks every session queued behind you
#      and buys nothing;
#   2. takes the lock, then RE-CHECKS the card (it may have filled while we waited);
#   3. samples every GPU process by FULL PATH every $GPU_SAMPLE_SECS (default 0.1 s) for
#      the WHOLE run. It was 1 s, and a planted-fault A/B -- which panics on its first
#      tensor -- finished between two samples and could not be verified. nvidia-smi
#      answers in 10-20 ms here, so 0.1 s is cheap;
#   4. exits 75 (EX_TEMPFAIL) -- "CONTENDED", never a verdict -- if anything outside
#      $GPU_OWNED_PREFIX appeared. The command's own exit status is otherwise returned.
#
# Usage:
#   GPU_OWNED_PREFIX=/path/to/your/target/ scripts/lib/gpu_exclusive_run.sh <cmd> [args...]
# Env:
#   GPU_OWNED_PREFIX   REQUIRED. Process paths starting with this are yours; nothing else
#                      may appear on the card during the run. No default: a default would
#                      silently accept every process as "yours".
#   GPU_WAIT_SECS      how long to wait for an empty card before giving up (default 1800)
#   GPU_LOCK           lock file (default /tmp/apr-gpu.lock)
#   GPU_OCC_LOG        where the occupancy samples go (default: a temp file)
#   GPU_SAMPLE_SECS    sampling interval (default 0.1). A run shorter than this can still be
#                      missed; that case exits 75 as UNVERIFIED rather than claiming anything.
# Exit: the command's status; 75 = CONTENDED or the card never cleared; 2 = usage.
#
# SIGNALS. A `trap ... TERM` whose handler does not `exit` makes bash run the handler and
# then CARRY ON -- found the hard way, when a stopped mutation runner restored its file and
# went back into its wait loop, where it would have run a "fault" against the restored,
# correct kernel and reported the fault as SURVIVING. Every handler here exits.
set -u
[ "$#" -ge 1 ] || { echo "usage: GPU_OWNED_PREFIX=... $0 <cmd> [args...]" >&2; exit 2; }
PREFIX="${GPU_OWNED_PREFIX:-}"
[ -n "$PREFIX" ] || { echo "gpu_exclusive_run: GPU_OWNED_PREFIX is required" >&2; exit 2; }
WAIT="${GPU_WAIT_SECS:-1800}"
LOCK="${GPU_LOCK:-/tmp/apr-gpu.lock}"
OCC="${GPU_OCC_LOG:-$(mktemp)}"

card_apps() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null; }
wait_empty() {
  local waited=0
  while [ -n "$(card_apps)" ]; do
    [ "$waited" -ge "$WAIT" ] && return 1
    sleep 5; waited=$((waited + 5))
  done
  return 0
}

# INHERITED LOCK. A caller that already holds $LOCK -- `gpu-q` ends in `exec flock $LOCK
# choom -- <cmd>`, the ladder's apr_locked does the same -- used to DEADLOCK here: `flock 9`
# opens a NEW file description, flock(2) locks conflict across descriptions, and the inner
# call waited forever while the outer one held the fleet lock, stalling every GPU job on
# the box (proven on a private lock, aprender-70). It also mis-read a legitimate holder: the
# lockless wait_empty below saw the lock-holder's own GPU use and called it contention.
# So: if an ANCESTOR of this process holds the lock, use that hold. Read from /proc/locks,
# not from an environment marker -- a marker can be set by anyone, a held flock by an
# ancestor cannot be faked -- and ancestry, not "any holder": a lock held by an unrelated
# process is exactly the contention this script exists to wait for.
lock_holder_ancestor() { # -> the ancestor pid holding $LOCK, or nothing
  local ino holders p
  [ -e "$LOCK" ] || return 1
  ino=$(stat -Lc %i "$LOCK" 2>/dev/null) || return 1
  # "1: FLOCK ADVISORY WRITE <pid> <maj:min:inode> ..."; blocked waiters read "1: -> FLOCK".
  holders=$(awk -v i=":$ino" '$2 == "FLOCK" && substr($6, length($6) - length(i) + 1) == i { print $5 }' /proc/locks 2>/dev/null)
  [ -n "$holders" ] || return 1
  p=$PPID
  while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    if printf '%s\n' "$holders" | grep -qx "$p"; then printf '%s' "$p"; return 0; fi
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
  return 1
}

TOOK_LOCK=0
release_lock() { [ "$TOOK_LOCK" = 1 ] && flock -u 9; TOOK_LOCK=0; }
if holder=$(lock_holder_ancestor); then
  echo "gpu_exclusive_run: $LOCK is held by ancestor pid $holder -- using that hold, not re-taking it" >&2
else
  wait_empty || { echo "gpu_exclusive_run: CONTENDED -- card never cleared in ${WAIT}s: $(card_apps | tr '\n' ' ')" >&2; exit 75; }
  exec 9>"$LOCK"
  # BOUNDED. An unbounded wait here is what turned the inherited-lock case into a hang;
  # if the ancestry check above ever regresses, this makes it a named refusal instead.
  flock -w "$WAIT" 9 || { echo "gpu_exclusive_run: CONTENDED -- $LOCK not free after ${WAIT}s" >&2; exit 75; }
  TOOK_LOCK=1
fi
SAMPLER=""
cleanup() { [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null; }
trap 'cleanup' EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# The card may have filled while we queued for the lock.
if [ -n "$(card_apps)" ]; then
  release_lock
  echo "gpu_exclusive_run: CONTENDED -- card occupied after taking the lock: $(card_apps | tr '\n' ' ')" >&2
  exit 75
fi

: > "$OCC"
( while :; do card_apps >> "$OCC"; sleep "${GPU_SAMPLE_SECS:-0.1}"; done ) &
SAMPLER=$!
"$@"
rc=$?
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; SAMPLER=""
release_lock

# A sample whose name nvidia-smi cannot resolve reads "[No data]" -- which is what OUR OWN
# process looks like at teardown: the first release of this script refused a clean run
# because its own exiting test binary (the same pid, sampled six times with our path just
# before) appeared once as "<pid>, [No data]". So a nameless sample is ours ONLY if that
# pid was already seen under $PREFIX in this run. A nameless pid never seen with a path
# stays FOREIGN: it could be a foreigner starting up, and the conservative reading wins.
mine_pids=$(grep -F "$PREFIX" "$OCC" | cut -d, -f1 | sort -u)
foreign=$(grep -v '^$' "$OCC" | grep -vF "$PREFIX" | while IFS=, read -r pid rest; do
  printf '%s\n' "$mine_pids" | grep -qx "$pid" || printf '%s,%s\n' "$pid" "$rest"
done | sort -u)
if [ -n "$foreign" ]; then
  echo "gpu_exclusive_run: CONTENDED -- foreign GPU process(es) during the run, result refused:" >&2
  echo "$foreign" | sed 's/^/    /' >&2
  exit 75
fi
mine=$(grep -cF "$PREFIX" "$OCC")
if [ "$mine" -eq 0 ]; then
  # Nothing was observed at all -- neither this run nor a foreigner. Exclusivity that no
  # sample saw is not exclusivity; a run shorter than the sampling interval cannot claim it.
  echo "gpu_exclusive_run: UNVERIFIED -- no sample caught the run, so exclusivity was not observed" >&2
  exit 75
fi
echo "gpu_exclusive_run: EXCLUSIVE -- $mine sample(s), every one under $PREFIX" >&2
exit "$rc"
