#!/usr/bin/env bash
# run_t2_lambda.sh -- bounded-run wrapper for ONE T-2 preflight (t2_preflight.sh) on lambda-labs (APR-RELEASE-001 T-2).
# Version- and commit-parameterised: nothing about a release is hardcoded here.
#   run_t2_lambda.sh <version> <commit> --check-only        run every preflight assertion, exit non-zero on any violation, start nothing
#   run_t2_lambda.sh <version> <commit> --run               assertions, then clone+run bounded (needs SRC_REPO, RELEASE_EPIC)
#   run_t2_lambda.sh <version> <commit> --test-run -- CMD   test hook: same launcher/watcher/cleanup, CMD replaces t2_preflight
# There is NO override flag. Test-only env knobs (FLOOR_GB_TEST, POLL_S_TEST, STALL_S_TEST, T2_*) are honoured ONLY with
# --check-only / --test-run, and are unset in --run. Gates: disk floor 210G, MemAvailable >= 64GiB, load, swap, idle GPU.
set -euo pipefail

usage() { echo "usage: $0 <version> <commit-sha40> --check-only | --run | --test-run -- CMD..." >&2; exit 64; }
[ $# -ge 3 ] || usage
VERSION="$1"; SHA="$2"; shift 2
[[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "bad version: $VERSION" >&2; exit 64; }
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || { echo "bad commit (need a full 40-hex sha): $SHA" >&2; exit 64; }

MODE=""; TEST_CMD=()
case "$1" in
  --check-only) MODE=check ;;
  --run) MODE=run ;;
  --test-run) MODE=test; shift; [ "${1:-}" = "--" ] && shift; TEST_CMD=("$@"); [ "${#TEST_CMD[@]}" -gt 0 ] || usage ;;
  *) echo "unknown arg: $1" >&2; exit 64 ;;
esac
[ "$MODE" = test ] || [ $# -eq 1 ] || usage

EPIC="${RELEASE_EPIC:-}"
if [ "$MODE" = run ]; then
  [[ $EPIC =~ ^[0-9]+$ ]] || { echo "RELEASE_EPIC (release epic issue number) required for --run" >&2; exit 64; }
else EPIC="${EPIC:-0}"; fi
FLOOR_GB=210; WATCH_FLOOR_GB=210; POLL_S=30; STALL_S=3600; LOAD_MAX=40; MEM_MIN_KB=$((64*1024*1024)); CAP="6h"
if [ "$MODE" = run ]; then
  # overrides are inert in real runs
  unset WATCH_FLOOR_GB_TEST T2_LOAD1 T2_MEMAVAIL_KB T2_NVSMI_CMD T2_SWAP_GROWTH_KB T2_TMPDIR T2_CAP FLOOR_GB_TEST POLL_S_TEST STALL_S_TEST
else
  FLOOR_GB="${FLOOR_GB_TEST:-$FLOOR_GB}"; WATCH_FLOOR_GB="${WATCH_FLOOR_GB_TEST:-$WATCH_FLOOR_GB}"
  POLL_S="${POLL_S_TEST:-$POLL_S}"; STALL_S="${STALL_S_TEST:-$STALL_S}"
fi
[ "$POLL_S" -le 60 ] || { echo "FAIL: POLL_S>60" >&2; exit 1; }

RAID_ROOT="${RAID_ROOT:-/mnt/nvme-raid0}"
RUN_ID="t2-$VERSION-$(date -u +%Y%m%dT%H%M%SZ)-$$"
CHECKOUT="${CHECKOUT:-$RAID_ROOT/$RUN_ID/checkout}"
RUN_DIR="$(dirname "$CHECKOUT")"
TMPDIR="${T2_TMPDIR:-$RAID_ROOT/$RUN_ID-tmp}"
WORKLOG="${WORKLOG:-$RAID_ROOT/$RUN_ID-worklog}"
RECEIPT_DIR="${RECEIPT_DIR:-$RAID_ROOT/t2-lambda-receipts}"
REPO_ROOT="${REPO_ROOT:-$CHECKOUT}"
CARGO_HOME="${CARGO_HOME_T2:-$CHECKOUT/.cargo-home}"
LOG="$RECEIPT_DIR/$RUN_ID.log"; VERDICT="$RECEIPT_DIR/$RUN_ID.verdict"; PIDFILE="$RECEIPT_DIR/$RUN_ID.pgid"
export TMPDIR

FAILS=0
fail() { echo "PREFLIGHT-FAIL: $*" >&2; FAILS=$((FAILS+1)); }

canon() { readlink -m -- "$1"; }
under() { case "$(canon "$1")/" in "$(canon "$2")"/*) return 0 ;; *) return 1 ;; esac; }
fs_free_gb() { df -P -BG -- "$1" | awk 'NR==2{gsub("G","",$4); print $4}'; }
fs_dev() { df -P -- "$1" | awk 'NR==2{print $1}'; }
swap_used_kb() { awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{print t-f}' /proc/meminfo; }

preflight() {
  local f l m s1 s2 g rc t
  if [ ! -d "$RAID_ROOT" ]; then fail "RAID_ROOT $RAID_ROOT missing (no raid volume)"; return; fi
  if [ "$(fs_dev "$RAID_ROOT")" = "$(fs_dev /)" ]; then fail "RAID_ROOT $RAID_ROOT is on the root filesystem, not a raid volume"; fi
  under "$CHECKOUT" "$RAID_ROOT" || fail "CHECKOUT $CHECKOUT not under RAID_ROOT"
  [ ! -e "$CHECKOUT" ] || fail "CHECKOUT $CHECKOUT already exists (must be fresh)"
  [ "$(canon "$REPO_ROOT")" = "$(canon "$CHECKOUT")" ] || fail "REPO_ROOT $REPO_ROOT != dedicated checkout $CHECKOUT"
  under "$CARGO_HOME" "$CHECKOUT" || fail "CARGO_HOME $CARGO_HOME not inside checkout"
  if under "$CARGO_HOME" "${HOME:-/nonexistent}/.cargo"; then fail "CARGO_HOME under ~/.cargo"; fi
  case ":$PATH:" in *":$HOME/.cargo/bin:"*) : ;; *) fail "~/.cargo/bin not on PATH" ;; esac
  under "$TMPDIR" "$RAID_ROOT" || fail "TMPDIR $TMPDIR not on raid"
  if under "$TMPDIR" "$CHECKOUT"; then fail "TMPDIR $TMPDIR inside checkout"; fi
  [ ! -e "$TMPDIR" ] || fail "TMPDIR $TMPDIR already exists"
  f="$(fs_free_gb "$RAID_ROOT")"
  [ "${f:-0}" -ge "$FLOOR_GB" ] || fail "free ${f}G < floor ${FLOOR_GB}G"
  l="${T2_LOAD1:-$(cut -d' ' -f1 /proc/loadavg)}"
  awk -v l="$l" -v m="$LOAD_MAX" 'BEGIN{exit !(l<m)}' || fail "load1 $l >= $LOAD_MAX"
  m="${T2_MEMAVAIL_KB:-$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)}"
  [ "${m:-0}" -gt "$MEM_MIN_KB" ] || fail "MemAvailable ${m}kB <= 64GiB"
  s1="$(swap_used_kb)"; sleep 2; s2="$(swap_used_kb)"
  [ $((s2 - s1 + ${T2_SWAP_GROWTH_KB:-0})) -le 0 ] || fail "swap use growing ($s1 -> $s2 kB)"
  rc=0
  if [ -n "${T2_NVSMI_CMD:-}" ]; then g="$(bash -c "$T2_NVSMI_CMD" 2>&1)" || rc=$?
  elif command -v nvidia-smi >/dev/null 2>&1; then
    g="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)" || rc=$?
  else g="nvidia-smi missing"; rc=127; fi
  if [ "$rc" -ne 0 ]; then fail "nvidia-smi failed rc=$rc: $g (C14 not_measured; refuse)"
  elif [ -n "$(printf '%s' "$g" | tr -d '[:space:]')" ]; then fail "GPU busy (compute apps: $(echo "$g" | tr '\n' ' '))"; fi
  for t in setsid nice ionice timeout df awk ps; do command -v "$t" >/dev/null 2>&1 || fail "missing tool $t"; done
}

preflight
if [ "$FAILS" -ne 0 ]; then echo "CHECK FAILED: $FAILS violation(s)" >&2; exit 1; fi
if [ "$MODE" = check ]; then echo "CHECK OK: all preflight assertions pass (nothing started)"; exit 0; fi

# ---- from here: real/test launch. Paths validated above, so cleanup is safe. ----
mkdir -p "$RECEIPT_DIR" || exit 1
{ echo "log=$LOG"; echo "verdict=$VERDICT"; echo "pgid_file=$PIDFILE"; echo "checkout=$CHECKOUT"; echo "tmpdir=$TMPDIR"; echo "sha=$SHA"; echo "version=$VERSION"; } > "$RECEIPT_DIR/$RUN_ID.receipt-path"
echo "RECEIPT: log=$LOG verdict=$VERDICT (index: $RECEIPT_DIR/$RUN_ID.receipt-path)"

PGID=""; WATCH_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [ -n "$WATCH_PID" ]; then kill "$WATCH_PID" 2>/dev/null || true; fi
  if [ -n "$PGID" ] && kill -0 -- "-$PGID" 2>/dev/null; then
    { kill -TERM -- "-$PGID" 2>/dev/null; sleep 2; kill -KILL -- "-$PGID" 2>/dev/null; } || true
  fi
  [ -s "$VERDICT" ] || echo "not_measured: wrapper exited rc=$rc before a verdict" > "$VERDICT"
  if [ -d "$CHECKOUT/target/release-verdicts" ]; then cp -r "$CHECKOUT/target/release-verdicts" "$RECEIPT_DIR/$RUN_ID.verdicts" 2>/dev/null || true; fi
  rm -rf -- "${RUN_DIR:?}" "${TMPDIR:?}" "${WORKLOG:?}"
  echo "CLEANUP: removed $RUN_DIR $TMPDIR $WORKLOG; verdict: $(cat "$VERDICT")"
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP
mkdir -p "$RUN_DIR" "$TMPDIR" "$WORKLOG" || exit 1

if [ "$MODE" = run ]; then
  : "${SRC_REPO:?SRC_REPO (existing aprender git repo on lambda to clone from) required for --run}"
  if ! { git clone --no-checkout -- "$SRC_REPO" "$CHECKOUT" && git -C "$CHECKOUT" checkout --detach "$SHA"; } >>"$LOG" 2>&1; then
    echo "fail: clone/checkout" > "$VERDICT"; exit 3
  fi
  [ "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$SHA" ] || { echo "fail: HEAD != $SHA" > "$VERDICT"; exit 3; }
  CMD=(scripts/release/t2_preflight.sh "$VERSION" --commit "$SHA")
else
  mkdir -p "$CHECKOUT"; CMD=("${TEST_CMD[@]}")
fi
mkdir -p "$CARGO_HOME"

# launch: setsid -> own session/PGID; nice+ionice; hard 6h KILL cap
(
  cd "$CHECKOUT" || exit 3
  export REPO_ROOT CARGO_HOME TMPDIR WORKLOG RELEASE_EPIC="$EPIC"
  export PATH="$HOME/.cargo/bin:$PATH"
  export CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 NEXTEST_TEST_THREADS=8
  exec setsid nice -n 19 ionice -c3 timeout -s KILL "${T2_CAP:-$CAP}" "${CMD[@]}" >>"$LOG" 2>&1
) &
CHILD=$!
sleep 1
PGID="$CHILD"   # the subshell is not a group leader, so setsid execs in place: pid == sid == pgid
[ -n "$PGID" ] || { echo "not_measured: could not record PGID" > "$VERDICT"; exit 4; }
echo "$PGID" > "$PIDFILE"
echo "started pgid=$PGID (pidfile $PIDFILE)"

# watcher: disk floor, swap growth, >STALL_S stopped (T). Never SIGCONT.
watcher() {
  local swap0 stopped_since=0 now free1 free2 st
  swap0="$(swap_used_kb)"
  while kill -0 -- "-$PGID" 2>/dev/null; do
    free1="$(fs_free_gb "$CHECKOUT")"; free2="$(fs_free_gb "$TMPDIR")"
    if [ "${free1:-0}" -lt "$WATCH_FLOOR_GB" ] || [ "${free2:-0}" -lt "$WATCH_FLOOR_GB" ]; then
      echo "killed: disk free ${free1}G/${free2}G < floor ${WATCH_FLOOR_GB}G (not_measured)" > "$VERDICT"
      { kill -TERM -- "-$PGID" 2>/dev/null; sleep 3; kill -KILL -- "-$PGID" 2>/dev/null; } || true; return
    fi
    if [ $(( $(swap_used_kb) - swap0 )) -gt $((2*1024*1024)) ]; then
      echo "killed: swap growth >2GiB (not_measured)" > "$VERDICT"
      { kill -TERM -- "-$PGID" 2>/dev/null; sleep 3; kill -KILL -- "-$PGID" 2>/dev/null; } || true; return
    fi
    st="$(ps -eo pgid=,stat= | awk -v g="$PGID" '$1==g && $2 ~ /^T/ {print "T"; exit}')"
    now="$(date +%s)"
    if [ "$st" = "T" ]; then
      [ "$stopped_since" -ne 0 ] || stopped_since="$now"
      if [ $((now - stopped_since)) -gt "$STALL_S" ]; then
        echo "not_measured: process group stopped (T) >${STALL_S}s; killed, no SIGCONT" > "$VERDICT"
        kill -KILL -- "-$PGID" 2>/dev/null || true; return
      fi
    else stopped_since=0; fi
    sleep "$POLL_S"
  done
}
watcher & WATCH_PID=$!

RC=0; wait "$CHILD" || RC=$?
if [ ! -s "$VERDICT" ]; then
  if [ "$RC" -eq 0 ]; then echo "completed rc=0 (GO/NO-GO is the VERDICT line in $LOG)" > "$VERDICT"
  elif [ "$RC" -eq 137 ]; then echo "not_measured: SIGKILL (6h cap or external kill) rc=137" > "$VERDICT"
  else echo "completed rc=$RC (see $LOG)" > "$VERDICT"; fi
fi
exit "$RC"
