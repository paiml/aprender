#!/usr/bin/env bash
# crux_cell_teardown.sh: stop every server a CRUX cell started, and PROVE it is gone
# from the process table AND from the GPU, before the cell exits and the GPU lock drops.
#
# WHY. A llama-server from a CRUX cell was still resident on the 4090 after its cell
# released /tmp/apr-gpu.lock. The next lock holder, a prio-1 measurement that took the
# lock properly, was refused as CONTENDED (cop, 2026-09-23). `kill` + `wait` in the
# cell was not enough: `wait` only covers children of the SAME shell, a timeout can
# skip the kill line entirely, and a process that has exited can still hold device
# memory until the driver reaps its context. So this runs from the cell's
# EXIT/TERM/INT trap and checks both properties it owes the next holder:
#   1. every pid named in the pid files is gone: TERM, then up to 15 s, then KILL,
#      then up to 10 s more;
#   2. none of those pids is listed by `nvidia-smi --query-compute-apps` (polled for
#      up to 30 s). With no nvidia-smi, the host has no GPU to leak.
#
# Usage: crux_cell_teardown.sh <state file> <pid file>...   (a pid file holds one pid per line)
#   writes `clean`, or `FAILED: <why>`, to <state file>. The producer reads that file
#   AFTER the cell, and anything but `clean` makes every row of the cell RED.
# Env: CRUX_NVIDIA_SMI overrides the nvidia-smi binary, CRUX_TEARDOWN_GPU_POLLS the
#      number of 0.5 s polls (default 60), CRUX_TEARDOWN_KILL the command that sends the
#      TERM/KILL (default: the `kill` builtin); all three exist for the case table.
#
# PID 1 IS REFUSED BY NAME (#4120). A pid file naming 1 is never signalled: it FAILS the
# cell. On a dev box `kill 1` is EPERM and looks harmless, but inside a CI runner's
# container pid 1 IS the runner (Runner.Listener): the case table's own T2 row used
# init as its "unkillable server", and its `kill -TERM 1` / `kill -KILL 1` shut down the
# gx10/yoga runners mid guard-tree job (runs 35900354071, 35913559755, 35935643646).
# Exit: 0 clean · 1 FAILED · 2 usage.
set -u

[ $# -ge 1 ] || { printf 'usage: %s <state file> <pid file>...\n' "$0" >&2; exit 2; }
state="$1"
shift
pids=()
refused=""
for f in "$@"; do
  [ -f "$f" ] || continue
  while IFS= read -r p; do
    # Only a real pid. `0` would signal this whole PROCESS GROUP, the calling cell
    # and whatever runs it, and a leading zero is not a pid either. `1` is init, or in
    # a container the runner itself (#4120): refused by name, and the cell FAILS.
    # An empty line is no pid. Anything else that is not a plain pid (`0`, a leading
    # zero, `1\r`, `12 34`, `+1`) is a pid file this script cannot trust: skipping it
    # silently reported `clean` over a live server (review lane A, #4120), so it FAILS
    # the cell like pid 1 does.
    case "$p" in '') continue ;; esac
    case "$p" in *[!0-9]*|0*) refused="$refused$f "; continue ;; esac
    case "$p" in 1) refused="$refused$f "; continue ;; esac
    pids+=("$p")
  done < "$f"
done
# Liveness from /proc, NOT `kill -0`: kill -0 FAILS (EPERM) on a process this user
# cannot signal, which reads a live server as gone (the T2 row caught exactly that).
# A zombie counts as gone: it has released its memory and its device context.
alive() {
  local p out=() st
  for p in "$@"; do
    if [ -d "/proc/$p" ]; then
      st=$(sed -n 's/^[0-9]* (.*) \([A-Za-z]\).*/\1/p' "/proc/$p/stat" 2> /dev/null)
      [ "$st" = Z ] && continue
      out+=("$p")
    elif kill -0 "$p" 2> /dev/null; then
      out+=("$p")
    fi
  done
  printf '%s ' "${out[@]}"
}

# The one place a TERM or KILL leaves this script (the case table swaps it for a logging
# no-op). alive() below probes with signal 0, which delivers nothing.
KILL_SEAM="${CRUX_TEARDOWN_KILL:-}"
sig() {
  if [ -n "$KILL_SEAM" ]; then "$KILL_SEAM" "$@"; else kill "$@"; fi
}

left=""
if [ "${#pids[@]}" -gt 0 ]; then
  sig -TERM "${pids[@]}" 2> /dev/null
  for _ in $(seq 1 30); do
    left=$(alive "${pids[@]}")
    [ -z "${left// /}" ] && break
    sleep 0.5
  done
  if [ -n "${left// /}" ]; then
    # shellcheck disable=SC2086
    sig -KILL $left 2> /dev/null
    for _ in $(seq 1 20); do
      left=$(alive "${pids[@]}")
      [ -z "${left// /}" ] && break
      sleep 0.5
    done
  fi
fi
if [ -n "$refused" ]; then
  printf 'FAILED: pid file(s) %sname pid 1 (init; in a CI container, the runner) or a malformed pid -- refused, never signalled\n' "$refused" > "$state"
  exit 1
fi
if [ -n "${left// /}" ]; then
  printf 'FAILED: server pid(s) %ssurvived TERM and KILL\n' "$left" > "$state"
  exit 1
fi

SMI="${CRUX_NVIDIA_SMI:-nvidia-smi}"
if [ "${#pids[@]}" -gt 0 ] && command -v "$SMI" > /dev/null 2>&1; then
  on_gpu=""
  for _ in $(seq 1 "${CRUX_TEARDOWN_GPU_POLLS:-60}"); do
    apps=$("$SMI" --query-compute-apps=pid --format=csv,noheader 2> /dev/null | tr -d ' ')
    on_gpu=""
    for p in "${pids[@]}"; do
      printf '%s\n' "$apps" | grep -qx "$p" && on_gpu="$on_gpu$p "
    done
    [ -z "$on_gpu" ] && break
    sleep 0.5
  done
  if [ -n "$on_gpu" ]; then
    printf 'FAILED: pid(s) %sstill listed by nvidia-smi compute-apps 30 s after exit\n' "$on_gpu" > "$state"
    exit 1
  fi
fi
printf 'clean\n' > "$state"
exit 0
