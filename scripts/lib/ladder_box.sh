#!/usr/bin/env bash
# ladder_box.sh — run a ladder/sweep command in a hard resource box with an IO-PSI brake (#4520 step 1).
#
# Trigger: `model_ladder.sh --host lambda` re-opened a 17 GB GGUF with --no-gpu for every verb, read
# ~2.5 GB/s and held lambda (the operator's desktop) at IO PSI 90% / load 190 for 15+ minutes.
# Rule: a gate may make its host slower, never unusable.
#
# Usage:  bash scripts/lib/ladder_box.sh --events <file.jsonl> [--io-path <dir>]... -- <command> [args...]
#
# The command runs as a transient systemd --user unit with
#   IOWeight=10, CPUQuota, MemoryMax, and IOReadBandwidthMax on the block device of every --io-path
#   (default: $HOME/models and the current directory).
# While it runs, the host's IO pressure (`some avg10` of $LADDER_PSI_FILE, default /proc/pressure/io)
# is sampled every $LADDER_PSI_PERIOD s: above $LADDER_PSI_STOP % the unit is sent SIGSTOP, and once
# it falls below $LADDER_PSI_CONT % it is sent SIGCONT. Every brake event and a PSI trace sample
# every $LADDER_PSI_TRACE_EVERY periods go to --events as one JSON object per line.
# The environment is passed through, plus LADDER_BOXED=1 so the ladder can refuse an unboxed run.
#
# Exit: the command's own exit status · 2 usage / systemd-run unavailable (nothing ran).
#
# Knobs (env): LADDER_IO_READ_MAX (200M) LADDER_MEM_MAX (half of MemTotal) LADDER_CPU_QUOTA (800%)
#              LADDER_PSI_FILE LADDER_PSI_STOP (30) LADDER_PSI_CONT (15) LADDER_PSI_PERIOD (2)
#              LADDER_PSI_TRACE_EVERY (15)
set -uo pipefail

EVENTS="" IO_PATHS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --events)  [ $# -ge 2 ] || { echo "ladder_box: --events needs a file" >&2; exit 2; }; EVENTS="$2"; shift 2 ;;
    --io-path) [ $# -ge 2 ] || { echo "ladder_box: --io-path needs a dir" >&2; exit 2; }; IO_PATHS+=("$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "ladder_box: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$EVENTS" ] || { echo "ladder_box: --events is required" >&2; exit 2; }
[ $# -gt 0 ] || { echo "ladder_box: no command after --" >&2; exit 2; }
command -v systemd-run >/dev/null 2>&1 || { echo "ladder_box: systemd-run not found — refusing to run unboxed" >&2; exit 2; }
[ ${#IO_PATHS[@]} -gt 0 ] || IO_PATHS=("$HOME/models" "$PWD")

PSI_FILE="${LADDER_PSI_FILE:-/proc/pressure/io}"
PSI_STOP="${LADDER_PSI_STOP:-30}"
PSI_CONT="${LADDER_PSI_CONT:-15}"
PERIOD="${LADDER_PSI_PERIOD:-2}"
TRACE_EVERY="${LADDER_PSI_TRACE_EVERY:-15}"
[ -r "$PSI_FILE" ] || { echo "ladder_box: PSI source $PSI_FILE unreadable — the brake would be blind, refusing" >&2; exit 2; }

props=(-p IOWeight=10 -p "CPUQuota=${LADDER_CPU_QUOTA:-800%}" -p "MemoryMax=${LADDER_MEM_MAX:-$(awk '/^MemTotal:/{printf "%dK", $2/2}' /proc/meminfo)}")
declare -A seen=()
for p in "${IO_PATHS[@]}"; do
  [ -e "$p" ] || continue
  dev=$(df --output=source "$p" 2>/dev/null | tail -n1)
  case "$dev" in /dev/*) ;; *) continue ;; esac
  [ -n "${seen[$dev]:-}" ] && continue
  seen[$dev]=1
  props+=(-p "IOReadBandwidthMax=$dev ${LADDER_IO_READ_MAX:-200M}")
done
[ ${#seen[@]} -gt 0 ] || { echo "ladder_box: no block device resolved from --io-path — IO read cap would be absent, refusing" >&2; exit 2; }

# the caller's exported environment as NAME=value (systemd 249 -E does not copy a bare NAME); never parsed from `env`
pass_env=() skipped_env=()
while IFS= read -r name; do
  case "$name" in LADDER_BOXED|_|SHLVL|PWD|OLDPWD) continue ;; esac
  # systemd refuses the whole block over one value with a control character (a multi-line var): skip it, by name
  if [[ "${!name}" == *[[:cntrl:]]* ]]; then skipped_env+=("$name"); continue; fi
  pass_env+=(-E "$name=${!name}")
done < <(compgen -e)

UNIT="ladder-box-$(id -u)-$$"
mkdir -p "$(dirname "$EVENTS")"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# some avg10 as an integer percent×100 (bash has no floats); empty on a malformed file
psi_some() { awk '$1=="some"{for(i=2;i<=NF;i++) if($i ~ /^avg10=/){sub(/^avg10=/,"",$i); printf "%d", $i*100; exit}}' "$PSI_FILE" 2>/dev/null; }
event() { printf '{"t":"%s","unit":"%s","event":"%s","psi_some_avg10":%s%s}\n' "$(now)" "$UNIT" "$1" "$2" "${3:-}" >> "$EVENTS"; }

devs_json=$(printf '"%s",' "${!seen[@]}"); devs_json="[${devs_json%,}]"
# Which caps can engage: a controller not delegated to the user manager is ACCEPTED by systemd-run and silently
# not enforced (gx10 delegates cpu memory pids, no io — measured 2026-09-27). Say so in the events; never claim it.
ctl=$(cat "/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.subtree_control" 2>/dev/null)
enforced=$(for c in io memory cpu; do case " $ctl " in *" $c "*) printf '"%s",' "$c" ;; esac; done); enforced="[${enforced%,}]"
case " $ctl " in *" io "*) ;; *) echo "ladder_box: io controller not delegated here — IOReadBandwidthMax/IOWeight are NOT enforced; the PSI brake is the only IO bound" >&2 ;; esac
skip_json=$(printf '"%s",' "${skipped_env[@]}"); skip_json="[${skip_json%,}]"; [ "$skip_json" = '[""]' ] && skip_json='[]'
event start null ",\"io_read_max\":\"${LADDER_IO_READ_MAX:-200M}\",\"devices\":$devs_json,\"psi_stop\":$PSI_STOP,\"psi_cont\":$PSI_CONT,\"psi_file\":\"$PSI_FILE\",\"env_skipped\":$skip_json,\"controllers_enforced\":$enforced"

systemd-run --user --quiet --pipe --wait --collect --same-dir --unit="$UNIT" \
  -E LADDER_BOXED=1 "${pass_env[@]}" \
  "${props[@]}" -- "$@" &
RUN_PID=$!

# A box killed mid-run must not leave its unit frozen or orphaned: thaw, stop, record.
abort() { systemctl --user kill --signal=SIGCONT "$UNIT" 2>/dev/null; systemctl --user stop "$UNIT" 2>/dev/null; event aborted null; exit 143; }
trap abort TERM INT HUP

stopped=0 n=0 stops=0
while kill -0 "$RUN_PID" 2>/dev/null; do
  sleep "$PERIOD"
  v=$(psi_some)
  if [ -z "$v" ]; then event psi_unreadable null; continue; fi
  pct=$(awk -v v="$v" 'BEGIN{printf "%.2f", v/100}')
  if [ "$stopped" -eq 0 ] && [ "$v" -gt $((PSI_STOP * 100)) ]; then
    systemctl --user kill --signal=SIGSTOP "$UNIT" 2>/dev/null && { stopped=1; stops=$((stops + 1)); event SIGSTOP "$pct"; }
  elif [ "$stopped" -eq 1 ] && [ "$v" -lt $((PSI_CONT * 100)) ]; then
    systemctl --user kill --signal=SIGCONT "$UNIT" 2>/dev/null && { stopped=0; event SIGCONT "$pct"; }
  fi
  n=$((n + 1))
  [ $((n % TRACE_EVERY)) -eq 0 ] && event trace "$pct" ",\"stopped\":$stopped"
done
wait "$RUN_PID"; rc=$?
event exit null ",\"rc\":$rc,\"brake_stops\":$stops"
exit "$rc"
