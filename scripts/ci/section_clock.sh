#!/bin/bash
# section_clock.sh - run one CI section step under a clock that its neighbours cannot move.
#
# usage: section_clock.sh [options] -- <command> [args...]
#   --label L          name printed in the verdict lines (default: section)
#   --hang-min M       a hang is M minutes with no new output (default 10)
#   --hang-s S         the same in seconds (case tables)
#   --poll-s S         how often the output size is checked (default 15)
#   --grace-s S        TERM to KILL grace on a hang (default 15)
#   --cpu-idle-s S     CPU-seconds of one idle run of this section; the rule E budget is S x 1.3
#   --cpu-mode M       report (default) prints the rule E verdict; enforce fails the step on it
#   --meter M          rusage (default): user+sys of the command tree, which a neighbour cannot
#                      inflate; cgroup: usage_usec of this process's cgroup v2, which also counts a
#                      server outside the tree but counts every section that shares the cgroup
#   --cgroup-root DIR  where cgroup v2 is mounted (default /sys/fs/cgroup)
#   --slot-dir DIR     the build-slot lock dir (default /run/lock/fleet-build/v1)
#
# The clock starts when the heavy-build slot is held: the step must run under
# `build_slot.sh run -- ...`, which exports FLEET_BUILD_SLOT_T0 (slot held), FLEET_BUILD_SLOT_WAIT_S,
# FLEET_BUILD_SLOT and FLEET_BUILD_SLOT_PID. The step is refused (rc 2), never run unclocked, unless
# that pid is an ancestor named by the owner file of a slot that is locked now.
# Waiting for a slot is never counted against a section. Once it runs, two checks replace the wall
# deadline: a hang is no output for the hang time (killpg TERM, grace, KILL, rc 124); and rule E
# judges the CPU-seconds the command used against its budget. A meter that cannot be opened is
# cpu not_measured: printed in report mode, and in enforce mode the step fails (rc 3), never a pass.
#
# exit: the command's own rc; 124 hang; 1 rule E over budget (enforce); 3 cpu not_measured
# (enforce); 2 usage or no slot. The last line is `section_clock: {json}` (wait_s, cpu_s, meter,
# budget_s, hang, rc); with GITHUB_OUTPUT set, wait_s/cpu_s/meter/hang are step outputs too.
set -euo pipefail

label=section
hang_s=600
poll_s=15
grace_s=15
idle_s=''
cpu_mode=report
meter=rusage
cg_root=/sys/fs/cgroup
slot_dir=/run/lock/fleet-build/v1

usage() {
    sed -n '3,16p' "$0" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --label) label="${2:?}"; shift 2 ;;
        --hang-min) hang_s=$(( ${2:?} * 60 )); shift 2 ;;
        --hang-s) hang_s="${2:?}"; shift 2 ;;
        --poll-s) poll_s="${2:?}"; shift 2 ;;
        --grace-s) grace_s="${2:?}"; shift 2 ;;
        --cpu-idle-s) idle_s="${2:?}"; shift 2 ;;
        --cpu-mode) cpu_mode="${2:?}"; shift 2 ;;
        --meter) meter="${2:?}"; shift 2 ;;
        --cgroup-root) cg_root="${2:?}"; shift 2 ;;
        --slot-dir) slot_dir="${2:?}"; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[ $# -gt 0 ] || usage
case "$cpu_mode" in report|enforce) ;; *) usage ;; esac
case "$meter" in rusage|cgroup) ;; *) usage ;; esac

# in_slot: the slot proof of build_slot.sh's re-entrant rule. FLEET_BUILD_SLOT_PID is this process
# or an ancestor, slot.<FLEET_BUILD_SLOT>.owner names that pid, and the slot is locked right now.
# An env var leaked to an unrelated process, a stale owner line or a forged pid proves nothing.
in_slot() {
    local want="${FLEET_BUILD_SLOT_PID:-}" slot="${FLEET_BUILD_SLOT:-}" p="$$" owner
    case "$want" in '' | *[!0-9]*) return 1 ;; esac
    case "$slot" in '' | *[!0-9]*) return 1 ;; esac
    while [ "$p" != "$want" ]; do
        p=$(awk '$1 == "PPid:" { print $2 }' "/proc/$p/status" 2> /dev/null) || return 1
        case "$p" in '' | 0) return 1 ;; esac
    done
    owner=$(awk 'NR == 1 { print $1 }' "$slot_dir/slot.$slot.owner" 2> /dev/null) || return 1
    [ "$owner" = "$want" ] || return 1
    [ -e "$slot_dir/slot.$slot" ] || return 1
    ! flock -n "$slot_dir/slot.$slot" true 2> /dev/null
}

if [ -z "${FLEET_BUILD_SLOT_T0:-}" ] || ! in_slot; then
    printf '::error::section_clock: %s holds no build slot (FLEET_BUILD_SLOT_T0 and a live FLEET_BUILD_SLOT_PID owner); run it under build_slot.sh run, never unclocked\n' "$label" >&2
    exit 2
fi
wait_s="${FLEET_BUILD_SLOT_WAIT_S:-not_measured}"

# cg_usage: usage_usec of this process's cgroup v2, or nothing when it cannot be opened
cg_usage() {
    local rel
    rel=$(sed -n 's/^0:://p' /proc/self/cgroup 2> /dev/null) || return 0
    [ -n "$rel" ] || return 0
    awk '$1 == "usage_usec" { print $2 }' "$cg_root$rel/cpu.stat" 2> /dev/null || true
}

work=$(mktemp -d)
trap 'rm -rf -- "${work:?}"' EXIT
out="$work/out"
: > "$out"
cg0=''
if [ "$meter" = cgroup ]; then cg0=$(cg_usage); fi

# The command runs in its own session (so a hang kill takes its whole process group) under an
# inner bash whose `times` gives the user+sys of the command tree, and of nothing else.
setsid bash -c '"$@"; rc=$?; times > "$0.times"; exit "$rc"' "$out" "$@" > "$out" 2>&1 < /dev/null &
cpid=$!
tail -n +1 -F -s 0.2 --pid="$cpid" "$out" 2> /dev/null &
tpid=$!

hang=false
size=0
last="${EPOCHREALTIME/./}"
while kill -0 "$cpid" 2> /dev/null; do
    sleep "$poll_s"
    now="${EPOCHREALTIME/./}"
    cur=$(stat -c %s "$out")
    idle=$(( (now - last) / 1000000 ))
    if [ "$cur" != "$size" ]; then
        size="$cur"
        last="$now"
    elif [ "$idle" -ge "$hang_s" ] && kill -0 "$cpid" 2> /dev/null; then
        hang=true
        kill -TERM -- "-$cpid" 2> /dev/null || true
        t=0
        while kill -0 "$cpid" 2> /dev/null && [ "$t" -lt "$grace_s" ]; do
            sleep 1
            t=$((t + 1))
        done
        kill -KILL -- "-$cpid" 2> /dev/null || true
        break
    fi
done
rc=0
wait "$cpid" || rc=$?
wait "$tpid" 2> /dev/null || true

# cpu_s: the meter's CPU-seconds, or not_measured
cpu_s=not_measured
if [ "$hang" = false ]; then
    if [ "$meter" = rusage ] && [ -s "$out.times" ]; then
        cpu_s=$(awk 'NR == 2 {
            n = 0
            for (i = 1; i <= 2; i++) { split($i, p, "m"); sub("s", "", p[2]); n += p[1] * 60 + p[2] }
            printf "%.3f", n }' "$out.times")
    elif [ "$meter" = cgroup ] && [ -n "$cg0" ]; then
        cg1=$(cg_usage)
        if [ -n "$cg1" ]; then cpu_s=$(awk -v a="$cg0" -v b="$cg1" 'BEGIN { printf "%.3f", (b - a) / 1e6 }'); fi
    fi
fi
budget_s=none
if [ -n "$idle_s" ]; then budget_s=$(awk -v s="$idle_s" 'BEGIN { printf "%.3f", s * 1.3 }'); fi

if [ "$hang" = true ]; then
    printf '::error::section_clock: %s gave no output for %s s (hang); killed its process group\n' "$label" "$hang_s"
    rc=124
elif [ "$budget_s" != none ]; then
    if [ "$cpu_s" = not_measured ]; then
        if [ "$cpu_mode" = enforce ]; then
            printf '::error::rule E: %s cpu not_measured (meter %s) - never a pass\n' "$label" "$meter"
            if [ "$rc" -eq 0 ]; then rc=3; fi
        else
            printf 'section_clock: rule E report-only: %s cpu not_measured (meter %s)\n' "$label" "$meter"
        fi
    elif awk -v c="$cpu_s" -v b="$budget_s" 'BEGIN { exit !(c > b) }'; then
        if [ "$cpu_mode" = enforce ]; then
            printf '::error::rule E: %s used %s s CPU > budget %s s\n' "$label" "$cpu_s" "$budget_s"
            if [ "$rc" -eq 0 ]; then rc=1; fi
        else
            printf 'section_clock: rule E report-only: %s used %s s CPU > budget %s s\n' "$label" "$cpu_s" "$budget_s"
        fi
    else
        printf 'section_clock: rule E: %s used %s s CPU <= budget %s s\n' "$label" "$cpu_s" "$budget_s"
    fi
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
    printf 'wait_s=%s\ncpu_s=%s\nmeter=%s\nhang=%s\n' "$wait_s" "$cpu_s" "$meter" "$hang" >> "$GITHUB_OUTPUT"
fi
printf 'section_clock: {"label":"%s","wait_s":"%s","cpu_s":"%s","meter":"%s","budget_s":"%s","hang":%s,"rc":%s}\n' \
    "$label" "$wait_s" "$cpu_s" "$meter" "$budget_s" "$hang" "$rc"
exit "$rc"
