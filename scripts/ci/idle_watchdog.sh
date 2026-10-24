#!/usr/bin/env bash
# idle_watchdog.sh DIR -- CMD [ARG...]
# Runs CMD; kills its process group (TERM, then KILL after FAT_IDLE_GRACE seconds) once no DIR/*.log
# has grown for FAT_IDLE_LIMIT_S seconds, and exits 124. A step that writes nothing for that long is
# hung, not slow. FAT_IDLE_LIMIT_S unset, 0 or not a plain decimal = off: CMD is exec'd untouched.
# FAT_IDLE_TICK (default 5) is the poll period in seconds. The merge-queue shard sets 1200 around
# fat_driver.py run, whose SIGTERM handler reaps every step group and container (fat_driver.py on_signal).
# Replaces the fat_driver.wait_or_idle Python watchdog (#6285); the limit parses as a plain decimal.
#   --limit-us   print the parsed limit in microseconds (0 = off)
#   --self-test  parity rows: stalled log fires, growing log never, a timeout still works, parsing
set -u

limit_us() {
    local v=${FAT_IDLE_LIMIT_S:-0} ip fp
    [[ $v =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo 0; return; }
    ip=${v%%.*}
    fp=0
    [[ $v == *.* ]] && fp=${v#*.}
    fp=${fp}000000
    ip=$((10#$ip))
    fp=$((10#${fp:0:6}))
    echo $((ip * 1000000 + fp))
}

logs_size() {
    local t=0 s f
    for f in "$1"/*.log; do
        [ -f "$f" ] || continue
        s=$(stat -c %s "$f") || continue
        t=$((t + s))
    done
    echo "$t"
}

watch_cmd() {
    local dir=$1 limit=$2 tick=${FAT_IDLE_TICK:-5} grace=${FAT_IDLE_GRACE:-150}
    shift 2
    local pid sp who rc size last now since
    set -m # the child leads its own process group, so one signal reaches all of it
    "$@" &
    pid=$!
    trap 'kill -TERM -- "-$pid" 2>/dev/null' TERM INT
    last=$(logs_size "$dir")
    since=${EPOCHREALTIME/./}
    while :; do
        sleep "$tick" &
        sp=$!
        wait -n -p who "$pid" "$sp"
        rc=$?
        if [ "${who:-}" = "$pid" ]; then
            kill "$sp" 2>/dev/null
            exit "$rc"
        fi
        if [ -z "${who:-}" ] && ! kill -0 "$pid" 2>/dev/null; then # child already reaped: never spin on a failing wait
            kill "$sp" 2>/dev/null
            exit 125
        fi
        size=$(logs_size "$dir")
        now=${EPOCHREALTIME/./}
        if [ "$size" != "$last" ]; then
            last=$size
            since=$now
        elif [ $((now - since)) -ge "$limit" ]; then
            echo "::error::step printed nothing for $(((now - since) / 1000000))s (FAT_IDLE_LIMIT_S): a hang, not a slow run; killing group $pid"
            kill -TERM -- "-$pid" 2>/dev/null
            sleep "$grace" &
            wait -n -p who "$pid" $!
            [ "${who:-}" = "$pid" ] || kill -KILL -- "-$pid" 2>/dev/null
            exit 124
        fi
    done
}

self_test() {
    local me bad=0 rc t0 out
    me=$(readlink -f "$0")
    td=$(mktemp -d) || exit 2 # global: the EXIT trap reads it after this function returns
    trap 'rm -rf -- "${td:?}"' EXIT
    row() { # LABEL GOT WANT
        if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2' want '$3'"; bad=$((bad + 1)); fi
    }
    export FAT_IDLE_TICK=0.2 FAT_IDLE_GRACE=1

    t0=$SECONDS
    out=$(FAT_IDLE_LIMIT_S=1 bash "$me" "$td" -- sleep 30.3 2>&1); rc=$?
    row "idle: the hung step is killed promptly, not left to finish" "$((SECONDS - t0 < 10 ? 1 : 0))" 1
    sleep 0.3
    row "idle: the hung step's group is gone" "$(pgrep -fx 'sleep 30[.]3' >/dev/null && echo alive || echo gone)" gone
    row "idle: a step that prints nothing past FAT_IDLE_LIMIT_S is declared hung" "$rc" 124
    row "idle: the hang is named" "$([[ $out == *'printed nothing'* ]] && echo yes)" yes

    out=$(FAT_IDLE_LIMIT_S=1 bash "$me" "$td" -- sh -c 'for i in 1 2 3 4 5 6; do echo tick >>"$0/s.log"; sleep 0.4; done' "$td" 2>&1); rc=$?
    row "idle: a step that keeps printing is never declared hung" "$rc" 0
    row "idle: the exit status of the command passes through" "$(FAT_IDLE_LIMIT_S=1 bash "$me" "$td" -- sh -c 'exit 7'; echo $?)" 7

    t0=$SECONDS
    FAT_IDLE_LIMIT_S=100 timeout 2 bash "$me" "$td" -- sleep 31.7; rc=$?
    row "idle: the timeout still ends the run alongside the watchdog" "$rc" 124
    sleep 0.5
    row "idle: the timeout leaves no orphan behind" "$(pgrep -fx 'sleep 31[.]7' >/dev/null && echo orphan || echo none)" none
    row "idle: the timeout fired at its own deadline, not the idle one" "$((SECONDS - t0 < 8 ? 1 : 0))" 1

    row "idle: unset means off" "$(env -u FAT_IDLE_LIMIT_S bash "$me" --limit-us)" 0
    row "idle: FAT_IDLE_LIMIT_S=1500 is 25 min" "$(FAT_IDLE_LIMIT_S=1500 bash "$me" --limit-us)" 1500000000
    row "idle: a decimal limit parses" "$(FAT_IDLE_LIMIT_S=1.5 bash "$me" --limit-us)" 1500000
    row "idle: garbage means off" "$(FAT_IDLE_LIMIT_S=abc bash "$me" --limit-us)" 0
    row "idle: a negative limit means off" "$(FAT_IDLE_LIMIT_S=-5 bash "$me" --limit-us)" 0
    row "idle: off runs the command untouched" "$(FAT_IDLE_LIMIT_S=0 bash "$me" "$td" -- sh -c 'sleep 0.7; exit 3'; echo $?)" 3

    if [ "$bad" -ne 0 ]; then echo "idle_watchdog self-test: $bad row(s) failed"; exit 1; fi
    echo "idle_watchdog self-test: all rows pass"
}

case "${1:-}" in
--self-test) self_test; exit ;;
--limit-us) limit_us; exit ;;
esac
dir=${1:?usage: idle_watchdog.sh DIR -- CMD [ARG...]}
shift
[ "${1:-}" = -- ] && shift
[ "$#" -gt 0 ] || { echo "idle_watchdog.sh: no command" >&2; exit 2; }
us=$(limit_us)
[ "$us" -gt 0 ] || exec "$@"
watch_cmd "$dir" "$us" "$@"
