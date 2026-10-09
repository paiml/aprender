#!/usr/bin/env bash
# fat_teardown.sh - remove the section containers a fat_driver job left behind (#4948).
#
# fat_driver.py starts a section's job container as `fat-<run_id>-<section>`, on
# `tail -f /dev/null`, and removes it in the section's `finally`. A CANCELLED run
# never gets there: the runner signals the step, then kills it, and the forked
# background driver (`--background-until`) may still be mid-section. Containers
# were found on a self-hosted runner 1 to 4 days old, holding 11.5G, with
# nothing left that would ever stop them.
#
# So every job that runs fat_driver ends with an `if: always()` step that calls
# this, which GitHub runs on success, failure AND cancel. On the passing path the
# driver has already removed every container, so this finds none and changes
# nothing.
#
# WHICH CONTAINERS. Only this job's: the name starts `fat-<run_id>-` AND one of
# its mounts lies under this job's RUNNER_TEMP. The run id alone is not enough:
# the shards of one matrix job share a run id and can run side by side on one
# host, each runner instance with its own _temp.
#
# ORDER. The background driver is stopped first, by the pid it recorded, and
# only if that pid is still a fat_driver process. Otherwise it could start the
# next section's container after the sweep.
#
# USAGE
#   fat_teardown.sh <run_id> <runner_temp>
#   fat_teardown.sh --self-test        (needs docker and a local image)
#
# EXIT
#   0  no container of this job is left
#   1  one is still there after `docker rm -f`, or bad arguments
#   2  the self-test cannot run here (no docker, or no local image)

set -uo pipefail

PROG=${0##*/}
HERE=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

# Names of the containers that belong to (run_id, runner_temp), one per line.
own_containers() {
    local run=$1 rt=$2 n src
    rt=${rt%/}
    docker ps -a --filter "name=^fat-${run}-" --format '{{.Names}}' 2>/dev/null |
    while IFS= read -r n; do
        # The filter is a regex search; re-check the exact prefix, so run 12
        # never matches fat-123-...
        case "$n" in "fat-${run}-"*) ;; *) continue ;; esac
        while IFS= read -r src; do
            case "$src" in "$rt"/*) printf '%s\n' "$n"; break ;; esac
        done < <(docker inspect --format '{{range .Mounts}}{{println .Source}}{{end}}' "$n" 2>/dev/null)
    done
}

stop_driver() {
    local rt=$1 pidf pid i
    pidf="${rt%/}/fat/driver.pid"
    [ -f "$pidf" ] || return 0
    pid=$(cat "$pidf" 2>/dev/null)
    case "$pid" in ''|*[!0-9]*) return 0 ;; esac
    # A recorded pid can be reused once its process is gone: kill it only if it
    # is still fat_driver.
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'fat_driver.py' || return 0
    echo "$PROG: stopping the background driver, pid $pid"
    kill -TERM "$pid" 2>/dev/null
    for i in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    echo "$PROG: pid $pid still alive after 30 s; SIGKILL"
    kill -KILL "$pid" 2>/dev/null
    return 0
}

teardown() {
    local run=$1 rt=$2 c n=0 left
    # No docker on this runner: the driver could not have started a container.
    command -v docker >/dev/null 2>&1 || { echo "$PROG: no docker on this runner, so no section container to remove"; exit 0; }
    case "$run" in ''|*[!0-9A-Za-z_-]*) echo "$PROG: FAIL - bad run id '$run'"; exit 1 ;; esac
    case "$rt" in /?*) ;; *) echo "$PROG: FAIL - runner temp must be an absolute path, got '$rt'"; exit 1 ;; esac
    stop_driver "$rt"
    while IFS= read -r c; do
        [ -n "$c" ] || continue
        # The container ran as root: hand its files back to the runner user
        # first, as fat_driver's stop_container does, or the runner's own _temp
        # cleanup meets root-owned files.
        docker exec "$c" chown -R "$(id -u):$(id -g)" "${rt%/}/fat" >/dev/null 2>&1
        docker rm -f "$c" >/dev/null 2>&1
        echo "$PROG: removed $c"
        n=$((n + 1))
    done < <(own_containers "$run" "$rt")
    left=$(own_containers "$run" "$rt")
    if [ -n "$left" ]; then
        echo "$PROG: FAIL - still present after docker rm -f: $(printf "%s" "$left" | tr "\n" " ")"
        exit 1
    fi
    echo "$PROG: $n container(s) removed; none of run $run under $rt is left"
    exit 0
}

# ---------------------------------------------------------------------------
# --self-test: real containers, so the docker filters themselves are tested.
# ---------------------------------------------------------------------------
self_test() {
    local img=${FAT_TEARDOWN_TEST_IMAGE:-debian:trixie-slim} td run fails=0 ok=0 rc out
    command -v docker >/dev/null 2>&1 || { echo "$PROG: ENV - docker is not on PATH"; exit 2; }
    docker image inspect "$img" >/dev/null 2>&1 || { echo "$PROG: ENV - image $img is not present"; exit 2; }
    td=$(mktemp -d -t fat-teardown-st.XXXXXX) || exit 2
    run="st$$"
    mkdir -p "$td/rt1/fat/s" "$td/rt10/fat/s" "$td/rt2/fat/s"
    start() { # name mount-dir
        docker run -d --name "$1" --entrypoint tail -v "$2:$2" "$img" -f /dev/null >/dev/null
    }
    gone()  { ! docker inspect "$1" >/dev/null 2>&1; }
    running() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null); case "$s" in ""|Z*) echo no ;; *) echo yes ;; esac; }
    want()  { # desc expected actual
        if [ "$2" = "$3" ]; then echo "ok   $1"; ok=$((ok + 1))
        else echo "FAIL $1 (want '$2', got '$3')"; fails=$((fails + 1)); fi
    }
    cleanup_st() {
        # Every container the self-test starts, -d included: a run cut short
        # (an outage, a kill) must not leave one behind (it did, 2026-10-09).
        docker rm -f "fat-${run}-a" "fat-${run}-b" "fat-${run}-c" "fat-${run}-d" "fat-${run}1-a" "fat-x${run}-a" >/dev/null 2>&1
        rm -rf -- "${td:?}"
    }
    trap cleanup_st EXIT

    echo "== $PROG --self-test (image $img) =="
    start "fat-${run}-a"  "$td/rt1/fat/s"    # this run, this runner: removed
    start "fat-${run}-b"  "$td/rt1/fat/s"    # a second one of this job: removed
    start "fat-${run}-c"  "$td/rt2/fat/s"    # same run, another runner (a shard): kept
    start "fat-${run}1-a" "$td/rt1/fat/s"    # another run whose id extends this one: kept
    start "fat-x${run}-a" "$td/rt1/fat/s"    # the run id inside, not at the start: kept
    docker run -d --name "fat-${run}-d" --entrypoint tail -v "$td/rt10/fat/s:$td/rt10/fat/s" "$img" -f /dev/null >/dev/null
    # fat-<run>-d is under rt10, which only shares a string prefix with rt1: kept

    out=$(bash "$HERE/$PROG" "$run" "$td/rt1" 2>&1); rc=$?
    want "exit 0 when this job's containers are gone" 0 "$rc"
    want "this job first container is removed"  yes "$(gone "fat-${run}-a" && echo yes || echo no)"
    want "this job second container is removed" yes "$(gone "fat-${run}-b" && echo yes || echo no)"
    want "a shard on another runner keeps its container" no "$(gone "fat-${run}-c" && echo yes || echo no)"
    want "a run id that extends this one is kept" no "$(gone "fat-${run}1-a" && echo yes || echo no)"
    want "a name that only contains the run id is kept" no "$(gone "fat-x${run}-a" && echo yes || echo no)"
    want "a runner temp that only shares a prefix is kept" no "$(gone "fat-${run}-d" && echo yes || echo no)"
    want "it reports the two it removed" 1 "$(printf '%s\n' "$out" | grep -c '2 container(s) removed')"

    out=$(bash "$HERE/$PROG" "$run" "$td/rt1" 2>&1); rc=$?
    want "a second sweep (the passing path) finds none and exits 0" "0:1" "$rc:$(printf '%s\n' "$out" | grep -c '0 container(s) removed')"

    # The recorded driver pid is stopped; a pid that is not fat_driver is not.
    printf '%s' "$$" > "$td/rt2/fat/driver.pid"
    bash "$HERE/$PROG" "$run" "$td/rt2" >/dev/null 2>&1
    want "a recorded pid that is not fat_driver is left alone" yes "$(kill -0 $$ 2>/dev/null && echo yes || echo no)"
    mkdir -p "$td/bin"
    cp /bin/sleep "$td/bin/fat_driver.py"
    "$td/bin/fat_driver.py" 300 &
    local dpid=$!
    printf '%s' "$dpid" > "$td/rt2/fat/driver.pid"
    bash "$HERE/$PROG" "$run" "$td/rt2" >/dev/null 2>&1
    # No `wait` here: it would block until the sleep ends by itself. A killed
    # child that is not reaped yet is a zombie, which counts as stopped.
    want "the recorded fat_driver pid is stopped" no "$(running "$dpid")"
    kill -KILL "$dpid" 2>/dev/null; wait "$dpid" 2>/dev/null
    want "the shard container is removed with its own runner temp" yes "$(gone "fat-${run}-c" && echo yes || echo no)"

    bash "$HERE/$PROG" "$run" "relative/path" >/dev/null 2>&1; rc=$?
    want "a relative runner temp is refused" 1 "$rc"
    bash "$HERE/$PROG" "" "$td/rt1" >/dev/null 2>&1; rc=$?
    want "an empty run id is refused" 1 "$rc"

    docker rm -f "fat-${run}-d" >/dev/null 2>&1
    printf '\n%s passed, %s failed\n' "$ok" "$fails"
    [ "$fails" -eq 0 ] || { echo 'SELF-TEST FAILED'; exit 1; }
    exit 0
}

case "${1:-}" in
  --self-test) self_test ;;
  -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
  *)           [ "$#" -eq 2 ] || { echo "usage: $PROG <run_id> <runner_temp> | --self-test"; exit 1; }
               teardown "$1" "$2" ;;
esac
