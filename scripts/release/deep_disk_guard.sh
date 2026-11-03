#!/usr/bin/env bash
# deep_disk_guard.sh - fail a deep-nightly job EARLY, and by name, when the runner disk is short.
#
# WHY. deep-nightly 2026-10-31 (run 41252910525): deep-bins-smoke and deep-nodefault died in
# `mktemp: failed to create directory via template '/tmp/tmp.XXXXXXXXXX': No space left on device`, and
# deep-doctests ended exit 101 on the same host. The lane then read as a code failure. The guard measures
# free space on the three places a lane writes (TMPDIR, the checkout, cargo's target dir) before any build,
# prints the numbers, and exits 3 ("INFRA disk") below the floor, so the red names its cause (#6126).
# It deletes nothing: a shared clean-room host holds other runs' directories, and no path here is this job's own.
#
#   bash scripts/release/deep_disk_guard.sh              # DEEP_DISK_MIN_GIB (default 38 = measured peak 25 GiB x 1.5)
#   bash scripts/release/deep_disk_guard.sh --self-test  # proves it can go RED
# Exit 0 enough space everywhere | 3 a path is below the floor | 2 cannot measure.
set -uo pipefail

free_gib() { df -P -BG -- "$1" 2>/dev/null | awk 'NR==2 {gsub(/G/, "", $4); print $4}'; }

guard() {
    local min=$1 bad=0 name path free
    local target
    target=$(cargo metadata --no-deps --format-version 1 2>/dev/null | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
    for pair in "TMPDIR:${TMPDIR:-/tmp}" "checkout:$PWD" "target:${target:-$PWD}"; do
        name=${pair%%:*}; path=${pair#*:}
        [ -d "$path" ] || path=$PWD
        free=$(free_gib "$path")
        if [ -z "$free" ]; then echo "INFRA disk: cannot measure $name ($path)" >&2; return 2; fi
        printf 'deep_disk_guard: %-8s %s GiB free at %s (floor %s)\n' "$name" "$free" "$path" "$min"
        [ "$free" -ge "$min" ] || bad=1
    done
    if [ "$bad" -ne 0 ]; then echo "infra: disk — a lane path is below the ${min} GiB floor (free GiB printed above); not_measured, not a code failure (#6126)" >&2; return 3; fi
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    rc=0
    guard 0 > /dev/null 2>&1 || { echo "FAIL self-test: floor 0 must pass"; rc=1; }
    guard 99999999 > /dev/null 2>&1; g=$?
    [ "$g" -eq 3 ] || { echo "FAIL self-test: impossible floor must exit 3, got $g"; rc=1; }
    [ "$rc" -eq 0 ] && echo "self-test: PASS"
    exit "$rc"
fi
guard "${DEEP_DISK_MIN_GIB:-38}"
