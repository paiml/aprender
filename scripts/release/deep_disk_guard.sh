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

# The temp dir the job really writes to: TMPDIR, else the runner's per-job RUNNER_TEMP, else /tmp (#6797:
# /tmp sat on a 29 GiB volume while the checkout and RUNNER_TEMP had 162 GiB, so the floor read a volume no step used).
resolve_tmp() { printf '%s' "${TMPDIR:-${RUNNER_TEMP:-/tmp}}"; }

# Point later steps at that same dir: with TMPDIR unset, mktemp and cargo would still write to /tmp.
export_tmp() {
    if [ -z "${TMPDIR:-}" ] && [ -n "${RUNNER_TEMP:-}" ] && [ -n "${GITHUB_ENV:-}" ]; then
        printf 'TMPDIR=%s\n' "$RUNNER_TEMP" >> "$GITHUB_ENV"
    fi
}

free_gib() { df -P -BG -- "$1" 2>/dev/null | awk 'NR==2 {gsub(/G/, "", $4); print $4}'; }

guard() {
    local min=$1 bad=0 name path free
    local target
    target=$(cargo metadata --no-deps --format-version 1 2>/dev/null | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
    for pair in "TMPDIR:$(resolve_tmp)" "checkout:$PWD" "target:${target:-$PWD}"; do
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
    # resolve_tmp case table: TMPDIR beats RUNNER_TEMP beats /tmp; empty values fall through.
    while IFS='|' read -r t r want; do
        got=$(TMPDIR=$t RUNNER_TEMP=$r resolve_tmp)
        [ "$got" = "$want" ] || { echo "FAIL self-test: resolve_tmp TMPDIR='$t' RUNNER_TEMP='$r' want '$want' got '$got'"; rc=1; }
    done <<'CASES'
/a||/a
|/r|/r
/a|/r|/a
||/tmp
CASES
    ge=$(mktemp); (unset TMPDIR; RUNNER_TEMP=/r GITHUB_ENV=$ge export_tmp)
    [ "$(cat "$ge")" = "TMPDIR=/r" ] || { echo "FAIL self-test: export_tmp must write TMPDIR=/r"; rc=1; }
    rm -f "${ge:?}"
    guard 0 > /dev/null 2>&1 || { echo "FAIL self-test: floor 0 must pass"; rc=1; }
    # guard() must measure the dir resolve_tmp names (#6797): with TMPDIR unset it reads RUNNER_TEMP, not /tmp.
    # Capture to a variable, not `| grep -q`: grep exiting early SIGPIPEs guard's next printf under pipefail.
    rt=$(mktemp -d) && [ -d "$rt" ] || { echo "FAIL self-test: mktemp"; exit 1; }
    go=$(unset TMPDIR; RUNNER_TEMP=$rt guard 0 2>&1)
    case "$go" in *"TMPDIR   "*" at $rt "*) ;; *) echo "FAIL self-test: guard must measure RUNNER_TEMP when TMPDIR is unset"; rc=1 ;; esac
    rmdir -- "${rt:?}"
    guard 99999999 > /dev/null 2>&1; g=$?
    [ "$g" -eq 3 ] || { echo "FAIL self-test: impossible floor must exit 3, got $g"; rc=1; }
    [ "$rc" -eq 0 ] && echo "self-test: PASS"
    exit "$rc"
fi
export_tmp
guard "${DEEP_DISK_MIN_GIB:-38}"
