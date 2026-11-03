#!/usr/bin/env bash
# #6719 falsifier: bump-version.sh --scratch-ok accepts "${TMPDIR:-/tmp}" children and nothing wider.
# --mutants proves the table fails on the pre-fix line and on a widened one.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=scripts/bump-version.sh

# row <name> <expect 0|1> <TMPDIR or "-" for unset> <path>
fails=0
row() {
    local name=$1 want=$2 tmpdir=$3 path=$4 rc=0
    if [ "$tmpdir" = "-" ]; then env -u TMPDIR bash "$SCRIPT" --scratch-ok "$path" || rc=$?
    else env TMPDIR="$tmpdir" bash "$SCRIPT" --scratch-ok "$path" || rc=$?; fi
    if [ "$rc" = "$want" ]; then echo "PASS  $name"; else echo "FAIL  $name (rc=$rc want=$want)"; fails=1; fi
}

table() {
    row "/tmp child, TMPDIR unset"                0 - /tmp/tmp.abc
    row "TMPDIR child outside /tmp (the incident)" 0 /mnt/ci/tmp /mnt/ci/tmp/tmp.abc
    row "TMPDIR with trailing slash"              0 /mnt/ci/tmp/ /mnt/ci/tmp/tmp.abc
    row "TMPDIR set, /tmp child still ok"         0 /mnt/ci/tmp /tmp/tmp.abc
    row "path outside TMPDIR and /tmp"            1 /mnt/ci/tmp /mnt/other/tmp.abc
    row "sibling sharing TMPDIR prefix"           1 /mnt/ci/tmp /mnt/ci/tmpx/tmp.abc
    row "TMPDIR itself is not a child"            1 /mnt/ci/tmp /mnt/ci/tmp
    row "TMPDIR=/ does not widen"                 1 / /etc/x
    row "TMPDIR empty does not widen"             1 "" /etc/x
    row "empty path"                              1 /mnt/ci/tmp ""
}

run() { SCRIPT=$1; fails=0; table; return "$fails"; }

if [ "${1:-}" = "--mutants" ]; then
    d=$(mktemp -d); trap 'rm -rf "${d:?}"' EXIT
    run "$S" >/dev/null || { echo "FAIL: table is red on the real script"; exit 1; }
    # pre-fix: only /tmp and /var/folders
    sed 's#/tmp/\*|/var/folders/\*|"\$base"/\*) return 0#/tmp/*|/var/folders/*) return 0#' "$S" > "$d/old.sh"
    # widened: any path
    sed 's#/tmp/\*|/var/folders/\*|"\$base"/\*) return 0#*) return 0#' "$S" > "$d/wide.sh"
    # slash fallback removed
    sed 's#\[ -n "\$base" \] || base=/tmp#:#' "$S" > "$d/noslash.sh"
    for m in old wide noslash; do
        cmp -s "$S" "$d/$m.sh" && { echo "FAIL: mutant $m did not change the script"; exit 1; }
        if run "$d/$m.sh" >/dev/null; then echo "FAIL: mutant $m survived"; exit 1; fi
        echo "mutant $m: killed"
    done
    exit 0
fi
run "$S"
