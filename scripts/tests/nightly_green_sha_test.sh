#!/usr/bin/env bash
# Cases for scripts/nightly_green_sha.sh (#7462): the nightly builds the newest
# first-parent main commit whose 'ci / gate' is success, walking at most 20.
# Input lines are "<sha> <state>", newest first; line 1 is HEAD.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
script="$here/../nightly_green_sha.sh"
tmp=$(mktemp -d)
trap 'rm -rf "${tmp:?}"' EXIT

fails=0
total=0

check() {
    total=$((total + 1))
    if [ "$2" = "$3" ]; then
        echo "ok   $1"
    else
        echo "FAIL $1: got '$2' want '$3'"
        fails=$((fails + 1))
    fi
}

pick() { bash "$script" pick < "$1" 2> /dev/null || true; }

# (a) HEAD green -> HEAD
printf 'h success\np1 success\np2 failure\n' > "$tmp/a"
check "(a) HEAD green -> HEAD" "$(pick "$tmp/a")" "h"

# (b) HEAD red, parent 1 green -> parent 1
printf 'h failure\np1 success\np2 success\n' > "$tmp/b"
check "(b) HEAD red, parent 1 green -> p1" "$(pick "$tmp/b")" "p1"

# (c) HEAD and the 19 parents after it all red (20 commits) -> red-ci sentinel
{
    printf 'h failure\n'
    for i in $(seq 1 19); do printf 'p%d failure\n' "$i"; done
} > "$tmp/c"
check "(c) all 20 red -> red-ci" "$(pick "$tmp/c")" "red-ci"

# (d) HEAD pending -> HEAD unchanged (a green parent is not preferred)
printf 'h pending\np1 success\n' > "$tmp/d"
check "(d) HEAD pending -> HEAD" "$(pick "$tmp/d")" "h"

# (e) walk stops after 20 commits: the 21st is green but never reached -> red-ci
{
    cat "$tmp/c"
    printf 'p20 success\n'
} > "$tmp/e"
check "(e) green at commit 21 is not reached -> red-ci" "$(pick "$tmp/e")" "red-ci"

if [ "$fails" -eq 0 ]; then
    echo "nightly_green_sha self-test: $total/$total ok"
    exit 0
fi
echo "nightly_green_sha self-test: $fails/$total FAILED"
exit 1
