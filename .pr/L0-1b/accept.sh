#!/usr/bin/env bash
# L0-1b accept.sh — filled in as steps land (I5)
set -uo pipefail; cd "$(dirname "$0")/../.."; rc=0
run() { printf '== %s\n' "$*"; "$@"; local r=$?; printf 'rc=%s\n' "$r"; [ "$r" = 0 ] || rc=1; }
[ -f docs/audits/l0-1b-arms.md ] && run grep -q 'measured cosine' docs/audits/l0-1b-arms.md
printf '== step 0 (apr parity --per-op names a first diverging op on the 1.5B): not landed\nrc=1\n'; rc=1
exit "$rc"
