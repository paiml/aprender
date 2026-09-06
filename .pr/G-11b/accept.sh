#!/usr/bin/env bash
# G-11b accept.sh — re-runs every A_i in one call (I5)
set -uo pipefail; cd "$(dirname "$0")/../.."; rc=0
run() { printf '== %s\n' "$*"; "$@"; local r=$?; printf 'rc=%s\n' "$r"; [ "$r" = 0 ] || rc=1; }
run bash scripts/pp066_state.sh --self-test
run bash scripts/pp066_state.sh --no-gh
run bash scripts/fleet_verify.sh --self-test
run make -n fleet-verify ROW=G-11b
exit "$rc"
