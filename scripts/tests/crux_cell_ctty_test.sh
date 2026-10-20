#!/usr/bin/env bash
# crux_cell_ctty_test.sh -- a CRUX cell must not hand its engine a controlling tty (L2).
# An engine that writes its answer to /dev/tty when it can open it (llama-cli does) must
# land on stdout in the cell, even when the cell is launched from a terminal.
#   scripts/tests/crux_cell_ctty_test.sh   exit 0 = pass
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
d=$(mktemp -d)
trap 'rm -rf -- "${d:?}"' EXIT

# Pull the real cell_add out of the harness; the script is not sourceable.
eval "$(sed -n '/^cell_add() {/,/^}/p' "$root/scripts/crux_inference_dogfood.sh")"
TMO=20

cell="$d/cell.sh"
: > "$cell"
cell_add "$cell" "$d/eng" bash -c 'if echo answer > /dev/tty 2>/dev/null; then :; else echo answer; fi'

# script(1) gives the cell a controlling pty, as a tmux launch does.
script -qec "bash $cell" /dev/null > /dev/null 2>&1 < /dev/null || true

got=$(cat "$d/eng.out" 2>/dev/null || true)
if [ "$got" != answer ] || [ "$(cat "$d/eng.rc" 2>/dev/null)" != 0 ]; then
    echo "FAIL: engine output lost under a controlling tty: out='$got' rc='$(cat "$d/eng.rc" 2>/dev/null)'" >&2
    exit 1
fi
echo "crux_cell_ctty_test: ok"
