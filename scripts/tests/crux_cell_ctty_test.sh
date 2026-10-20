#!/usr/bin/env bash
# crux_cell_ctty_test.sh -- a CRUX cell must not hand its engine a controlling tty (L2).
# An engine that writes its answer to /dev/tty when it can open it (llama-cli does) must
# land on stdout in the cell, even when the cell is launched from a terminal. Covers
# cell_add, cell_add_stdin (the apr chat path) and the cell_result "rc 0, 0 B" branch.
#   scripts/tests/crux_cell_ctty_test.sh   exit 0 = pass
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
d=$(mktemp -d)
trap 'rm -rf -- "${d:?}"' EXIT

# Pull the real functions out of the harness; the script is not sourceable.
for fn in cell_add cell_add_stdin cell_result; do
    eval "$(sed -n "/^$fn() {/,/^}/p" "$root/scripts/crux_inference_dogfood.sh")"
done
TMO=20
CELL_WHY=""
emit_gen() { printf 'EMIT rc=[%s] refused=[%s]\n' "$3" "$6" > "$d/emit.txt"; } # stub for the python row writer

fail() { echo "FAIL: $*" >&2; exit 1; }
# script(1) gives the cell a controlling pty, as a tmux launch does.
run_in_pty() { script -qec "bash $1" /dev/null > /dev/null 2>&1 < /dev/null || true; }
tty_engine='if echo answer > /dev/tty 2>/dev/null; then :; else cat > /dev/null; echo answer; fi'

# cell_add
cell="$d/cell.sh"; : > "$cell"
cell_add "$cell" "$d/eng" bash -c "$tty_engine"
run_in_pty "$cell"
[ "$(cat "$d/eng.out" 2>/dev/null)" = answer ] && [ "$(cat "$d/eng.rc" 2>/dev/null)" = 0 ] \
    || fail "cell_add: output lost under a controlling tty: out='$(cat "$d/eng.out" 2>/dev/null)'"

# cell_add_stdin (apr chat feeds turns from a file)
printf 'hello\n' > "$d/turns.txt"
cell="$d/cell2.sh"; : > "$cell"
cell_add_stdin "$cell" "$d/chat" "$d/turns.txt" bash -c "$tty_engine"
run_in_pty "$cell"
[ "$(cat "$d/chat.out" 2>/dev/null)" = answer ] && [ "$(cat "$d/chat.rc" 2>/dev/null)" = 0 ] \
    || fail "cell_add_stdin: output lost under a controlling tty: out='$(cat "$d/chat.out" 2>/dev/null)'"
grep -q 'setsid -w' "$d/cell2.sh" || fail "cell_add_stdin line has no setsid -w"

# cell_result: rc 0 with 0 B is a harness error, never a blank cell
: > "$d/blank.out"; : > "$d/blank.err"; echo 0 > "$d/blank.rc"
cell_result e p "$d/blank"
grep -q 'rc=\[\] refused=\[harness error: rc 0 but 0 B' "$d/emit.txt" || fail "cell_result: blank rc 0 not a harness error: $(cat "$d/emit.txt")"
# non-empty stdout, rc 0: a normal row
echo hi > "$d/ok.out"; : > "$d/ok.err"; echo 0 > "$d/ok.rc"
cell_result e p "$d/ok"
grep -q 'rc=\[0\] refused=\[\]' "$d/emit.txt" || fail "cell_result: normal row altered: $(cat "$d/emit.txt")"
# rc nonzero with 0 B stays the engine's own failure row
: > "$d/bad.out"; : > "$d/bad.err"; echo 1 > "$d/bad.rc"
cell_result e p "$d/bad"
grep -q 'rc=\[1\] refused=\[\]' "$d/emit.txt" || fail "cell_result: nonzero rc row altered: $(cat "$d/emit.txt")"
# explicit stdout file argument is the one checked
: > "$d/x.out"; : > "$d/x.err"; echo 0 > "$d/x.rc"; echo '{}' > "$d/x.json"
cell_result e p "$d/x" "$d/x.json"
grep -q 'rc=\[0\] refused=\[\]' "$d/emit.txt" || fail "cell_result: [stdout file] arg ignored: $(cat "$d/emit.txt")"

echo "crux_cell_ctty_test: ok"
