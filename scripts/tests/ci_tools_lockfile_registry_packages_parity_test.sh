#!/usr/bin/env bash
# ci_tools_lockfile_registry_packages_parity_test.sh -- PMAT-4827: the identical-output case
# table for `aprender-ci-tools lockfile-registry-packages` against
# scripts/lib/lockfile_registry_packages.py (C301: no Python in the build; the .py is the
# EXTERNAL VALIDATOR here, never a build step).
#
# Every case runs one Cargo.lock through both sides; stdout must be byte-identical and the
# exit status equal. Inputs are hand-written edge cases plus the repo's real Cargo.lock files.
#
# Its own script, not a section of ci_tools_py_parity_test.sh: sibling ports bump that file's
# EXPECTED_CASES, and a shared counter is a merge conflict.
#
# Not vacuous (L25): the run fails if fewer cases ran than the table declares, and a planted
# mismatch must be reported as one before any real case counts.
#
# Usage: scripts/tests/ci_tools_lockfile_registry_packages_parity_test.sh   (CI_TOOLS_BIN=<path> to skip the build)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
LRP_PY="$ROOT/scripts/lib/lockfile_registry_packages.py"
EXPECTED_CASES=24

BIN="${CI_TOOLS_BIN:-}"
if [[ -z "$BIN" ]]; then
    cargo build -q -p aprender-ci-tools
    tdir="$(cargo metadata --no-deps --format-version 1 | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4)"
    BIN="$tdir/debug/aprender-ci-tools"
fi
if [[ ! -x "$BIN" ]]; then
    echo "FAIL: no aprender-ci-tools binary at $BIN" >&2
    exit 1
fi
BIN="$(realpath -e -- "$BIN")"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
pass=0
fail=0

# side OUT CMD...: stdout to OUT, print the exit status.
side() {
    local out="$1" rc=0
    shift
    "$@" >"$out" 2>/dev/null || rc=$?
    echo "$rc"
}

# compare NAME LOCK: .py vs bin on one lockfile.
compare() {
    local a b
    a="$(side "$tmp/py.out" "$PY" "$LRP_PY" "$2")"
    b="$(side "$tmp/rs.out" "$BIN" lockfile-registry-packages "$2")"
    if [[ "$a" == "$b" ]] && cmp -s "$tmp/py.out" "$tmp/rs.out"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: $1 (py rc=$a, rs rc=$b)" >&2
        diff "$tmp/py.out" "$tmp/rs.out" | head -n 6 >&2 || true
    fi
}

# --- L25: a planted mismatch must be seen as one. -------------------------------------
printf '[[package]]\nname = "x"\nsource = "registry+u"\n' >"$tmp/plant.lock"
real_bin="$BIN"
printf '#!/bin/sh\necho wrong\n' >"$tmp/liar"
chmod +x "$tmp/liar"
BIN="$tmp/liar"
compare plant "$tmp/plant.lock" 2>/dev/null
BIN="$real_bin"
if [[ "$fail" -ne 1 || "$pass" -ne 0 ]]; then
    echo "FAIL: the planted mismatch was not reported (pass=$pass fail=$fail)" >&2
    exit 1
fi
pass=0
fail=0

# --- Hand-written edge cases. -----------------------------------------------------------
R='source = "registry+https://github.com/rust-lang/crates.io-index"'
hw() { # NAME PRINTF_FORMAT
    printf "$2" >"$tmp/hw.lock"
    compare "$1" "$tmp/hw.lock"
}
hw "empty file" ''
hw "no packages" 'version = 3\n'
hw "one registry package" "[[package]]\nname = \"serde\"\nversion = \"1\"\n$R\n"
hw "workspace member (no source)" '[[package]]\nname = "mine"\nversion = "0"\n'
hw "git source" '[[package]]\nname = "g"\nsource = "git+https://x"\n'
hw "mixed" "[[package]]\nname = \"a\"\n$R\n[[package]]\nname = \"m\"\n[[package]]\nname = \"b\"\n$R\n"
hw "source must not leak forward" "[[package]]\nname = \"a\"\n[[package]]\nname = \"b\"\n$R\n"
hw "source before name" "[[package]]\n$R\nname = \"a\"\n"
hw "source without name" "[[package]]\n$R\n"
hw "name without source" '[[package]]\nname = "a"\n'
hw "no final newline" "[[package]]\nname = \"a\"\n$R"
hw "CRLF" "[[package]]\r\nname = \"a\"\r\n$R\r\n"
hw "lone CR" "[[package]]\rname = \"a\"\r$R\r"
hw "indented lines" "  [[package]]\n  name = \"a\"\n  $R\n"
hw "other table ends package" "[[package]]\nname = \"a\"\n$R\n[metadata]\nname = \"zz\"\n$R\n"
hw "dependencies array not a table" "[[package]]\nname = \"a\"\n$R\ndependencies = [\n \"b\",\n]\n"
hw "quotes stripped greedily" '[[package]]\nname = ""a""\nsource = "registry+x"\n'
hw "equals inside value" '[[package]]\nname = "a=b"\nsource = "registry+x?y=z"\n'
hw "name key prefix only" '[[package]]\nnamex = "a"\nsource = "registry+x"\n'
hw "non-UTF-8 bytes in a name" '[[package]]\nname = "a\377b"\nsource = "registry+x"\n'
hw "unicode whitespace around" '[[package]]\nname = \302\240"a"\302\240\nsource = "registry+x"\n'

# --- The repo's real lockfiles. -----------------------------------------------------------
for lock in Cargo.lock crates/facades/Cargo.lock; do
    [[ -s "$ROOT/$lock" ]] || { echo "FAIL: $lock is missing or empty" >&2; exit 1; }
    compare "real:$lock" "$ROOT/$lock"
done

# --- A missing file: both refuse with the same status and print nothing. --------------------
compare "missing file" "$tmp/does-not-exist.lock"

ran=$((pass + fail))
echo "lockfile-registry-packages parity: $pass passed, $fail failed, $ran of $EXPECTED_CASES declared"
if [[ "$fail" -ne 0 ]]; then
    exit 1
fi
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
