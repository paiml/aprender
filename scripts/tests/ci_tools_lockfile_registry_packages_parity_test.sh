#!/usr/bin/env bash
# ci_tools_lockfile_registry_packages_parity_test.sh -- #4827: the identical-output case
# table for `aprender-ci-tools lockfile-registry-packages` against
# scripts/lib/lockfile_registry_packages.py (C301: no Python in the build; the .py is the
# EXTERNAL VALIDATOR here, never a build step).
#
# Every case runs the same argv through both sides from the same directory. stdout must be
# byte-identical and the exit status equal, and a side that fails must say why on stderr.
# The fixtures cover the block-scoped parse (registry, path and git packages, `source`
# before `name`, another table ending the section), text-mode reading (`\r\n` and lone `\r`
# line ends, invalid UTF-8, a BOM), the whitespace `str.strip()` removes and Rust's `trim`
# does not, and argv taken verbatim (`--` and `-h` as file names, extra arguments ignored,
# a non-UTF-8 file name). The repo's own Cargo.lock is one case, and must list packages.
#
# Not vacuous (L25): the run fails if fewer cases ran than the table declares, and two
# planted liars (one extra stdout line, one wrong exit status) must be reported as
# mismatches before any real case counts.
#
# Usage: scripts/tests/ci_tools_lockfile_registry_packages_parity_test.sh
#        (CI_TOOLS_BIN=<path> to skip the build)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
LRP_PY="$ROOT/scripts/lib/lockfile_registry_packages.py"
EXPECTED_CASES=40

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
REAL_BIN="$BIN"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
work="$tmp/work"
mkdir -p "$work"
pass=0
fail=0

# side PREFIX CMD...: run CMD in $work; stdout to PREFIX.out, stderr to PREFIX.err,
# print the exit status.
side() {
    local prefix="$1" rc=0
    shift
    (cd "$work" && "$@") >"$prefix.out" 2>"$prefix.err" || rc=$?
    echo "$rc"
}

# same ARGS...: 0 when both sides agree on stdout and status, and a failing side wrote a
# reason to stderr.
same() {
    local a b
    a="$(side "$tmp/py" "$PY" "$LRP_PY" "$@")"
    b="$(side "$tmp/rs" "$BIN" lockfile-registry-packages "$@")"
    [[ "$a" == "$b" ]] || return 1
    cmp -s "$tmp/py.out" "$tmp/rs.out" || return 1
    if [[ "$a" != 0 ]]; then
        [[ -s "$tmp/py.err" && -s "$tmp/rs.err" ]] || return 1
    fi
    return 0
}

# check NAME ARGS...: one counted case.
check() {
    local name="$1"
    shift
    if same "$@"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: $name" >&2
        diff <(cat "$tmp/py.out") <(cat "$tmp/rs.out") | head -n 10 >&2 || true
    fi
}

# fx NAME CONTENT: a fixture in $work, CONTENT expanded by printf %b.
fx() {
    printf '%b' "$2" >"$work/$1"
}

REG='registry+https://github.com/rust-lang/crates.io-index'
fx basic.lock "version = 4\n\n[[package]]\nname = \"a\"\nversion = \"0.1.0\"\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.0\"\nsource = \"$REG\"\n\n[[package]]\nname = \"g\"\nsource = \"git+https://github.com/o/g#abc\"\n\n[[package]]\nname = \"ws\"\n"

# --- planted liars: the comparator must catch both before any case counts ---
liar="$tmp/liar"
printf '#!/usr/bin/env bash\nshift\n"%s" "%s" "$@"\nrc=$?\necho planted\nexit "$rc"\n' "$PY" "$LRP_PY" >"$liar"
chmod +x "$liar"
BIN="$liar"
if same basic.lock; then
    echo "FAIL: a planted extra stdout line was not detected" >&2
    exit 1
fi
printf '#!/usr/bin/env bash\nshift\n"%s" "%s" "$@"\nexit 3\n' "$PY" "$LRP_PY" >"$liar"
BIN="$liar"
if same basic.lock; then
    echo "FAIL: a planted wrong exit status was not detected" >&2
    exit 1
fi
BIN="$REAL_BIN"

# --- the real lockfile ---
cp "$ROOT/Cargo.lock" "$work/real.lock"
if [[ "$(cd "$work" && "$PY" "$LRP_PY" real.lock | wc -l)" -lt 1 ]]; then
    echo "FAIL: the repo Cargo.lock listed no registry package; the case would be vacuous" >&2
    exit 1
fi
check "repo Cargo.lock" real.lock

# --- block-scoped parse ---
check "registry, path, git and workspace packages" basic.lock
fx order.lock "[[package]]\nsource = \"$REG\"\nname = \"late\"\nversion = \"1\"\n"
check "source before name" order.lock
fx meta.lock "[[package]]\nname = \"p\"\nsource = \"$REG\"\n[metadata]\nname = \"m\"\nsource = \"$REG\"\n"
check "another table ends the section; its pair flushes at EOF" meta.lock
fx leak.lock "[[package]]\nname = \"ws\"\n[[package]]\nsource = \"$REG\"\n[[package]]\nname = \"r\"\nsource = \"$REG\"\n"
check "a source never leaks onto the preceding package" leak.lock
fx top.lock "name = \"top\"\nsource = \"$REG\"\n[[package]]\nname = \"b\"\n"
check "a pair before the first [[package]]" top.lock
fx comment.lock "[[package]] # c\nname = \"x\"\nsource = \"$REG\"\n[[package]]\nname = \"y\"\nsource = \"$REG\"\n"
check "[[package]] with a trailing comment is another table" comment.lock
fx dupkey.lock "[[package]]\nname = \"first\"\nname = \"second\"\nsource = \"git+x\"\nsource = \"$REG\"\n"
check "a repeated key: the last wins" dupkey.lock
fx dupname.lock "[[package]]\nname = \"d\"\nsource = \"$REG\"\n[[package]]\nname = \"d\"\nsource = \"$REG\"\n"
check "the same name twice is printed twice" dupname.lock
fx noeol.lock "[[package]]\nname = \"n\"\nsource = \"$REG\""
check "no newline at end of file" noeol.lock
fx empty.lock ""
check "empty file" empty.lock
fx nopkg.lock "version = 4\n[metadata]\n"
check "no package at all" nopkg.lock

# --- key and value spelling ---
fx quotes.lock "[[package]]\nname = \"\"q\"\"\nsource = $REG\n[[package]]\nname = \"half\nsource = \"$REG\n"
check "repeated and unbalanced quotes; unquoted source" quotes.lock
fx eqs.lock "[[package]]\nname = \"a\" = \"b\"\nsource = \"$REG\"\n"
check "only the first = splits" eqs.lock
fx nospace.lock "[[package]]\nname=\"tight\"\nsource = \"$REG\"\n[[package]]\nname = \"loose\"\nsource=\"$REG\"\n"
check "key = without spaces does not match" nospace.lock
fx emptyname.lock "[[package]]\nname = \nsource = \"$REG\"\n[[package]]\nname = \"\"\nsource = \"$REG\"\n"
check "empty name values" emptyname.lock
fx srcspell.lock "[[package]]\nname = \"a\"\nsource = \" registry+x\"\n[[package]]\nname = \"b\"\nsource = \"Registry+x\"\n[[package]]\nname = \"c\"\nsource = \"registry+\"\n"
check "source prefix is exact" srcspell.lock
fx inner.lock "[[package]]\nname = \" sp \"\nsource = \"$REG\"\n"
check "whitespace inside quotes is kept" inner.lock

# --- text-mode reading ---
fx crlf.lock "[[package]]\r\nname = \"crlf\"\r\nsource = \"$REG\"\r\n"
check "CRLF line ends" crlf.lock
fx cr.lock "[[package]]\rname = \"cr\"\rsource = \"$REG\"\r"
check "lone CR line ends" cr.lock
fx mixed.lock "[[package]]\r\nname = \"m1\"\rsource = \"$REG\"\n[[package]]\r\rname = \"m2\"\r\nsource = \"$REG\""
check "mixed line ends" mixed.lock
fx bad.lock "[[package]]\nname = \"x\xff\xfe\"\nsource = \"$REG\"\n"
check "invalid UTF-8 bytes" bad.lock
fx partial.lock "[[package]]\nname = \"a\xe2\x82b\xf0\x9f\x98c\xc0\xafd\xed\xa0\x80e\"\nsource = \"$REG\"\n"
check "truncated, overlong and surrogate sequences" partial.lock
fx bom.lock "\xef\xbb\xbf[[package]]\nname = \"bom\"\nsource = \"$REG\"\n"
check "a BOM before the first [[package]]" bom.lock
fx utf8.lock "[[package]]\nname = \"caf\xc3\xa9-\xe2\x9c\x93\"\nsource = \"$REG\"\n"
check "non-ASCII names" utf8.lock

# --- whitespace str.strip() removes ---
fx ws.lock "\t [[package]] \t\n  name = \"tab\"\t\n\tsource = \"$REG\"  \n"
check "tabs and spaces around lines" ws.lock
fx fs.lock "\x1c[[package]]\x1f\n\x1dname = \"fs\"\x1e\nsource = \"$REG\"\x1f\n"
check "x1c-x1f: Python strips them, Rust trim does not" fs.lock
fx vt.lock "\x0b[[package]]\x0c\nname = \"vt\"\x0b\nsource = \"$REG\"\x0c\n"
check "vertical tab and form feed" vt.lock
fx uni.lock "\xc2\xa0[[package]]\xe3\x80\x80\n\xe2\x80\xa8name = \"nb\"\xc2\x85\nsource = \"$REG\"\xe2\x80\xa9\n"
check "NBSP, U+3000, NEL, U+2028 and U+2029" uni.lock
fx zw.lock "[[package]]\nname = \"zw\"\xe2\x80\x8b\nsource = \"$REG\"\n\xe2\x80\x8b[[package]]\nname = \"after\"\nsource = \"$REG\"\n"
check "U+200B is not whitespace on either side" zw.lock

# --- argv taken verbatim ---
check "a missing file" no-such.lock
check "no argument"
check "an empty argument" ""
mkdir -p "$work/adir"
check "a directory" adir
cp "$work/basic.lock" "$work/--"
check "-- is a file name" --
cp "$work/basic.lock" "$work/-h"
check "-h is a file name" -h
check "arguments after the first are ignored" basic.lock no-such.lock
check "a missing first argument is not rescued by the second" no-such.lock basic.lock
cp "$work/basic.lock" "$work/$(printf 'n\xff.lock')"
check "a non-UTF-8 file name" "$(printf 'n\xff.lock')"
check "an absolute path" "$work/order.lock"

total=$((pass + fail))
echo "ci_tools_lockfile_registry_packages_parity: $pass/$total identical (declared $EXPECTED_CASES)"
if [[ "$total" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $total cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
