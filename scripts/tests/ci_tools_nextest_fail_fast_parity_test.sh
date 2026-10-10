#!/usr/bin/env bash
# ci_tools_nextest_fail_fast_parity_test.sh -- #5032: the case table for
# `aprender-ci-tools nextest-fail-fast` against the inline Python judge in
# scripts/check_nextest_ci_profile_no_fail_fast.sh (C301: no Python in the build; that
# Python is the EXTERNAL VALIDATOR here, never a build step).
#
# The judge is cut out of the guard (the text between <<'PY' and PY) and run as the guard
# runs it: once as is, which takes tomllib, and once with NEXTEST_GUARD_FORCE_FALLBACK=1,
# which takes its purpose-built reader. The bin runs the same file with --reader library
# and --reader fallback. Exit status must be equal on every case, and the verdict line
# must be byte-identical after two by-design differences are mapped, and nothing else:
#   - the reader's name (tomllib/tomli vs `toml crate`; the purpose-built reader's
#     "(no tomllib/tomli on this interpreter)", which the bin has no reason to say);
#   - the words of a TOML parse error (`TOMLDecodeError: ...`), which are each parser's own.
# Where a case's verdict is known (the guard's own table), both sides must also give it.
# Where the two TOML parsers disagree by design (the README's "differs" table), each side
# must give its own exit status, so a parser upgrade that moves one fails here.
#
# Not vacuous (L25): the run fails if fewer cases ran than the table declares, and two
# planted mismatches (a wrong exit status; the right one with the wrong words) must each be
# reported as one before any real case counts.
#
# Usage: scripts/tests/ci_tools_nextest_fail_fast_parity_test.sh
#        CI_TOOLS_BIN=<path> skips the build; PYTHON=<python3.11+> (the library reader is
#        tomllib, so an older interpreter is refused rather than compared as a fallback).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
GUARD="$ROOT/scripts/check_nextest_ci_profile_no_fail_fast.sh"
EXPECTED_CASES=123

if ! "$PY" -I -c 'import tomllib' 2>/dev/null; then
    echo "FAIL: $PY has no tomllib, so the guard's library reader cannot run here; set PYTHON to python3.11+" >&2
    exit 1
fi

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
sed -n "/<<'PY'\$/,/^PY\$/p" "$GUARD" | sed '1d;$d' >"$tmp/judge.py"
if ! grep -q '^def load_minimal' "$tmp/judge.py" || ! grep -q '^def load_with_lib' "$tmp/judge.py"; then
    echo "FAIL: cannot cut the judge out of $GUARD" >&2
    exit 1
fi
pass=0
fail=0

# norm: a verdict line with the two by-design differences mapped.
norm() {
    sed -E \
        -e 's/reader=(tomllib|tomli|toml crate)([,)])/reader=LIB\2/' \
        -e 's/reader=purpose-built reader( \(no tomllib\/tomli on this interpreter\))?([,)])/reader=MIN\2/' \
        -e 's/\(TOMLDecodeError: .*\) -- cannot judge/(TOMLDecodeError: <parser text>) -- cannot judge/'
}

# compare NAME MODE WANT FILE: the judge vs the bin on one file; MODE library|fallback,
# WANT the exit status both must give, or - when only parity is checked.
compare() {
    local name="$1" mode="$2" want="$3" f="$4" prc=0 rrc=0 fb=""
    [[ "$mode" == fallback ]] && fb=1
    NEXTEST_GUARD_RUNNER=parity NEXTEST_GUARD_FORCE_FALLBACK="$fb" \
        "$PY" -I "$tmp/judge.py" "$f" >"$tmp/py.out" 2>/dev/null || prc=$?
    NEXTEST_GUARD_RUNNER=parity "$BIN" nextest-fail-fast --reader "$mode" "$f" \
        >"$tmp/rs.out" 2>"$tmp/rs.err" || rrc=$?
    norm <"$tmp/py.out" >"$tmp/py.n"
    norm <"$tmp/rs.out" >"$tmp/rs.n"
    if [[ "$prc" == "$rrc" && ( "$want" == - || "$want" == "$prc" ) ]] \
        && grep -q '^VERDICT ' "$tmp/py.n" && cmp -s "$tmp/py.n" "$tmp/rs.n" && [[ ! -s "$tmp/rs.err" ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: [$mode] $name (py rc=$prc, rs rc=$rrc, want $want)" >&2
        diff "$tmp/py.n" "$tmp/rs.n" | head -n 6 >&2 || true
    fi
}

# row WANT_LIBRARY WANT_FALLBACK NAME BODY: one case under both readers.
row() {
    printf '%s' "$4" >"$tmp/c.toml"
    compare "$3" library "$1" "$tmp/c.toml"
    compare "$3" fallback "$2" "$tmp/c.toml"
}

# differ NAME WANT_JUDGE WANT_BIN BODY: a row of the README's "differs" table, under the
# library reader, where the two parsers disagree by design. Each side must give its own
# exit status and a VERDICT line, so the table is measured, not claimed.
differ() {
    local prc=0 rrc=0
    printf '%s' "$4" >"$tmp/c.toml"
    NEXTEST_GUARD_RUNNER=parity "$PY" -I "$tmp/judge.py" "$tmp/c.toml" >"$tmp/py.out" 2>/dev/null || prc=$?
    NEXTEST_GUARD_RUNNER=parity "$BIN" nextest-fail-fast --reader library "$tmp/c.toml" \
        >"$tmp/rs.out" 2>"$tmp/rs.err" || rrc=$?
    if [[ "$prc" == "$2" && "$rrc" == "$3" ]] && grep -q '^VERDICT ' "$tmp/py.out" \
        && grep -q '^VERDICT ' "$tmp/rs.out" && [[ ! -s "$tmp/rs.err" ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: [differs] $1 (py rc=$prc want $2, rs rc=$rrc want $3)" >&2
    fi
}

# --- L25: a planted mismatch must be seen as one. -------------------------------------
# Two liars: one with the wrong exit status, and one with the right exit status and the
# wrong words, so neither the status check nor the line check can pass alone.
printf '[profile.ci]\nfail-fast = true\n' >"$tmp/plant.toml"
real_bin="$BIN"
# shellcheck disable=SC2016
printf '#!/bin/sh\necho "VERDICT ok $4: [profile.ci].fail-fast = false -- a red run still measures everything else (reader=toml crate, runner=parity)"\n' >"$tmp/liar"
# shellcheck disable=SC2016
printf '#!/bin/sh\necho "VERDICT FAIL $4: [profile.ci].fail-fast = true"\nexit 1\n' >"$tmp/liar-text"
chmod +x "$tmp/liar" "$tmp/liar-text"
BIN="$tmp/liar"
compare plant library - "$tmp/plant.toml" 2>/dev/null
BIN="$tmp/liar-text"
compare plant-text library 1 "$tmp/plant.toml" 2>/dev/null
BIN="$real_bin"
if [[ "$fail" -ne 2 || "$pass" -ne 0 ]]; then
    echo "FAIL: a planted mismatch was not reported (pass=$pass fail=$fail)" >&2
    exit 1
fi
pass=0
fail=0

# --- The guard's own case table (its --self-test runs it under both readers). ---------
row 0 0 "false under [profile.ci]"                 $'[profile.ci]\nretries = 2\nfail-fast = false\n'
row 1 1 "true under [profile.ci]"                  $'[profile.ci]\nretries = 2\nfail-fast = true\n'
row 1 1 "key absent"                               $'[profile.ci]\nretries = 2\n'
row 1 1 "false under a different profile only"     $'[profile.default]\nfail-fast = false\n[profile.ci]\nretries = 2\n'
row 1 1 "no [profile.ci]"                          $'[profile.default]\nfail-fast = false\n'
row 1 1 "false in a comment, true in the key"      $'[profile.ci]\n# fail-fast = false\nfail-fast = true\n'
row 1 1 "the string \"false\""                     $'[profile.ci]\nfail-fast = "false"\n'
row 1 1 "false under [[profile.ci.overrides]]"     $'[profile.ci]\nretries = 2\n[[profile.ci.overrides]]\nfilter = "test(/x/)"\nfail-fast = false\n'
row 0 0 "the real file's shape"                    $'[profile.ci]\nretries = 2\nfail-fast = false\nslow-timeout = { period = "60s", terminate-after = 20 }\nstatus-level = "slow"\n[profile.ci.junit]\npath = "junit.xml"\n'
row 2 2 "[[profile.ci]]"                           $'[[profile.ci]]\nfail-fast = false\n'
row 2 2 "unparseable TOML"                         $'[profile.ci\nfail-fast = false\n'
row 0 0 "a top-level nextest key"                  $'experimental = ["setup-scripts"]\n[profile.ci]\nretries = 2\nfail-fast = false\n'
row 2 2 "an inline top-level profile"              $'profile = { ci = { "fail-fast" = true } }\n[profile.ci]\nfail-fast = false\n'
row 2 2 "the inline profile, key quoted"           $'"profile" = { ci = { "fail-fast" = true } }\n[profile.ci]\nfail-fast = false\n'
row 2 2 "a dotted top-level profile.ci.fail-fast"  $'profile.ci.fail-fast = true\n[profile.ci]\nfail-fast = false\n'
compare "the real .config/nextest.toml" library 0 "$ROOT/.config/nextest.toml"
compare "the real .config/nextest.toml" fallback 0 "$ROOT/.config/nextest.toml"
# what the purpose-built reader must refuse rather than guess
row - 2 "a multi-line string hiding a header"      $'[profile.default]\nnote = """title"\n[profile.ci]\nfail-fast = false\nx = """"\n'
row - 2 "a value continuing past its line"         $'[profile.ci]\nfail-fast = false\nslow-timeout = { period = "60s",\n  terminate-after = 20 }\n'
row - 2 "fail-fast = fals"                         $'[profile.ci]\nfail-fast = fals\n'
row - 2 "an unknown top-level key"                 $'experimentl = ["setup-scripts"]\n[profile.ci]\nfail-fast = false\n'

# --- The text of every message, and the edges of each reader. -------------------------
row 1 1 "an int"                                   $'[profile.ci]\nfail-fast = 1\n'
row 1 1 "-0"                                       $'[profile.ci]\nfail-fast = -0\n'
row 1 1 "a string with a quote"                    $'[profile.ci]\nfail-fast = "it\'s"\n'
row 1 1 "a literal string"                         $'[profile.ci]\nfail-fast = \'x\'\n'
row 1 2 "a float"                                  $'[profile.ci]\nfail-fast = 1.5\n'
row 1 2 "a float in exponent form"                 $'[profile.ci]\nfail-fast = 1e20\n'
row 1 2 "an array"                                 $'[profile.ci]\nfail-fast = [true, "a"]\n'
row 1 2 "an inline table"                          $'[profile.ci]\nfail-fast = { a = 1 }\n'
row 1 1 "true with a comment"                      $'[profile.ci]\nfail-fast = true # c\n'
row 1 2 "a quoted false with a comment"            $'[profile.ci]\nfail-fast = "false" # c\n'
row 2 2 "profile = 1"                              $'profile = 1\n'
row 2 1 "[profile] ci = 1"                         $'[profile]\nci = 1\n'
row 0 0 "CRLF"                                     $'[profile.ci]\r\nfail-fast = false\r\n'
row - 0 "a lone CR"                                $'[profile.ci]\rfail-fast = false\r'
row - 0 "\\x1c as whitespace"                      $'[\x1cprofile.ci]\nfail-fast\x1c= false\n'
row 2 2 "a BOM"                                    $'\xef\xbb\xbf[profile.ci]\nfail-fast = false\n'
row 2 2 "not UTF-8"                                $'[profile.ci]\nfail-fast = false\n\xff\n'
row 2 2 "a duplicate table"                        $'[a]\n[a]\n'
row 2 2 "a duplicate key"                          $'[profile.ci]\nx = 1\nx = 2\n'
row 2 1 "leading zeros"                            $'[profile.ci]\nfail-fast = 007\n'
row 2 1 "a non-ASCII decimal digit"                $'[profile.ci]\nfail-fast = \xd9\xa3\n'
row 0 1 "spaces around the dot"                    $'[ profile . ci ]\nfail-fast = false\n'
row 0 0 "a quoted header part"                     $'["profile".ci]\nfail-fast = false\n'
row 2 2 "[profile.ci] twice"                       $'[profile.ci]\nfail-fast = false\n[profile.ci]\n'
compare "a missing file" library 2 "$tmp/absent.toml"
compare "a missing file" fallback 2 "$tmp/absent.toml"

# --- Dates, times and floats both parsers read the same way. --------------------------
ok=$'[profile.ci]\nfail-fast = false\n'
row 2 0 "a date tomllib refuses: Feb 30"           "${ok}t = 2021-02-30"$'\n'
row 2 0 "a time with no seconds"                   "${ok}t = 07:32"$'\n'
row 2 0 "an offset of +24:00"                      "${ok}t = 1979-05-27T07:32:00+24:00"$'\n'
row 2 0 "DEL in a comment"                         "${ok}# a"$'\x7f\n'
row 0 0 "seven fractional-second digits"           "${ok}t = 1979-05-27T07:32:00.1234567Z"$'\n'
row 1 2 "fail-fast = inf"                          $'[profile.ci]\nfail-fast = inf\n'
row 1 2 "fail-fast = -nan"                         $'[profile.ci]\nfail-fast = -nan\n'

# --- CPython's 4300-digit int() limit, in the purpose-built reader. -------------------
# Leading zeros count and the sign does not.
d4300="$(printf '%*s' 4300 '' | tr ' ' '1')"
printf '%sx = %s\n' "$ok" "$d4300" >"$tmp/c.toml"
compare "a 4300-digit key" fallback 0 "$tmp/c.toml"
printf '%sx = 1%s\n' "$ok" "$d4300" >"$tmp/c.toml"
compare "a 4301-digit key" fallback 2 "$tmp/c.toml"
printf '[profile.ci]\nfail-fast = -%s\n' "$d4300" >"$tmp/c.toml"
compare "fail-fast = -<4300 digits>" fallback 1 "$tmp/c.toml"
printf '[profile.ci]\nfail-fast = -0%s\n' "$d4300" >"$tmp/c.toml"
compare "fail-fast = -0<4300 digits>" fallback 2 "$tmp/c.toml"

# --- Where the two TOML parsers disagree (the README's "differs" table). ---------------
nest() { printf '%*s' "$1" '' | tr ' ' "$2"; printf 1; printf '%*s' "$1" '' | tr ' ' "$3"; }
differ "a leap second in a datetime"     2 0 "${ok}t = 1979-05-27T07:32:60Z"$'\n'
differ "a leap second in a time"         2 0 "${ok}t = 07:32:60"$'\n'
differ "year 0000"                       2 0 "${ok}t = 0000-01-01"$'\n'
differ "an integer above i64"            0 2 "${ok}x = 9223372036854775808"$'\n'
differ "an integer below i64"            0 2 "${ok}x = -9223372036854775809"$'\n'
differ "a hex integer above i64"         0 2 "${ok}x = 0xffffffffffffffff"$'\n'
differ "an octal integer above i64"      0 2 "${ok}x = 0o1000000000000000000000"$'\n'
differ "a binary integer above i64"      0 2 "${ok}x = 0b1$(printf '%*s' 63 '' | tr ' ' 0)"$'\n'
differ "fail-fast above i64"             1 2 $'[profile.ci]\nfail-fast = 9223372036854775808\n'
differ "a 4301-digit integer"            2 2 "${ok}x = 1${d4300}"$'\n'
differ "a float that overflows to inf"   0 2 "${ok}x = 1e400"$'\n'
differ "fail-fast = 1e400"               1 2 $'[profile.ci]\nfail-fast = 1e400\n'
printf '%sx = %s\n' "$ok" "$(nest 79 '[' ']')" >"$tmp/c.toml"
compare "arrays nested 79 deep" library 0 "$tmp/c.toml"
differ "arrays nested 80 deep"           0 2 "${ok}x = $(nest 80 '[' ']')"$'\n'
differ "inline tables nested 81 deep"    0 2 "${ok}x = $(nest 81 '{' '}' | sed 's/{/{a=/g')"$'\n'

total=$((pass + fail))
echo "nextest-fail-fast parity: $pass of $total equal (judge: $("$PY" -I -c 'import sys; print(sys.version.split()[0])'))"
if [[ "$fail" -ne 0 || "$total" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: $fail mismatches; $total cases ran, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
