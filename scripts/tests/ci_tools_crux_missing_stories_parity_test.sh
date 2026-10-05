#!/usr/bin/env bash
# ci_tools_crux_missing_stories_parity_test.sh -- the identical-output case table for
# `aprender-ci-tools crux-missing-stories` against scripts/crux_missing_stories.py (#4816).
#
# Each case is the JSON on stdin, the exit code both sides must give (0, or 1 where the
# .py raised) and the number of rows the .py must print, which pins every case to the
# input it was written for: a raise after some rows still prints them. stdout must be
# byte-identical. crux_missing_stories.py is the EXTERNAL VALIDATOR, never a build step.
#
# Not run, by design: input Python's json accepts and serde_json refuses (NaN/Infinity,
# lone surrogates, numbers beyond f64, nesting deeper than 128) and a printed demand_score
# the port cannot render as Python did (a list or object, -0, a magnitude of 2^63 or
# more). The port refuses those with exit 2; see the module doc of crux_missing_stories.rs.
#
# The .py reads stdin as strict UTF-8, as under a UTF-8 locale: under C/POSIX it read
# invalid UTF-8 through.
#
# Not vacuous (L25): the run fails if fewer cases ran than the table declares, and a
# planted lying binary must be reported as exactly one failure before any case counts.
#
# Usage: scripts/tests/ci_tools_crux_missing_stories_parity_test.sh   (CI_TOOLS_BIN=<path> to skip the build)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
CMS="$ROOT/scripts/crux_missing_stories.py"
EXPECTED_CASES=117

. scripts/ci_tools_bin.sh || exit 1
BIN="$CI_TOOLS_BIN"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
pass=0
fail=0
n=0

# run_case NAME WANT_RC WANT_ROWS INPUT ARGS... -> pass/fail on exit codes, row count, bytes
run_case() {
    local name="$1" want="$2" rows="$3" input="$4" py_rc=0 rs_rc=0 ok=1 got
    shift 4
    PYTHONIOENCODING=utf-8:strict "$PY" "$CMS" "$@" <"$input" >"$tmp/py.out" 2>/dev/null || py_rc=$?
    "$BIN" crux-missing-stories "$@" <"$input" >"$tmp/rs.out" 2>/dev/null || rs_rc=$?
    [[ "$py_rc" == "$want" && "$rs_rc" == "$want" ]] || ok=0
    got="$(wc -l <"$tmp/py.out")"
    [[ "$got" -eq "$rows" ]] || ok=0
    cmp -s "$tmp/py.out" "$tmp/rs.out" || ok=0
    if [[ "$ok" == 1 ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "FAIL: $name (want rc $want rows $rows; py rc $py_rc rows $got, rs rc $rs_rc)" >&2
        diff "$tmp/py.out" "$tmp/rs.out" >&2 || true
    fi
}

# c NAME WANT_RC WANT_ROWS JSON [ARGS...] -> one case. JSON goes through printf %b, so a
# JSON escape is written \\u00e9 / \\t and \xNN writes a raw byte.
c() {
    n=$((n + 1))
    printf '%b' "$4" >"$tmp/case$n.json"
    local name="$1" want="$2" rows="$3"
    shift 4
    run_case "$name" "$want" "$rows" "$tmp/case$n.json" "$@"
}

# st ID STATUS -> a story with every field; m SCORE -> a missing story with that score
st() { printf '{"status":"%s","id":"%s","title":"T %s","demand_score":4,"competitor":"C"}' "$2" "$1" "$1"; }
m() { printf '{"status":"missing","id":"A-1","title":"T","demand_score":%s,"competitor":"C"}' "$1"; }
# f KEY VALUE -> a missing story with KEY set to VALUE (raw JSON), or dropped for @drop
f() {
    local k v out='{"status":"missing"'
    for k in id title demand_score competitor; do
        case "$k" in id) v='"A-1"' ;; title) v='"T"' ;; demand_score) v=3 ;; competitor) v='"C"' ;; esac
        [[ "$k" == "$1" ]] && v="$2"
        [[ "$v" == "@drop" ]] || out="$out,\"$k\":$v"
    done
    printf '%s}' "$out"
}
# d N -> an integer literal of N digits
d() { printf '1%.0s' $(seq 1 "$1"); }
OK="$(st A-0 missing)"

# --- the planted lying binary must be caught before any real case counts --------
printf '#!/usr/bin/env bash\nprintf "A-0\\tT A-0\\t5\\tC\\n"\n' >"$tmp/liar"
chmod +x "$tmp/liar"
real_bin="$BIN"
BIN="$tmp/liar"
c "planted lying binary (this one FAIL line is expected)" 0 1 "[$OK]"
if [[ "$fail" != 1 || "$pass" != 0 ]]; then
    echo "FAIL: the planted lying binary was not caught as exactly one failure" >&2
    exit 1
fi
BIN="$real_bin"
pass=0
fail=0
n=0

# --- ROWS: which stories print, in which order -------------------------------------
c "ROWS: one missing story" 0 1 "[$OK]"
c "ROWS: mixed statuses keep input order" 0 3 "[$(st A-1 missing),$(st A-2 done),$(st A-3 missing),$(st A-4 partial),$(st A-5 missing)]"
c "ROWS: none missing" 0 0 "[$(st A-1 done),$(st A-2 partial)]"
c "ROWS: empty list" 0 0 '[]'
c "ROWS: empty object top level iterates nothing" 0 0 '{}'
c "ROWS: empty string top level iterates nothing" 0 0 '""'
c "ROWS: status is case-sensitive" 0 0 "[$(st A-1 Missing),$(st A-2 MISSING)]"
c "ROWS: status with spaces is not missing" 0 0 "[$(st A-1 ' missing')]"
c "ROWS: status non-string" 0 0 '[{"status":1,"id":"a"},{"status":true},{"status":null},{"status":["missing"]}]'
c "ROWS: no status key" 0 0 '[{"id":"A-1","title":"T","demand_score":1,"competitor":"C"}]'
c "ROWS: empty story object" 0 0 '[{}]'
c "ROWS: non-missing story may lack every field" 0 1 "[{\"status\":\"done\"},$OK]"
c "ROWS: extra keys ignored" 0 1 '[{"status":"missing","id":"X","title":"T","demand_score":1,"competitor":"C","extra":[1,{"a":null}]}]'
c "ROWS: duplicate status key, last wins (missing)" 0 1 '[{"status":"done","status":"missing","id":"X","title":"T","demand_score":1,"competitor":"C"}]'
c "ROWS: duplicate status key, last wins (done)" 0 0 '[{"status":"missing","status":"done","id":"X","title":"T","demand_score":1,"competitor":"C"}]'
c "ROWS: duplicate id key, last wins" 0 1 '[{"status":"missing","id":"old","id":"new","title":"T","demand_score":1,"competitor":"C"}]'
c "ROWS: 200 missing stories" 0 200 "[$(for i in $(seq 1 199); do printf '%s,' "$(st "A-$i" missing)"; done)$(st A-200 missing)]"

# --- SCORE: str(demand_score) ------------------------------------------------------
c "SCORE: int" 0 1 "[$(m 5)]"
c "SCORE: zero" 0 1 "[$(m 0)]"
c "SCORE: negative int" 0 1 "[$(m -7)]"
c "SCORE: i64 max" 0 1 "[$(m 9223372036854775807)]"
c "SCORE: i64 min" 0 1 "[$(m -9223372036854775808)]"
c "SCORE: u64 max" 0 1 "[$(m 18446744073709551615)]"
c "SCORE: float 1.0" 0 1 "[$(m 1.0)]"
c "SCORE: float 0.0" 0 1 "[$(m 0.0)]"
c "SCORE: float -1.5" 0 1 "[$(m -1.5)]"
c "SCORE: float 0.1" 0 1 "[$(m 0.1)]"
c "SCORE: float 4.35 (shortest repr)" 0 1 "[$(m 4.35)]"
c "SCORE: float 0.30000000000000004" 0 1 "[$(m 0.30000000000000004)]"
c "SCORE: float 1E2 is 100.0" 0 1 "[$(m 1E2)]"
c "SCORE: float 0.0001 stays fixed" 0 1 "[$(m 0.0001)]"
c "SCORE: float 0.00001 goes e-05" 0 1 "[$(m 0.00001)]"
c "SCORE: float 2.5e-7" 0 1 "[$(m 2.5e-7)]"
c "SCORE: float 1e15 stays fixed" 0 1 "[$(m 1e15)]"
c "SCORE: float 1e16 goes e+16" 0 1 "[$(m 1e16)]"
c "SCORE: float 9.2e18 under 2^63" 0 1 "[$(m 9.2e18)]"
c "SCORE: float 123456.789" 0 1 "[$(m 123456.789)]"
c "SCORE: float max" 0 1 "[$(m 1.7976931348623157e308)]"
c "SCORE: float smallest subnormal" 0 1 "[$(m 5e-324)]"
c "SCORE: float underflow 1e-400 is 0.0" 0 1 "[$(m 1e-400)]"
c "SCORE: float needing exact parse 2.1531120041346774e-5" 0 1 "[$(m 2.1531120041346774e-5)]"
c "SCORE: float 17 digits 1.2345678901234567e16" 0 1 "[$(m 1.2345678901234567e16)]"
c "SCORE: true" 0 1 "[$(m true)]"
c "SCORE: false" 0 1 "[$(m false)]"
c "SCORE: null" 0 1 "[$(m null)]"
c "SCORE: string" 0 1 "[$(m '"high"')]"
c "SCORE: empty string" 0 1 "[$(m '""')]"
c "SCORE: numeric string kept as text" 0 1 "[$(m '"05.0"')]"

# --- TEXT: field bytes pass through ------------------------------------------------
c "TEXT: non-ASCII raw UTF-8" 0 1 '[{"status":"missing","id":"é-1","title":"日本語 ✓","demand_score":1,"competitor":"Ñ"}]'
c "TEXT: unicode escapes" 0 1 '[{"status":"missing","id":"\\u00e9","title":"\\u65e5\\u672c","demand_score":1,"competitor":"\\u2028x"}]'
c "TEXT: surrogate pair escape" 0 1 '[{"status":"missing","id":"\\ud83d\\ude00","title":"T","demand_score":1,"competitor":"C"}]'
c "TEXT: escaped tab inside a field" 0 1 '[{"status":"missing","id":"a\\tb","title":"T","demand_score":1,"competitor":"C"}]'
c "TEXT: escaped newline inside a field" 0 2 '[{"status":"missing","id":"A","title":"line1\\nline2","demand_score":1,"competitor":"C"}]'
c "TEXT: escaped CR inside a field" 0 1 '[{"status":"missing","id":"A","title":"x\\ry","demand_score":1,"competitor":"C"}]'
c "TEXT: escaped NUL inside a field" 0 1 '[{"status":"missing","id":"A","title":"x\\u0000y","demand_score":1,"competitor":"C"}]'
c "TEXT: escaped slash, quote, backslash" 0 1 '[{"status":"missing","id":"a\\/b","title":"\\"q\\"","demand_score":1,"competitor":"\\\\"}]'
c "TEXT: empty id/title/competitor" 0 1 "[$(f id '""' | sed 's/"title":"T"/"title":""/; s/"competitor":"C"/"competitor":""/')]"
c "TEXT: escaped status value" 0 1 '[{"status":"\\u006dissing","id":"A","title":"T","demand_score":1,"competitor":"C"}]'
c "TEXT: control chars 0x7f and U+0085" 0 1 '[{"status":"missing","id":"\x7f","title":"\\u0085","demand_score":1,"competitor":"C"}]'

# --- WS: whitespace between tokens -------------------------------------------------
c "WS: leading and trailing whitespace" 0 1 " \t\n[$OK]\n \r\n"
c "WS: CRLF between tokens" 0 1 "[\r\n$OK\r\n]\r\n"
c "WS: lone CR between tokens" 0 1 "[\r$OK\r]"
c "WS: pretty-printed" 0 1 '[\n  {\n    "status": "missing",\n    "id": "A",\n    "title": "T",\n    "demand_score": 2,\n    "competitor": "C"\n  }\n]\n'

# --- RAISE: where the .py raised (exit 1), after the rows it printed ----------------
c "RAISE: top level is a number" 1 0 '3'
c "RAISE: top level is null" 1 0 'null'
c "RAISE: top level is true" 1 0 'true'
c "RAISE: top level is a non-empty object" 1 0 "{\"a\":$OK}"
c "RAISE: top level is a non-empty string" 1 0 '"missing"'
c "RAISE: story is a number, after a row" 1 1 "[$OK,3]"
c "RAISE: story is a string, after a row" 1 1 "[$OK,\"missing\"]"
c "RAISE: story is null, after a row" 1 1 "[$OK,null]"
c "RAISE: story is a list" 1 0 "[[$OK]]"
c "RAISE: missing story lacks id, after a row" 1 1 "[$OK,$(f id @drop)]"
c "RAISE: missing story lacks title" 1 0 "[$(f title @drop)]"
c "RAISE: missing story lacks demand_score" 1 0 "[$(f demand_score @drop)]"
c "RAISE: missing story lacks competitor" 1 0 "[$(f competitor @drop)]"
c "RAISE: id is an int" 1 0 "[$(f id 7)]"
c "RAISE: title is null" 1 0 "[$(f title null)]"
c "RAISE: competitor is a list" 1 0 "[$(f competitor '["C"]')]"
c "RAISE: id is a bool, after two rows" 1 2 "[$OK,$OK,$(f id true)]"
c "RAISE: competitor missing AND score is a list" 1 0 "[$(f competitor @drop | sed 's/"demand_score":3/"demand_score":[1]/')]"
c "RAISE: id an int AND score -0" 1 0 "[$(f id 1 | sed 's/"demand_score":3/"demand_score":-0/')]"
c "RAISE: NaN, then the list is not closed" 1 0 '[{"x":NaN}'
c "RAISE: NaN, then a trailing comma" 1 0 '[{"x":NaN},]'
c "RAISE: -NaN is not a constant" 1 0 '[{"x":-NaN}]'
c "RAISE: +Infinity is not a constant" 1 0 '[{"x":+Infinity}]'
c "RAISE: NaN then .5 is not a number" 1 0 '[{"x":NaN.5}]'
c "RAISE: Infinity then e5 is not a number" 1 0 '[{"x":Infinitye5}]'
c "RAISE: -Infinity then .5 is not a number" 1 0 '[{"x":-Infinity.5}]'
c "RAISE: lone surrogate, then a trailing comma" 1 0 '[{"x":"\\ud800"},]'
c "RAISE: 1e400, then garbage" 1 0 '[{"x":1e400} x]'
c "RAISE: an integer of 4301 digits" 1 0 "[{\"x\":$(d 4301)}]"
c "RAISE: an integer of 4301 digits, negative" 1 0 "[{\"x\":-$(d 4301)}]"
c "RAISE: an integer of 4301 digits after a missing story prints nothing" 1 0 "[$OK,{\"x\":$(d 4301)}]"

# --- BADJSON: invalid JSON or UTF-8 is a raise, nothing printed ---------------------
c "BADJSON: empty stdin" 1 0 ''
c "BADJSON: whitespace only" 1 0 ' \n'
c "BADJSON: unclosed list" 1 0 "[$OK"
c "BADJSON: trailing data" 1 0 "[$OK] x"
c "BADJSON: two documents" 1 0 "[] []"
c "BADJSON: trailing comma" 1 0 "[$OK,]"
c "BADJSON: leading zero" 1 0 '[01]'
c "BADJSON: single quotes" 1 0 "[{'status':'missing'}]"
c "BADJSON: UTF-8 BOM" 1 0 "\xef\xbb\xbf[$OK]"
c "BADJSON: invalid UTF-8 byte" 1 0 "[$OK,\"\xff\"]"
c "BADJSON: raw tab inside a string" 1 0 '[{"status":"missing","id":"a\tb","title":"T","demand_score":1,"competitor":"C"}]'
c "BADJSON: lowercase nan" 1 0 '[nan]'
c "BADJSON: lowercase infinity" 1 0 '[infinity]'
c "BADJSON: None" 1 0 '[None]'
c "BADJSON: bad escape" 1 0 '["\\x41"]'
c "BADJSON: short unicode escape" 1 0 '["\\u12"]'

# --- ARGV: the .py read no argument; every one is ignored --------------------------
c "ARGV: --help ignored" 0 1 "[$OK]" --help
c "ARGV: -h ignored" 0 1 "[$OK]" -h
c "ARGV: --version ignored" 0 1 "[$OK]" --version
c "ARGV: lone -- ignored" 0 1 "[$OK]" --
c "ARGV: two words ignored" 0 1 "[$OK]" stories.json extra
c "ARGV: - ignored" 0 1 "[$OK]" -
c "ARGV: --help on bad JSON still raises" 1 0 '[' --help

ran=$((pass + fail))
if [[ "$ran" -ne "$EXPECTED_CASES" || "$n" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran of $n built cases; the table declares $EXPECTED_CASES" >&2
    exit 1
fi
echo "crux-missing-stories parity: $pass/$ran identical"
[[ "$fail" -eq 0 ]]
