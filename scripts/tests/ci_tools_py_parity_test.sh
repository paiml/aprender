#!/usr/bin/env bash
# ci_tools_py_parity_test.sh -- PY-PORT-2: the identical-output case table for the
# first three ports into crates/aprender-ci-tools (C301: no Python in the build).
#
# Every case runs through BOTH the .py original and the Rust subcommand; stdout must
# be byte-identical and both must agree on success vs failure. The .py files are the
# EXTERNAL VALIDATOR here (C301 allows that), never a build step.
#
#   publishable-crates      vs scripts/lib/publishable_crates.py
#   package-include-diff    vs its frozen golden outputs (the .py is deleted, callers switched)
#   coverage-report-scope   vs scripts/coverage_report_scope.py
#   tarball-build-errors    vs scripts/lib/tarball_build_errors.py
#
# coverage_report_scope.py runs `cargo metadata` itself, so its fixture cases put a
# fake `cargo` first on PATH that prints the fixture; both sides then read the same
# bytes. Live cases run each pair on this checkout's real `cargo metadata`.
#
# Not checked here: the argv forms where clap's surface deliberately differs from the
# originals' hand-rolled loops (--help/--version, --exclude=NAME, extra positionals).
# crates/aprender-ci-tools/README.md lists them; no caller uses any of them.
#
# Not vacuous (L25): the run fails if fewer cases ran than the table declares, and a
# planted mismatch must be reported as one before any real case counts.
#
# Usage: scripts/tests/ci_tools_py_parity_test.sh   (CI_TOOLS_BIN=<path> to skip the build)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
EXPECTED_CASES=78

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

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
pass=0
fail=0

# run_side OUT STDIN -- CMD...: stdout to OUT, return the command's status class (0/1).
run_side() {
    local out="$1" in="$2"
    shift 3
    if "$@" <"$in" >"$out" 2>/dev/null; then
        echo 0
    else
        echo 1
    fi
}

# check NAME STDIN PYCMD... -- RSCMD...
check() {
    local name="$1" in="$2"
    shift 2
    local py=() rs=()
    while [[ "$1" != "--" ]]; do
        py+=("$1")
        shift
    done
    shift
    rs=("$@")
    local py_status rs_status
    py_status="$(run_side "$tmp/py.out" "$in" -- "${py[@]}")"
    rs_status="$(run_side "$tmp/rs.out" "$in" -- "${rs[@]}")"
    if [[ "$py_status" == "$rs_status" ]] && cmp -s "$tmp/py.out" "$tmp/rs.out"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: $name (py status $py_status, rust status $rs_status)" >&2
        diff <(od -c "$tmp/py.out") <(od -c "$tmp/rs.out") 2>&1 | head -8 >&2 || true
    fi
}

: >"$tmp/empty"

# --- The harness must see a planted mismatch, or its passes mean nothing. ---------
printf 'a\n' >"$tmp/plant"
before=$fail
check "plant" "$tmp/plant" cat -- printf 'b\n' 2>/dev/null
if [[ "$fail" -ne "$((before + 1))" ]]; then
    echo "FAIL: the planted mismatch was not reported; the harness is blind" >&2
    exit 1
fi
fail="$before"

# --- 1. publishable-crates -------------------------------------------------------
PC_PY=("$PY" scripts/lib/publishable_crates.py)
PC_RS=("$BIN" publishable-crates)
pkg() { # NAME PUBLISH_JSON_OR_EMPTY
    local p=""
    [[ -n "$2" ]] && p=",\"publish\":$2"
    printf '{"name":"%s","manifest_path":"/w/crates/%s/Cargo.toml"%s}' "$1" "$1" "$p"
}
i=0
for publish in '' 'null' '[]' '["crates-io"]' '["other-registry"]' 'false'; do
    i=$((i + 1))
    printf '{"packages":[%s]}' "$(pkg a-b "$publish")" >"$tmp/pc$i.json"
    check "publishable-crates publish=${publish:-absent}" "$tmp/pc$i.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
done
printf '{"packages":[%s,{"name":"aprender","manifest_path":"/w/Cargo.toml"},%s,{"name":"x","manifest_path":"Cargo.toml"}]}' \
    "$(pkg z '')" "$(pkg a '[]')" >"$tmp/pc-order.json"
check "publishable-crates order+root+bare" "$tmp/pc-order.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
printf '{"packages":[]}' >"$tmp/pc-zero.json"
check "publishable-crates zero packages" "$tmp/pc-zero.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
# `for pkg in {}` / `in ""` runs zero times: rc 0, empty stdout (quorum r5 finding).
printf '{"packages":{}}' >"$tmp/pc-zero-obj.json"
check "publishable-crates packages={}" "$tmp/pc-zero-obj.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
printf '{"packages":""}' >"$tmp/pc-zero-str.json"
check "publishable-crates packages=\"\"" "$tmp/pc-zero-str.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
n=0
for bad in '{' '' '[]' '{"packages":[{"name":"x"}]}' '{"packages":[{"manifest_path":"/x/Cargo.toml"}]}' \
    '{"packages":[{"name":1,"manifest_path":"/x/Cargo.toml"}]}' '{"packages":{"a":1}}' '{"packages":"ab"}' \
    '{"packages":null}' '{"packages":5}' '{"packages":[5]}'; do
    n=$((n + 1))
    printf '%s' "$bad" >"$tmp/pc-bad$n.json"
    check "publishable-crates refuses #$n" "$tmp/pc-bad$n.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
done
# A bad package AFTER good ones: the original prints as it goes, so the good lines
# must be on stdout before the failure, on both sides.
printf '{"packages":[%s,{"name":"x"},%s]}' "$(pkg a '')" "$(pkg c '')" >"$tmp/pc-late1.json"
check "publishable-crates good then no manifest_path" "$tmp/pc-late1.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
printf '{"packages":[%s,%s,{"manifest_path":"/w/y/Cargo.toml"}]}' "$(pkg a '')" "$(pkg b '[]')" >"$tmp/pc-late2.json"
check "publishable-crates good then no name" "$tmp/pc-late2.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
cargo metadata --no-deps --format-version 1 >"$tmp/live-meta.json"
check "publishable-crates LIVE" "$tmp/live-meta.json" "${PC_PY[@]}" -- "${PC_RS[@]}"
if [[ "$("${PC_RS[@]}" <"$tmp/live-meta.json" | wc -l)" -lt 1 ]]; then
    echo "FAIL: live publishable-crates printed 0 packages (vacuous)" >&2
    fail=$((fail + 1))
fi

# --- 2. package-include-diff -----------------------------------------------------
# The .py original is DELETED (callers switched); its outputs for every row below were
# captured from it before removal and are frozen here as the expected stdout. A row is
# `printf '%s' EXPECTED` on the oracle side, so `check` still compares bytes + status.
pid_case() { # NAME LISTING_PRINTF INCLUDES_PRINTF EXPECTED_STDOUT
    printf "$2" >"$tmp/l"
    printf "$3" >"$tmp/i"
    check "package-include-diff $1" "$tmp/empty" \
        printf '%s' "$4" -- \
        "$BIN" package-include-diff "$tmp/l" "$tmp/i"
}
pid_case "all present" 'a.rs\nb.rs\n' 'a.rs\tm.rs\nb.rs\tm.rs\n' ''
pid_case "one missing" 'a.rs\n' 'a.rs\tm.rs\nb.rs\tn.rs\n' $'b.rs\tn.rs\n'
pid_case "empty listing" '' 'a.rs\tm.rs\n' $'a.rs\tm.rs\n'
pid_case "empty includes" 'a.rs\n' '' ''
pid_case "duplicates kept" '' 'x\ty\nx\ty\n' $'x\ty\nx\ty\n'
pid_case "no tab" '' 'lonely\n' $'lonely\t\n'
pid_case "second tab" '' 't\ts\tz\n' $'t\ts\tz\n'
pid_case "CRLF" 'a.rs\r\n' 'a.rs\tm.rs\r\nq\tr\r\n' $'q\tr\n'
pid_case "lone CR" 'a.rs\rb.rs' 'b.rs\tm\ra.rs\tm\rc\td' $'c\td\n'
pid_case "blank includes" '' '\n\n' ''
pid_case "whitespace include row" '' '  \n' $'  \t\n'
pid_case "whitespace listing" '  \n\x1c\n' '  \tm\n\x1c\tn\n' $'  \tm\n\x1c\tn\n'
pid_case "inner spaces" ' a.rs \n' ' a.rs \tm\n' ''
pid_case "no final newline" 'a.rs' 'a.rs\tm' ''
pid_case "bad utf8" '\xff.rs\n' '\xff.rs\tm\n\xfe\tq\n\xe2\x82\tr\n' $'\xef\xbf\xbd\tq\n\xef\xbf\xbd\tr\n'
# A listing that cannot be opened: the original raised (status 1, empty stdout).
check "package-include-diff missing listing" "$tmp/empty" \
    false -- \
    "$BIN" package-include-diff "$tmp/nope" "$tmp/i"
git ls-files crates/aprender-ci-tools >"$tmp/live-l"
{ sed 's/$/\tsrc.rs/' "$tmp/live-l"; printf 'crates/aprender-ci-tools/src/planted.rs\tlib.rs\n'; } >"$tmp/live-i"
check "package-include-diff LIVE+plant" "$tmp/empty" \
    printf '%s' $'crates/aprender-ci-tools/src/planted.rs\tlib.rs\n' -- \
    "$BIN" package-include-diff "$tmp/live-l" "$tmp/live-i"
if ! "$BIN" package-include-diff "$tmp/live-l" "$tmp/live-i" | grep -q 'planted.rs'; then
    echo "FAIL: the planted missing include was not reported" >&2
    fail=$((fail + 1))
fi

# --- 3. coverage-report-scope (fake cargo on PATH serves the fixture) -------------
mkdir -p "$tmp/fakebin"
printf '#!/usr/bin/env bash\nif [[ -n "${FAKE_CARGO_FAIL:-}" ]]; then exit 101; fi\ncat "$FAKE_META"\n' >"$tmp/fakebin/cargo"
chmod +x "$tmp/fakebin/cargo"
printf '{"packages":[{"name":"b"},{"name":"aprender-gpu"},{"name":"a-z"},{"name":"A"},{"name":"--exclude"}]}' >"$tmp/crs.json"
crs_case() { # NAME ARGS...
    local name="$1"
    shift
    check "coverage-report-scope $name" "$tmp/empty" \
        env PATH="$tmp/fakebin:$PATH" FAKE_META="$tmp/crs.json" "$PY" scripts/coverage_report_scope.py "$@" -- \
        env PATH="$tmp/fakebin:$PATH" FAKE_META="$tmp/crs.json" "$BIN" coverage-report-scope "$@"
}
crs_case "no args"
crs_case "exclude one" --exclude aprender-gpu
crs_case "exclude two" --exclude aprender-gpu --exclude b
crs_case "exclude dup" --exclude b --exclude b
crs_case "exclude unknown" --exclude nope
crs_case "exclude last (empty)" --exclude
crs_case "exclude hyphen value" --exclude --exclude
crs_case "exclude all" --exclude A --exclude a-z --exclude aprender-gpu --exclude b --exclude --exclude
crs_case "positional" stray
crs_case "unknown flag" --frob
check "coverage-report-scope cargo fails" "$tmp/empty" \
    env PATH="$tmp/fakebin:$PATH" FAKE_META="$tmp/crs.json" FAKE_CARGO_FAIL=1 "$PY" scripts/coverage_report_scope.py -- \
    env PATH="$tmp/fakebin:$PATH" FAKE_META="$tmp/crs.json" FAKE_CARGO_FAIL=1 "$BIN" coverage-report-scope
check "coverage-report-scope LIVE" "$tmp/empty" "$PY" scripts/coverage_report_scope.py -- "$BIN" coverage-report-scope
check "coverage-report-scope LIVE exclude gpu" "$tmp/empty" \
    "$PY" scripts/coverage_report_scope.py --exclude aprender-gpu -- "$BIN" coverage-report-scope --exclude aprender-gpu

# --- 5. tarball-build-errors ---
# The caller (scripts/package_tarball_build.sh) branches on the exact exit code (0 clean,
# 1 crate RED, 2 bad input, 3 unowned errors only, 4 build host failed), so every case
# compares the exact code before the stdout. The .py stays in the tree (N-1: its caller is
# on the publish gate path), so it is run from the tree.
TBE_PY=("$PY" scripts/lib/tarball_build_errors.py)
check_tbe() { # NAME ARGS... (the same args go to both sides)
    local name="$1" prc rrc
    shift
    "${TBE_PY[@]}" "$@" </dev/null >/dev/null 2>&1 && prc=0 || prc=$?
    "$BIN" tarball-build-errors "$@" </dev/null >/dev/null 2>&1 && rrc=0 || rrc=$?
    if [[ "$prc" -ne "$rrc" ]]; then
        fail=$((fail + 1))
        echo "MISMATCH: tarball-build-errors $name exit code (py $prc, rust $rrc)" >&2
        return
    fi
    check "tarball-build-errors $name (rc $prc)" "$tmp/empty" \
        "${TBE_PY[@]}" "$@" -- "$BIN" tarball-build-errors "$@"
}
t="$tmp/tbe"
mkdir -p "$t/adir"
D='/w/ws/pkgs'
printf '   Compiling a v1.0.0\nwarning: unused variable `x` (error is not at line start)\n' >"$t/clean.log"
printf '\n' >"$t/nl.log"
{
    printf '%s/a-1.0.0/src/lib.rs:1:2: error: x\n' "$D"
    printf 'error: failed to write `/t/x.rlib`: No space left on device (os error 28)\n'
} >"$t/host1.log"
{
    printf 'e1 No space left on device\ne2 os error 28\ne3 failed to write `f`\ne4 Disk quota exceeded\n'
    printf 'e5 (signal: 9, SIGKILL: kill)\ne6 process didn'"'"'t exit successfully (signal: 9)\ne7 Cannot allocate memory\n'
} >"$t/host7.log"
{
    printf '%s/a-b-0.70.0-rc.1/src/lib.rs:3:4: error[E0425]: cannot find value `y`\n' "$D"
    printf '%s/a-b-0.70.0-rc.1/tests/t.rs:9:1: error: mismatched types\n' "$D"
    printf 'error: could not compile `a` (lib) due to 1 previous error\n'
    printf 'error: could not compile `a-b` (lib test) due to 2 previous errors\n'
    printf 'error: could not compile `a-b` (test "t")\n'
} >"$t/prefix.log"
printf 'pkgs/zz-2.0.0/src/m.rs:10:5: error: no owner reported\n' >"$t/noowner.log"
for i in $(seq 1 25); do printf '%s/big-1.0.0/src/lib.rs:%s:1: error: e%s\n' "$D" "$i" "$i"; done >"$t/cap20.log"
printf 'error: could not compile `only` (bin "only")\nerror: could not compile `two` (lib)\n' >"$t/failedonly.log"
printf 'error: unexpected argument '"'"'--bogus'"'"' found\nerror: aborting\n' >"$t/unowned.log"
{
    for i in $(seq 1 12); do printf 'error: other %s\n' "$i"; done
    printf 'error: could not compile `c` (lib)\n'
} >"$t/cap10.log"
printf '%s/k-1.0.0/src/a.rs:1:1: error: A\r%s/k-1.0.0/src/b.rs:2:2: error: B\x0b%s/k-1.0.0/src/c.rs:3:3: error: C\x0cerror: tail\x1cerror: fs\r\n' "$D" "$D" "$D" >"$t/breaks.log"
printf '%s/u-1.0.0/src/a.rs:1:1: error: L\xe2\x80\xa8%s/u-1.0.0/src/b.rs:2:2: error: P\xc2\x85error: nel\n' "$D" "$D" >"$t/ubreaks.log"
printf '%s/v-1.0.0/src/a.rs:1:1: error: bad \xff\xfe utf8 \xe2\x82\n' "$D" >"$t/badutf8.log"
printf '%s/s-1.0.0/src/a\x1fb.rs:1:1: error: unit-separator in path\n' "$D" >"$t/us.log"
printf '%s/w-1.0.0/src/a.rs:1:1: error[E-1]: not a word code\n%s/w-1.0.0/src/b.rs:2:2: error[]: empty code\n' "$D" "$D" >"$t/code.log"
printf '/x/pkgs/outer-1.0.0/pkgs/inner-2.0.0/src/a.rs:4:4: error: nested\nerror: could not compile `inner` (lib)\n' >"$t/nested.log"
printf '%s/é-crate-1.0.0/src/a.rs:1:1: error: non-ascii\nerror: could not compile `é` (lib)\nerror: could not compile `é-crate` (lib)\n' "$D" >"$t/unicode.log"
check_tbe "clean" "$t/clean.log"
check_tbe "newline only" "$t/nl.log"
check_tbe "empty log" "$tmp/empty"
check_tbe "missing log" "$t/missing.log"
check_tbe "directory" "$t/adir"
check_tbe "no args"
check_tbe "two args" "$t/clean.log" "$t/clean.log"
check_tbe "hyphen arg" "-x"
check_tbe "double dash" "--"
check_tbe "host beats crate" "$t/host1.log"
check_tbe "host 7 patterns, cap 5" "$t/host7.log"
check_tbe "longest prefix, rc version" "$t/prefix.log"
check_tbe "no owner" "$t/noowner.log"
check_tbe "cap 20" "$t/cap20.log"
check_tbe "failed only" "$t/failedonly.log"
check_tbe "unowned only" "$t/unowned.log"
check_tbe "cap 10 + attributed" "$t/cap10.log"
check_tbe "line breaks" "$t/breaks.log"
check_tbe "unicode line breaks" "$t/ubreaks.log"
check_tbe "invalid utf-8" "$t/badutf8.log"
check_tbe "unit separator in path" "$t/us.log"
check_tbe "error codes" "$t/code.log"
check_tbe "nested pkgs" "$t/nested.log"
check_tbe "non-ascii crate" "$t/unicode.log"

ran=$((pass + fail))
echo "ci_tools_py_parity: $pass/$ran identical (declared $EXPECTED_CASES)"
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
