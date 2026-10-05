#!/usr/bin/env bash
# ci_tools_py_parity_test.sh -- PY-PORT-2: the identical-output case table for the
# ports into crates/aprender-ci-tools (C301: no Python in the build).
#
# Every case runs through BOTH the .py original and the Rust subcommand; stdout must
# be byte-identical and both must agree on success vs failure. The .py files are the
# EXTERNAL VALIDATOR here (C301 allows that), never a build step.
#
#   publishable-crates      vs scripts/lib/publishable_crates.py
#   package-include-diff    vs its frozen golden outputs (the .py is deleted, callers switched)
#   coverage-report-scope   vs scripts/coverage_report_scope.py
#   tarball-shrink-report   vs scripts/lib/tarball_shrink_report.py
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
EXPECTED_CASES=63

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

# --- 4. tarball-shrink-report ---
# The original refuses a missing input with exit 2, not 1, and the script documents it,
# so these cases also compare the exact exit code.
check_rc() { # NAME -- PYARGS... (the same args go to both sides)
    local name="$1" prc rrc
    shift
    "$PY" scripts/lib/tarball_shrink_report.py "$@" </dev/null >/dev/null 2>&1 && prc=0 || prc=$?
    "$BIN" tarball-shrink-report "$@" </dev/null >/dev/null 2>&1 && rrc=0 || rrc=$?
    if [[ "$prc" -ne "$rrc" ]]; then
        fail=$((fail + 1))
        echo "MISMATCH: tarball-shrink-report $name exit code (py $prc, rust $rrc)" >&2
        return
    fi
    check "tarball-shrink-report $name" "$tmp/empty" \
        "$PY" scripts/lib/tarball_shrink_report.py "$@" -- "$BIN" tarball-shrink-report "$@"
}
w="$tmp/tsr"
mkdir -p "$w/empty/pkgs" "$w/full/pkgs/a/tests/deep" "$w/full/pkgs/b-crate/src" "$w/full/pkgs/c/src" \
    "$w/full/pkgs/a-very-long-crate-name-past-thirty-two-columns/src" "$w/elsewhere" "$w/bad/pkgs/z/x.rs"
: >"$w/empty.log"
cat >"$w/full.log" <<'LOG'
warning: ignoring test `pre` as `tests/pre.rs` is not included in the published package
   Packaging a v0.1.0 (/w/a)
warning: ignoring test `t1` as `tests/t1.rs` is not included in the published package
warning: ignoring test `t2` as `tests/t2.rs` is not included in the published package, trailing
 warning: ignoring test `t3` as `tests/t3.rs` is not included in the published package
warning: ignoring test `` as `tests/t4.rs` is not included in the published package
Packaging nope v1
LOG
printf '\tPackaging b-crate v2\n' >>"$w/full.log"
for i in 1 2 3 4 5 6 7; do
    printf 'warning: ignoring test `m%s` as `tests/m%s.rs` is not included in the published package\r\n' "$i" "$i" >>"$w/full.log"
done
printf '  Packaging a v0.1.0\fwarning: ignoring test `ff` as `tests/ff.rs` is not included in the published package\n\xff\n' >>"$w/full.log"
printf 'fn path_or_skip(p: &str) {}\nlet a = path_or_skip("x"); let b = y_or_skip(1);\n_or_skip(z)\n' >"$w/full/pkgs/a/tests/t.rs"
printf 'x_or_skip(1)\rx_or_skip(2)\fx_or_skip(3)\n\xff a_or_skip(4)\n' >"$w/full/pkgs/a/tests/deep/crlf.rs"
printf 'h_or_skip(1)\n' >"$w/full/pkgs/a/.hidden.rs"
printf 'h_or_skip(1)\n' >"$w/full/pkgs/a/notes.txt"
printf 'pub fn dir_or_skip (p) {}\n  dir_or_skip(p);\n' >"$w/full/pkgs/b-crate/src/lib.rs"
printf 'fn main() {}\n' >"$w/full/pkgs/c/src/main.rs"
printf 'q_or_skip()\n' >"$w/full/pkgs/a-very-long-crate-name-past-thirty-two-columns/src/lib.rs"
printf 'q_or_skip()\n' >"$w/full/pkgs/stray-file.rs"
printf 'q_or_skip()\nq_or_skip()\n' >"$w/elsewhere/lib.rs"
ln -sfn "$w/elsewhere" "$w/full/pkgs/a/linked"
ln -sfn "$w/elsewhere" "$w/full/pkgs/linked-crate"
check_rc "empty log, empty pkgs" "$w/empty.log" "$w/empty"
check_rc "full fixture" "$w/full.log" "$w/full"
check_rc "missing log" "$w/nope.log" "$w/full"
check_rc "log is a directory" "$w/full" "$w/full"
check_rc "missing pkgs" "$w/full.log" "$w/elsewhere"
check_rc "unnormalised paths in the refusal" "$w//./nope.log" "$w/./elsewhere/"
check_rc "bad input: a directory named x.rs" "$w/full.log" "$w/bad"
check "tarball-shrink-report one argument" "$tmp/empty" \
    "$PY" scripts/lib/tarball_shrink_report.py "$w/full.log" -- "$BIN" tarball-shrink-report "$w/full.log"
if ! "$BIN" tarball-shrink-report "$w/full.log" "$w/full" | grep -q '^SHRINK: 11 integration'; then
    echo "FAIL: the planted not-shipped targets were not counted" >&2
    fail=$((fail + 1))
fi
mkdir -p "$w/live"
ln -sfn "$ROOT/crates" "$w/live/pkgs"
check_rc "LIVE crates/ as pkgs" "$w/empty.log" "$w/live"
if ! "$BIN" tarball-shrink-report "$w/empty.log" "$w/live" | grep -q '^SKIP SITES'; then
    echo "FAIL: live tarball-shrink-report found no skip site (vacuous)" >&2
    fail=$((fail + 1))
fi

ran=$((pass + fail))
echo "ci_tools_py_parity: $pass/$ran identical (declared $EXPECTED_CASES)"
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
