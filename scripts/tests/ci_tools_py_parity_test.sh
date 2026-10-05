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
#   coverage-report-scope   vs scripts/coverage_report_scope.py (deleted from the tree once its
#                           callers moved to the bin; read back from git by its pinned blob)
#   tarball-shrink-report   vs scripts/lib/tarball_shrink_report.py
#   tarball-workspace       vs scripts/lib/tarball_workspace.py, read from git (the file is
#                           deleted; its callers are switched). Its written Cargo.toml is
#                           compared too, mapping only the one header line that names it.
#   tarball-build-errors    vs scripts/lib/tarball_build_errors.py
#   llama-fit-verdict       vs scripts/lib/llama_fit_verdict.py (kept: model_ladder.sh's
#                           certification path still calls it), exact exit code too, run
#                           under CPython 3.12/3.13 (LFV_PYTHON=) whose Unicode it follows.
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
EXPECTED_CASES=201

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

# The deleted original, byte-for-byte as it last stood on main (d326bc1fb7). A clone that
# lacks the blob (shallow) fails here rather than skipping the section.
CRS_PY_BLOB=106561a2bfce54484d816665c797de5ea6335910
CRS_PY="$tmp/coverage_report_scope.py"
if ! git cat-file blob "$CRS_PY_BLOB" >"$CRS_PY" 2>/dev/null; then
    echo "FAIL: blob $CRS_PY_BLOB (scripts/coverage_report_scope.py) not in this clone; fetch full history" >&2
    exit 1
fi
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
if ! pid_out=$("$BIN" package-include-diff "$tmp/live-l" "$tmp/live-i") || ! grep -q 'planted.rs' <<<"$pid_out"; then
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
        env PATH="$tmp/fakebin:$PATH" FAKE_META="$tmp/crs.json" "$PY" "$CRS_PY" "$@" -- \
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
    env PATH="$tmp/fakebin:$PATH" FAKE_META="$tmp/crs.json" FAKE_CARGO_FAIL=1 "$PY" "$CRS_PY" -- \
    env PATH="$tmp/fakebin:$PATH" FAKE_META="$tmp/crs.json" FAKE_CARGO_FAIL=1 "$BIN" coverage-report-scope
check "coverage-report-scope LIVE" "$tmp/empty" "$PY" "$CRS_PY" -- "$BIN" coverage-report-scope
check "coverage-report-scope LIVE exclude gpu" "$tmp/empty" \
    "$PY" "$CRS_PY" --exclude aprender-gpu -- "$BIN" coverage-report-scope --exclude aprender-gpu

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
if ! tsr_out=$("$BIN" tarball-shrink-report "$w/full.log" "$w/full") || ! grep -q '^SHRINK: 11 integration' <<<"$tsr_out"; then
    echo "FAIL: the planted not-shipped targets were not counted" >&2
    fail=$((fail + 1))
fi
mkdir -p "$w/live"
ln -sfn "$ROOT/crates" "$w/live/pkgs"
check_rc "LIVE crates/ as pkgs" "$w/empty.log" "$w/live"
if ! live_out=$("$BIN" tarball-shrink-report "$w/empty.log" "$w/live") || ! grep -q '^SKIP SITES' <<<"$live_out"; then
    echo "FAIL: live tarball-shrink-report found no skip site (vacuous)" >&2
    fail=$((fail + 1))
fi

# --- 4. tarball-workspace (its .py is deleted: the validator is read from git) ---------
# Pinned by BLOB id, not by commit: the blob is in main's history (every commit that carried
# scripts/lib/tarball_workspace.py), so any full clone has it whichever way a branch merges.
TW_PY_BLOB=86d653ebf2c2b678a40dca40ab144c91a9cf0bb2   # scripts/lib/tarball_workspace.py
TW_COMPAT_BLOB=9eca16fb1c018730f3c002d22ef888ec3054e20e # scripts/lib/toml_compat.py
mkdir -p "$tmp/pyval" "$tmp/twfix"
if ! git cat-file -e "$TW_PY_BLOB" 2>/dev/null || ! git cat-file -e "$TW_COMPAT_BLOB" 2>/dev/null; then
    echo "FAIL: not measured: the original's blobs are not in this clone (shallow? fetch main's history)" >&2
    exit 1
fi
git cat-file blob "$TW_PY_BLOB" >"$tmp/pyval/tarball_workspace.py"
git cat-file blob "$TW_COMPAT_BLOB" >"$tmp/pyval/toml_compat.py"
# toml_compat's 3.10 fallback reads only single-line [package] strings; the validator is
# the real TOML parser, so an interpreter without tomllib cannot validate.
if ! "$PY" -c 'import tomllib' 2>/dev/null; then
    echo "FAIL: not measured: $PY has no tomllib (needs >= 3.11; set PYTHON=)" >&2
    exit 1
fi
TW_PY=("$PY" "$tmp/pyval/tarball_workspace.py")
TW_RS=("$BIN" tarball-workspace)
TW_CWD="$ROOT"
# tw-side.sh WS FIXTURE OUT CWD CMD...: a fresh copy of FIXTURE at WS, run CMD in CWD, then
# keep WS/Cargo.toml as OUT (ABSENT when nothing was written). Both sides see the same path.
cat >"$tmp/tw-side.sh" <<'EOF'
#!/usr/bin/env bash
ws="$1" fix="$2" out="$3" cwd="$4"
shift 4
rm -rf "${ws:?}"
cp -a "$fix" "$ws"
cd "$cwd" || exit 1
"$@"
rc=$?
if [[ -f "$ws/Cargo.toml" ]]; then cp "$ws/Cargo.toml" "$out"; else printf 'ABSENT' >"$out"; fi
exit "$rc"
EOF
chmod +x "$tmp/tw-side.sh"
TW_PY_HEAD='# Generated by scripts/lib/tarball_workspace.py '
TW_RS_HEAD='# Generated by aprender-ci-tools tarball-workspace '
tw_case() { # NAME STDIN FIXTURE ARGS... (ARGS may name $tmp/ws, the per-side copy)
    local name="$1" in="$2" fix="$3"
    shift 3
    local before="$pass"
    check "tarball-workspace $name" "$in" \
        "$tmp/tw-side.sh" "$tmp/ws" "$fix" "$tmp/tw-py.toml" "$TW_CWD" "${TW_PY[@]}" "$@" -- \
        "$tmp/tw-side.sh" "$tmp/ws" "$fix" "$tmp/tw-rs.toml" "$TW_CWD" "${TW_RS[@]}" "$@"
    # The written Cargo.toml is output too: identical but for the one documented header line.
    if [[ "$pass" -gt "$before" ]] &&
        ! cmp -s <(sed "1s|^$TW_PY_HEAD|$TW_RS_HEAD|" "$tmp/tw-py.toml") "$tmp/tw-rs.toml"; then
        pass=$((pass - 1))
        fail=$((fail + 1))
        echo "MISMATCH: tarball-workspace $name (written Cargo.toml)" >&2
        diff "$tmp/tw-py.toml" "$tmp/tw-rs.toml" | head -8 >&2 || true
    fi
}
twfix() { # FIXTURE DIR MANIFEST: one crate dir under FIXTURE/pkgs
    mkdir -p "$tmp/twfix/$1/pkgs/$2"
    printf '%b' "$3" >"$tmp/twfix/$1/pkgs/$2/Cargo.toml"
}
twfix multi b-0.70.0-rc.1 '[package]\nname = "b"\nversion = "0.70.0-rc.1"\n'
twfix multi a-1.0.0 "# c\n[package]\nname = 'a' # comment\nversion = \"1.0.0\"\n[package.metadata.x]\nname = \"no\"\n[[bin]]\nname = \"zz\"\n"
twfix multi Z-2 '[package]\nversion = "2"\nname = "Z"\n'
twfix multi 'ä-1' '[package]\nname = "ae"\nversion = "1"\n'
mkdir -p "$tmp/twfix/multi/pkgs/no-manifest" "$tmp/twfix/multi/pkgs/dir-manifest/Cargo.toml"
printf 'x' >"$tmp/twfix/multi/pkgs/stray.txt"
ln -sfn a-1.0.0 "$tmp/twfix/multi/pkgs/c-link"
printf 'stale = 1\n' >"$tmp/twfix/multi/Cargo.toml"
mkdir -p "$tmp/twfix/nopkgs" "$tmp/twfix/empty/pkgs/no-manifest"
mkdir -p "$tmp/twfix/pkgsfile" && printf 'x' >"$tmp/twfix/pkgsfile/pkgs"
twfix badtoml a-1 '[package]\nname = "a"\nversion = "1"\n'
twfix badtoml b-1 '[package]\nname = "b\nversion = "1"\n'
twfix noversion a-1 '[package]\nname = "a"\nversion = "1"\n'
twfix noversion b-1 '[package]\nname = "b"\n'
twfix nopackage a-1 '[lib]\nname = "a"\n'
twfix badutf8 a-1 '[package]\nname = "\xff"\nversion = "1"\n'
twfix dupkey a-1 '[package]\nname = "a"\nname = "b"\nversion = "1"\n'
twfix noname a-1 '[package]\nversion = "1"\n'
tw_case "DIR multi (order, skips, symlink, overwrite)" "$tmp/empty" "$tmp/twfix/multi" "$tmp/ws"
if [[ "$(grep -c . "$tmp/rs.out")" -ne 5 ]] || ! grep -q "^\[patch.crates-io\]" "$tmp/tw-rs.toml"; then
    echo "FAIL: the multi fixture did not yield 5 rows and a written workspace (vacuous)" >&2
    fail=$((fail + 1))
fi
tw_case "DIR trailing slash and dot parts" "$tmp/empty" "$tmp/twfix/multi" "$tmp//ws/./"
TW_CWD="$tmp" tw_case "DIR relative" "$tmp/empty" "$tmp/twfix/multi" ws
TW_CWD="$tmp/ws" tw_case "DIR ." "$tmp/empty" "$tmp/twfix/multi" .
tw_case "DIR no pkgs dir" "$tmp/empty" "$tmp/twfix/nopkgs" "$tmp/ws"
tw_case "DIR empty pkgs" "$tmp/empty" "$tmp/twfix/empty" "$tmp/ws"
tw_case "DIR pkgs is a file" "$tmp/empty" "$tmp/twfix/pkgsfile" "$tmp/ws"
tw_case "DIR bad toml after a good crate" "$tmp/empty" "$tmp/twfix/badtoml" "$tmp/ws"
tw_case "DIR no version" "$tmp/empty" "$tmp/twfix/noversion" "$tmp/ws"
tw_case "DIR no [package]" "$tmp/empty" "$tmp/twfix/nopackage" "$tmp/ws"
tw_case "DIR non-UTF-8 manifest" "$tmp/empty" "$tmp/twfix/badutf8" "$tmp/ws"
tw_case "DIR duplicate key" "$tmp/empty" "$tmp/twfix/dupkey" "$tmp/ws"
tw_case "DIR missing" "$tmp/empty" "$tmp/twfix/nopkgs" "$tmp/no-such-dir"
tw_case "--name rc version" "$tmp/empty" "$tmp/twfix/multi" --name "$tmp/ws/pkgs/b-0.70.0-rc.1"
tw_case "--name sub-table name ignored" "$tmp/empty" "$tmp/twfix/multi" --name "$tmp/ws/pkgs/a-1.0.0"
tw_case "--name symlinked dir" "$tmp/empty" "$tmp/twfix/multi" --name "$tmp/ws/pkgs/c-link"
tw_case "--name no manifest" "$tmp/empty" "$tmp/twfix/multi" --name "$tmp/ws/pkgs/no-manifest"
tw_case "--name manifest is a dir" "$tmp/empty" "$tmp/twfix/multi" --name "$tmp/ws/pkgs/dir-manifest"
tw_case "--name bad toml" "$tmp/empty" "$tmp/twfix/badtoml" --name "$tmp/ws/pkgs/b-1"
tw_case "--name no [package]" "$tmp/empty" "$tmp/twfix/nopackage" --name "$tmp/ws/pkgs/a-1"
tw_case "--name no name" "$tmp/empty" "$tmp/twfix/noname" --name "$tmp/ws/pkgs/a-1"
tw_case "two DIRs" "$tmp/empty" "$tmp/twfix/multi" "$tmp/ws" "$tmp/ws"
tw_case "no args" "$tmp/empty" "$tmp/twfix/multi"
printf '{"target_directory":"/t/x y","packages":[]}' >"$tmp/tw-meta.json"
tw_case "--target-dir" "$tmp/tw-meta.json" "$tmp/twfix/nopkgs" --target-dir
printf '{"packages":[]}' >"$tmp/tw-meta-none.json"
tw_case "--target-dir no key" "$tmp/tw-meta-none.json" "$tmp/twfix/nopkgs" --target-dir
printf '{' >"$tmp/tw-meta-bad.json"
tw_case "--target-dir bad JSON" "$tmp/tw-meta-bad.json" "$tmp/twfix/nopkgs" --target-dir
tw_case "--target-dir empty stdin" "$tmp/empty" "$tmp/twfix/nopkgs" --target-dir
tw_case "--target-dir LIVE" "$tmp/live-meta.json" "$tmp/twfix/nopkgs" --target-dir

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

# --- 6. llama-fit-verdict ---
# The original raises (exit 1, nothing printed) on a bad argv, rc, free_mib or input file;
# these cases compare the exact exit code first, then the bytes.
# Which characters are digits, and how deep the GGUF walk recurses, are the interpreter's:
# the port follows CPython 3.12/3.13 (Unicode 15), what the ladder's hosts run, so this
# section refuses any other (LFV_PYTHON=<python3.12 or 3.13>).
LFV_PY=scripts/lib/llama_fit_verdict.py
LFV_INTERP="${LFV_PYTHON:-$PY}"
if ! "$LFV_INTERP" -c 'import sys, unicodedata; sys.exit(not (sys.version_info[:2] in ((3, 12), (3, 13)) and unicodedata.unidata_version.startswith("15.")))'; then
    echo "FAIL: not measured: $LFV_INTERP is not CPython 3.12/3.13 (set LFV_PYTHON=)" >&2
    exit 1
fi
lfv() { # NAME -- the seven (or any number of) argv words, the same on both sides
    local name="$1" prc rrc
    shift
    "$LFV_INTERP" "$LFV_PY" "$@" </dev/null >/dev/null 2>&1 && prc=0 || prc=$?
    "$BIN" llama-fit-verdict "$@" </dev/null >/dev/null 2>&1 && rrc=0 || rrc=$?
    if [[ "$prc" -ne "$rrc" ]]; then
        fail=$((fail + 1))
        echo "MISMATCH: llama-fit-verdict $name exit code (py $prc, rust $rrc)" >&2
        return
    fi
    check "llama-fit-verdict $name" "$tmp/empty" "$LFV_INTERP" "$LFV_PY" "$@" -- "$BIN" llama-fit-verdict "$@"
}
# le VALUE BYTES: VALUE little-endian (two's complement), as raw bytes.
le() {
    local v="$1" n="$2" i
    for ((i = 0; i < n; i++)); do
        # shellcheck disable=SC2059
        printf "\\x$(printf '%02x' $(((v >> (8 * i)) & 255)))"
    done
}
gstr() { le "${#1}" 8; printf '%s' "$1"; }                     # a GGUF string (ASCII)
ghead() { printf 'GGUF'; le "${2:-3}" 4; le 0 8; le "$1" 8; }  # N_KV [VERSION]
ctxkv() { gstr llama.context_length; le "$1" 4; le "$2" "$3"; } # TYPE VALUE BYTES
g="$tmp/lfv"
mkdir -p "$g/dir"
{ # every skip path before the answer: a string, a string array, a u32 array, a nested array
    ghead 5
    gstr general.name; le 8 4; gstr 'x y'
    gstr tokenizer.tokens; le 9 4; le 8 4; le 2 8; gstr ab; gstr c
    gstr tokenizer.scores; le 9 4; le 4 4; le 3 8; le 1 12
    gstr nested; le 9 4; le 9 4; le 2 8; le 8 4; le 1 8; gstr q; le 0 4; le 1 8; le 7 1
    ctxkv 4 32768 4
} >"$g/v3.gguf"
{ ghead 1 2; ctxkv 4 2048 4; } >"$g/v2-2048.gguf"
{ ghead 1; ctxkv 4 0 4; } >"$g/zero.gguf"
{ ghead 1; ctxkv 5 -5 4; } >"$g/i32-neg.gguf"
{ ghead 1; ctxkv 10 -1 8; } >"$g/u64-max.gguf"
{ ghead 1; ctxkv 11 -7 8; } >"$g/i64-neg.gguf"
{ ghead 2; ctxkv 6 0 4; ctxkv 12 0 8; } >"$g/float-ctx.gguf"
{ ghead 1 1; ctxkv 4 8192 4; } >"$g/v1.gguf"
{ ghead 2; gstr a.b; le 13 4; ctxkv 4 8192 4; } >"$g/bad-type.gguf"
{ ghead 1; gstr llama.context_length; le 4 4; le 7 2; } >"$g/truncated.gguf"
{ ghead 1; le 40 8; printf 'short'; } >"$g/short-key.gguf"
{ ghead 2; gstr big; le 9 4; le 4 4; le $((1 << 61)) 8; ctxkv 4 8192 4; } >"$g/seek-overflow.gguf"
{ ghead 2; gstr big; le 9 4; le 4 4; le $((1 << 40)) 8; ctxkv 4 8192 4; } >"$g/seek-past-eof.gguf"
{ ghead 1; le 18 8; printf '\xe2.context_length'; le 4 4; le 999 4; } >"$g/bad-utf8-key.gguf"
{ ghead 1; ctxkv 4 3000 4; ctxkv 4 9 4; } >"$g/first-wins.gguf"
printf 'GGUF' >"$g/magic-only.gguf"
printf 'not a model\n' >"$g/text.gguf"
nest() { # DEPTH: DEPTH nested arrays, then the answer; CPython's recursion limit decides
    local i
    ghead 2
    gstr deep; le 9 4
    for ((i = 0; i < $1; i++)); do le 9 4; le 1 8; done
    le 4 4; le 1 8; le 0 4
    gstr x.context_length; le 4 4; le 777 4
}
nest 996 >"$g/nest-996.gguf"
nest 997 >"$g/nest-997.gguf"
PIN=a1b2c3d4e5f60718
printf 'version: 7000 (a1b2c3d4e5)\nbuilt with cc\n' >"$g/ver"
printf 'version: 1 (0123456)\n' >"$g/ver-other"
printf 'llama.cpp commit a1b2c3d4e5f60718ff\n' >"$g/ver-commit"
printf 'version (A1B2C3D) commit a1b2c3d\n' >"$g/ver-upper"
printf 'llama.cpp\n' >"$g/ver-none"
printf '\xff\xfe (a1b2c3d)\r\n' >"$g/ver-bytes"
printf 'load ...\nfitted:\n-c 8192 -ngl -1\n' >"$g/fits"
printf '  -c 8192 -ngl -1  \n\n\n' >"$g/fits-ws"
printf 'a\r\n-c 8192 -ngl -1\r\n' >"$g/fits-crlf"
printf -- '-c 8192 -ngl 33\n' >"$g/partial-ngl"
printf -- '-c 8192 -ngl -1 -ot "blk=CPU"\n' >"$g/partial-ot"
printf -- '-c 8192 -ngl -1 -ts 1,1\n' >"$g/partial-ts"
printf -- '-c 8192 -ngl -1 -ot\n' >"$g/partial-ot-end"
printf -- '-c 8192 -ngl -1 -otx\n' >"$g/ot-word"
printf -- '-c 2048 -ngl -1\n' >"$g/c2048"
printf -- '-c 2047 -ngl -1\n' >"$g/c2047"
printf -- '-c 0004096 -ngl -0001\n' >"$g/zeros"
printf -- '-c 8192 -ngl -1\nnoise after\n' >"$g/not-last"
printf -- '-ngl -1 -c 8192\n' >"$g/reordered"
printf -- '-c8192 -ngl -1\n' >"$g/no-space"
printf -- '-c 8192\x0b-ngl -1\n' >"$g/vt-split"
printf -- '-c 8192\x1f-ngl -1\x1c\n' >"$g/py-only-space"
printf -- '-c 8192\xe2\x80\xa8-ngl -1\n' >"$g/u2028-split"
printf -- '-c \xd9\xa8\xd9\xa1\xd9\xa9\xd9\xa2 -ngl -\xd9\xa1\n' >"$g/arabic-digits"
printf -- '-c \xf0\x91\xbd\x98\xf0\x91\xbd\x99\xf0\x91\xbd\x99\xf0\x91\xbd\x99\xf0\x91\xbd\x99 -ngl -1\n' >"$g/kawi-digits"
printf -- '-c \xf0\x9c\xb3\xb8\xf0\x9c\xb3\xb1\xf0\x9c\xb3\xb9\xf0\x9c\xb3\xb2 -ngl -1\n' >"$g/outlined-digits"
printf -- '-c \xc2\xb28192 -ngl -1\n' >"$g/superscript"
printf -- '-c %s -ngl -1\n' "$(printf '9%.0s' {1..4301})" >"$g/huge-ctx"
printf -- '-c %s -ngl -1\n' "$(printf '9%.0s' {1..60})" >"$g/big-ctx"
{ printf '"tab\there" \\ \x01 \xc3\xa9 \xf0\x9f\x98\x80 \xff '; printf 'z%.0s' {1..400}; printf '\n-c 8192 -ngl -1\n'; } >"$g/raw-escapes"
printf '\n \t\n' >"$g/blank"
lfv "fits, trained 32768 v3 (every skip path)" 1 "$PIN" 0 24000 "$g/ver" "$g/fits" "$g/v3.gguf"
lfv "tool absent" 0 "$PIN" 0 24000 "$g/ver" "$g/fits" "$g/v3.gguf"
lfv "tool_found not 1" true "$PIN" 0 24000 "$g/ver" "$g/fits" "$g/v3.gguf"
lfv "unpinned: other commit" 1 "$PIN" 0 24000 "$g/ver-other" "$g/fits" -
lfv "unpinned: no commit printed" 1 "$PIN" 0 24000 "$g/ver-none" "$g/fits" -
lfv "unpinned: no version file" 1 "$PIN" 0 24000 - "$g/fits" -
lfv "unpinned: pin under 7 chars" 1 a1b2c3 0 24000 "$g/ver" "$g/fits" -
lfv "unpinned: upper-case hex" 1 "$PIN" 0 24000 "$g/ver-upper" "$g/fits" -
lfv "pinned via commit form, built longer than pin" 1 "$PIN" 0 24000 "$g/ver-commit" "$g/fits" -
lfv "pinned, pin longer than built" 1 a1b2c3d4e5ffffff 0 24000 "$g/ver" "$g/fits" -
lfv "non-UTF-8 version bytes" 1 a1b2c3d 0 24000 "$g/ver-bytes" "$g/fits" -
lfv "non-ASCII pin" 1 "a1b2c3d4e5é" 0 24000 "$g/ver" "$g/fits" -
lfv "rc 1" 1 "$PIN" 1 24000 "$g/ver" "$g/fits" -
lfv "rc ' -3 '" 1 "$PIN" " -3 " 24000 "$g/ver" "$g/fits" -
lfv "rc +0_0" 1 "$PIN" +0_0 24000 "$g/ver" "$g/fits" -
lfv "rc arabic one" 1 "$PIN" "١" 24000 "$g/ver" "$g/fits" -
lfv "rc nbsp-padded 0" 1 "$PIN" $'\u00a00\u00a0' 24000 "$g/ver" "$g/fits" -
lfv "rc 1_ (raises)" 1 "$PIN" 1_ 24000 "$g/ver" "$g/fits" -
lfv "rc \\x1c0 (raises)" 1 "$PIN" $'\x1c0' 24000 "$g/ver" "$g/fits" -
lfv "rc empty (raises)" 1 "$PIN" "" 24000 "$g/ver" "$g/fits" -
lfv "rc 4301 digits (raises)" 1 "$PIN" "$(printf '0%.0s' {1..4301})" 24000 "$g/ver" "$g/fits" -
lfv "free -12" 1 "$PIN" 0 -12 "$g/ver" "$g/fits" -
lfv "free unknown" 1 "$PIN" 0 unknown "$g/ver" "$g/fits" -
lfv "free empty" 1 "$PIN" 0 "" "$g/ver" "$g/fits" -
lfv "free -" 1 "$PIN" 0 - "$g/ver" "$g/fits" -
lfv "free +5" 1 "$PIN" 0 +5 "$g/ver" "$g/fits" -
lfv "free ' 5'" 1 "$PIN" 0 " 5" "$g/ver" "$g/fits" -
lfv "free arabic 3" 1 "$PIN" 0 "٣" "$g/ver" "$g/fits" -
lfv "free 007" 1 "$PIN" 0 007 "$g/ver" "$g/fits" -
lfv "free --5 (raises)" 1 "$PIN" 0 --5 "$g/ver" "$g/fits" -
lfv "free superscript (raises)" 1 "$PIN" 0 "²" "$g/ver" "$g/fits" -
lfv "free 4301 digits (raises)" 1 "$PIN" 0 "$(printf '1%.0s' {1..4301})" "$g/ver" "$g/fits" -
lfv "free 40 digits" 1 "$PIN" 0 "$(printf '9%.0s' {1..40})" "$g/ver" "$g/fits" -
lfv "fits, padded" 1 "$PIN" 0 1 "$g/ver" "$g/fits-ws" -
lfv "fits, CRLF" 1 "$PIN" 0 1 "$g/ver" "$g/fits-crlf" -
lfv "partial: ngl 33" 1 "$PIN" 0 1 "$g/ver" "$g/partial-ngl" -
lfv "partial: -ot" 1 "$PIN" 0 1 "$g/ver" "$g/partial-ot" -
lfv "partial: -ts" 1 "$PIN" 0 1 "$g/ver" "$g/partial-ts" -
lfv "partial: -ot at the end" 1 "$PIN" 0 1 "$g/ver" "$g/partial-ot-end" -
lfv "-otx is not -ot" 1 "$PIN" 0 1 "$g/ver" "$g/ot-word" -
lfv "ctx 2048 < floor 4096" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/v3.gguf"
lfv "ctx 2048 = floor, trained 2048 v2" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/v2-2048.gguf"
lfv "ctx 2047 < floor, trained 2048" 1 "$PIN" 0 1 "$g/ver" "$g/c2047" "$g/v2-2048.gguf"
lfv "trained 0 is falsy" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/zero.gguf"
lfv "trained i32 -5" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/i32-neg.gguf"
lfv "trained u64 max" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/u64-max.gguf"
lfv "trained i64 -7" 1 "$PIN" 0 1 "$g/ver" "$g/fits" "$g/i64-neg.gguf"
lfv "float ctx keys skipped" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/float-ctx.gguf"
lfv "gguf v1" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/v1.gguf"
lfv "unknown value type" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/bad-type.gguf"
lfv "truncated value" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/truncated.gguf"
lfv "short key" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/short-key.gguf"
lfv "array seek overflow" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/seek-overflow.gguf"
lfv "array seek past EOF" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/seek-past-eof.gguf"
lfv "non-UTF-8 key" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/bad-utf8-key.gguf"
lfv "first context_length wins" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/first-wins.gguf"
lfv "magic only" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/magic-only.gguf"
lfv "not GGUF" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/text.gguf"
lfv "model is a directory" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/dir"
lfv "model missing" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/no-such.gguf"
lfv "996 nested arrays parse" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/nest-996.gguf"
lfv "997 nested arrays hit the recursion limit" 1 "$PIN" 0 1 "$g/ver" "$g/c2048" "$g/nest-997.gguf"
lfv "zero-padded numbers" 1 "$PIN" 0 1 "$g/ver" "$g/zeros" -
lfv "missing: verdict line not last" 1 "$PIN" 0 1 "$g/ver" "$g/not-last" -
lfv "reordered flags" 1 "$PIN" 0 1 "$g/ver" "$g/reordered" -
lfv "missing: -c8192" 1 "$PIN" 0 1 "$g/ver" "$g/no-space" -
lfv "missing: \\v splits the line" 1 "$PIN" 0 1 "$g/ver" "$g/vt-split" -
lfv "\\x1c/\\x1f are \\s" 1 "$PIN" 0 1 "$g/ver" "$g/py-only-space" -
lfv "missing: U+2028 splits the line" 1 "$PIN" 0 1 "$g/ver" "$g/u2028-split" -
lfv "arabic-indic digits" 1 "$PIN" 0 1 "$g/ver" "$g/arabic-digits" -
lfv "Kawi digits (Unicode 15)" 1 "$PIN" 0 1 "$g/ver" "$g/kawi-digits" -
lfv "missing: outlined digits (Unicode 16)" 1 "$PIN" 0 1 "$g/ver" "$g/outlined-digits" -
lfv "missing: superscript digit" 1 "$PIN" 0 1 "$g/ver" "$g/superscript" -
lfv "ctx 60 digits" 1 "$PIN" 0 1 "$g/ver" "$g/big-ctx" -
lfv "ctx 4301 digits (raises)" 1 "$PIN" 0 1 "$g/ver" "$g/huge-ctx" -
lfv "raw: escapes, bad bytes, 300-char cut" 1 "$PIN" 0 1 "$g/ver" "$g/raw-escapes" -
lfv "missing: blank stdout" 1 "$PIN" 0 1 "$g/ver" "$g/blank" -
lfv "missing: no stdout file given" 1 "$PIN" 0 1 "$g/ver" "" -
lfv "version file missing (raises)" 1 "$PIN" 0 1 "$g/no-such" "$g/fits" -
lfv "stdout file missing (raises)" 1 "$PIN" 0 1 "$g/ver" "$g/no-such" -
lfv "no args (raises)"
lfv "six args (raises)" 1 "$PIN" 0 1 "$g/ver" "$g/fits"
lfv "eight args (raises)" 1 "$PIN" 0 1 "$g/ver" "$g/fits" - extra
lfv "--help alone (raises)" --help
lfv "-h among seven" 1 "$PIN" 0 1 "$g/ver" "$g/fits" -h
lfv "--version alone (raises)" --version

ran=$((pass + fail))
echo "ci_tools_py_parity: $pass/$ran identical (declared $EXPECTED_CASES)"
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
