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
#   annotate-book-examples  vs scripts/annotate-book-examples.py (no caller; it rewrites
#                           book chapters, so each case also compares the two rewritten trees)
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
EXPECTED_CASES=139

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
run_side() { # OUT IN -- CMD...: stdout to OUT, stderr to OUT.err, prints the exit status
    local out="$1" in="$2" rc=0
    shift 3
    "$@" <"$in" >"$out" 2>"$out.err" || rc=$?
    echo "$rc"
}

# check NAME STDIN PYCMD... -- RSCMD...: identical stdout, and success vs failure alike.
# check_exact (same arguments) also wants the exact exit status, and stderr empty on both
# sides or on neither. Its TEXT is not compared: where the original dies, it prints a
# traceback and the port one line (each tool's README table records that).
check() { check_with class "$@"; }
check_exact() { check_with exact "$@"; }
check_with() {
    local mode="$1" name="$2" in="$3"
    shift 3
    local py=() rs=()
    while [[ "$1" != "--" ]]; do
        py+=("$1")
        shift
    done
    shift
    rs=("$@")
    local py_rc rs_rc same=1
    py_rc="$(run_side "$tmp/py.out" "$in" -- "${py[@]}")"
    rs_rc="$(run_side "$tmp/rs.out" "$in" -- "${rs[@]}")"
    cmp -s "$tmp/py.out" "$tmp/rs.out" || same=0
    if [[ "$mode" == exact ]]; then
        [[ "$py_rc" == "$rs_rc" ]] || same=0
        [[ -s "$tmp/py.out.err" ]] && [[ ! -s "$tmp/rs.out.err" ]] && same=0
        [[ ! -s "$tmp/py.out.err" ]] && [[ -s "$tmp/rs.out.err" ]] && same=0
    elif [[ "$((py_rc == 0))" != "$((rs_rc == 0))" ]]; then
        same=0
    fi
    if [[ "$same" -eq 1 ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: $name (py status $py_rc, rust status $rs_rc)" >&2
        diff <(od -c "$tmp/py.out") <(od -c "$tmp/rs.out") 2>&1 | head -8 >&2 || true
        echo "  py stderr: $(head -c 200 "$tmp/py.out.err" | tail -n 1)" >&2
        echo "  rs stderr: $(head -c 200 "$tmp/rs.out.err" | tail -n 1)" >&2
    fi
}

: >"$tmp/empty"

# --- The harness must see a planted mismatch, or its passes mean nothing. ---------
# One liar per thing it compares: stdout, success vs failure, and, in check_exact, the
# exact status and stderr presence. Each must be reported, and the stdout-only twins of
# the last two must pass under check, or exact mode is not what catches them.
printf 'a\n' >"$tmp/plant"
liar() { # EXPECT(pass|fail) CHECKER ARGS...
    local want="$1" p0=$pass f0=$fail
    shift
    "$@" 2>/dev/null
    if [[ "$want" == fail && "$fail" -ne "$((f0 + 1))" ]] || [[ "$want" == pass && "$pass" -ne "$((p0 + 1))" ]]; then
        echo "FAIL: planted case '$2' did not $want; the harness is blind" >&2
        exit 1
    fi
    pass=$p0
    fail=$f0
}
liar fail check "plant stdout" "$tmp/plant" cat -- printf 'b\n'
liar fail check "plant success" "$tmp/plant" cat -- bash -c 'cat; exit 1'
liar pass check "plant status twin" "$tmp/plant" bash -c 'cat; exit 1' -- bash -c 'cat; exit 2'
liar fail check_exact "plant status" "$tmp/plant" bash -c 'cat; exit 1' -- bash -c 'cat; exit 2'
liar pass check "plant stderr twin" "$tmp/plant" cat -- bash -c 'cat; echo x >&2'
liar fail check_exact "plant stderr" "$tmp/plant" cat -- bash -c 'cat; echo x >&2'
liar fail check_exact "plant stderr gone" "$tmp/plant" bash -c 'cat; echo x >&2' -- cat
liar pass check_exact "plant agree" "$tmp/plant" bash -c 'cat; echo x >&2; exit 3' -- bash -c 'cat; echo y >&2; exit 3'

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

# --- 7. annotate-book-examples (section 6 is perf041-report, on its own branch) ---
# The .py rewrites the chapters under its OWN repo (ROOT = the script's grandparent), so
# each case builds the same tree twice: the .py is copied into one copy's scripts/, the
# port is pointed at the other. stdout and success/failure are compared as everywhere,
# and then the two trees, so a chapter rewritten differently (or not at all) is a mismatch.
ABE_PY_SRC="$ROOT/scripts/annotate-book-examples.py"
abe_n=0
abe_case() { # NAME SETUP_FN [SETUP ARGS...]: SETUP_FN DIR builds book/src under DIR
    local name="$1" setup="$2" d before
    shift 2
    abe_n=$((abe_n + 1))
    d="$tmp/abe/$abe_n"
    mkdir -p "$d/py/scripts" "$d/rs"
    cp "$ABE_PY_SRC" "$d/py/scripts/annotate-book-examples.py"
    "$setup" "$d/py" "$@"
    "$setup" "$d/rs" "$@"
    before=$fail
    check_exact "annotate-book-examples $name" "$tmp/empty" \
        "$PY" "$d/py/scripts/annotate-book-examples.py" -- "$BIN" annotate-book-examples "$d/rs"
    if [[ "$fail" -eq "$before" ]] && ! diff -r --no-dereference -x scripts "$d/py" "$d/rs" >"$tmp/abe.diff" 2>&1; then
        pass=$((pass - 1))
        fail=$((fail + 1))
        echo "MISMATCH: annotate-book-examples $name (the rewritten trees differ)" >&2
        head -8 "$tmp/abe.diff" >&2
    fi
}
ch() { # DIR SUB NAME PRINTF-FORMAT [ARGS...]: one chapter, its bytes from printf
    local dir="$1" sub="$2" name="$3" fmt="$4"
    shift 4
    mkdir -p "$dir/book/src/$sub"
    # shellcheck disable=SC2059 # the format IS the fixture
    printf "$fmt" "$@" >"$dir/book/src/$sub/$name"
}
abe_none() { :; }
abe_clifile() { mkdir -p "$1/book/src" && printf 'x\n' >"$1/book/src/cli"; }
abe_emptydirs() { mkdir -p "$1/book/src/cli" "$1/book/src/lib"; }
abe_live() { mkdir -p "$1/book/src" && cp -r "$ROOT/book/src/cli" "$ROOT/book/src/lib" "$1/book/src/"; }
abe_stripped() { # the live book with every annotation removed, so all are re-derived
    abe_live "$1"
    local f
    for f in "$1"/book/src/cli/*.md "$1"/book/src/lib/*.md; do
        grep -v 'example-cost:' "$f" >"$f.tmp" || true
        mv "$f.tmp" "$f"
    done
}
abe_classes() {
    ch "$1" cli c.md '# Classes\n\nIntro.\n```bash\napr gpu status\n```\n```bash\napr ptx-map m.gguf\n```\n```bash\napr code\n```\n```bash\napr publish x\n```\n```bash\napr rm x\n```\n```bash\napr frobnicate\n```\n```bash\nls -la\n```\n```bash\naprun x\n```\n```bash\napr\n```\n```bash\napr Run m.gguf\n```\n```bash\napr 9run\n```\n```bash\n```\n```rust\nfn main() {}\n```\n'
}
abe_models() {
    ch "$1" cli m.md '```bash\napr run\n```\n```bash\napr run -m x.gguf\n```\n```bash\napr run --x a.apr b.gguf\n```\n```bash\napr qa m.safetensors\n```\n```bash\napr pull hf://Qwen/x\n```\n```bash\napr pull qwen2\n```\n```bash\napr pull Qwen2\n```\n```bash\napr pull other\n```\n```bash\napr run-x m.gguf\n```\n```bash\napr run:x m.gguf\n```\n```bash\napr run.gguf\n```\n'
}
abe_prompts() {
    ch "$1" cli p.md '```bash\n$ apr chat m.gguf\n```\n```bash\n  \n$ apr tui\napr run\n```\n```bash\napr run\nm.gguf\n```\n```bash\napr\xe3\x80\x80run\x1fm.gguf\n```\n```bash\napr\x1crun\n```\n```bash\necho x\napr gpu\n```\n```bash\n$apr gpu\n```\n'
}
abe_help() {
    ch "$1" cli h.md '```bash\napr run --help\n```\n```bash\napr serve --version\n```\n```bash\napr help run\n```\n```bash\napr helpx\n```\n```bash\napr help\xcc\x81 gpu\n```\n```bash\nfoo --help\napr gpu\n```\n'
}
abe_breaks() {
    ch "$1" cli crlf.md 'Text\r\n```bash\r\napr tui\r\n```\r\n'
    ch "$1" cli cr.md 'Text\r```rust\rfn f(){}\r```\r'
    ch "$1" cli u.md 'a\x0bb\x0c```bash\x1capr gpu\xe2\x80\xa8```\xc2\x85c\xe2\x80\xa9d\x1de\x1ef'
    ch "$1" cli us.md 'a\x1fb\n```rust\n'
}
abe_tails() {
    ch "$1" cli t1.md 'a\n\n\n'
    ch "$1" cli t2.md '```rust\n```'
    ch "$1" cli t3.md ''
    ch "$1" cli t4.md '\n'
    ch "$1" cli t5.md 'a\x0c'
}
abe_lookback() {
    ch "$1" cli l.md '<!-- example-cost: gpu -->\n\n```bash\n```\n<!-- example-cost: gpu -->\n\n\n```bash\n```\n<!-- example-cost: gpu -->\ntext\n```rust\n```\n<!-- example-cost: x -->\n```rust\n```\n   \n<!-- example-cost: nonsense words -->  \n \t\n```bash\napr run\n```\n'
}
abe_costlines() {
    ch "$1" cli k.md '<!--example-cost:x-->\n```rust\n```\n<!-- \x1c example-cost: -->\n```rust\n```\n<!--example-cost:-->\n```rust\n```\n<!-- example-cost: a>b -->\n```rust\n```\n<!-- example-cost: x --> y\n```rust\n```\n<!-- cost: x -->\n```rust\n```\nx<!-- example-cost: x -->\n```rust\n```\n<!-- example-cost: x ->\n```rust\n```\n<!--\xc2\xa0example-cost:\xe3\x80\x80y\xe2\x80\xa8-->\n```rust\n```\n'
}
abe_fences() {
    ch "$1" cli f.md '```bash \t\x1f\xc2\xa0\napr gpu\n```\n```Bash\napr gpu\n```\n ```bash\napr gpu\n```\n``` bash\n```\n```bashx\n```\n````bash\n```\n```bash\napr gpu\n```rust\n```\napr tui\n```bash\n'
}
abe_order() {
    local n
    for n in b.md a.md A.md .h.md .md y.md.md c.MD 'n\n.md' 'é.md' 'z z.md'; do
        # shellcheck disable=SC2059 # the name IS a printf fixture
        ch "$1" cli "$(printf "$n")" '```rust\n'
    done
    ch "$1" lib a.md '```bash\napr tui\n```\n'
    ch "$1" other o.md '```rust\n'
}
abe_libonly() { ch "$1" lib only.md 'x\n```bash\napr eval m.apr\n```\n'; }
abe_bom() { ch "$1" cli bom.md '\xef\xbb\xbf```bash\napr gpu\n```\n\xef\xbb\xbfx\n```rust\n```\n'; }
abe_badutf8() {
    ch "$1" cli a.md '```rust\n'
    ch "$1" cli b.md 'bad \xff utf-8\n```rust\n'
    ch "$1" cli c.md '```rust\n'
}
abe_surrogate() { ch "$1" cli s.md '```rust\n\xed\xa0\x80\n'; }
abe_dirmd() { ch "$1" cli a.md '```rust\n' && mkdir -p "$1/book/src/cli/d.md"; }
abe_dangling() { ch "$1" cli a.md '```rust\n' && ln -s nowhere "$1/book/src/cli/l.md"; }
abe_symlink() { # a chapter that is a link: the TARGET is rewritten, the link kept
    ch "$1" cli real.txt '```rust\n'
    ln -s real.txt "$1/book/src/cli/l.md"
}
abe_readonly() {
    ch "$1" cli a.md '```rust\n'
    ch "$1" cli b.md 'no fences\n'
    chmod 444 "$1/book/src/cli/b.md"
}
abe_case "no book" abe_none
abe_case "cli is a file" abe_clifile
abe_case "empty dirs" abe_emptydirs
abe_case "live book" abe_live
abe_case "live book, annotations stripped" abe_stripped
abe_case "cost classes" abe_classes
abe_case "model names" abe_models
abe_case "prompts and whitespace" abe_prompts
abe_case "help and version" abe_help
abe_case "line breaks" abe_breaks
abe_case "trailing newlines" abe_tails
abe_case "look-back" abe_lookback
abe_case "cost-line forms" abe_costlines
abe_case "fence forms" abe_fences
abe_case "order and names" abe_order
abe_case "lib only" abe_libonly
abe_case "BOM" abe_bom
abe_case "invalid utf-8 stops the run" abe_badutf8
abe_case "encoded surrogate" abe_surrogate
abe_case "a directory named .md" abe_dirmd
abe_case "dangling link" abe_dangling
abe_case "link to a chapter" abe_symlink
abe_case "read-only chapter" abe_readonly
abe_badother() { # glob skips a non-.md name whatever its bytes; the chapter still gets annotated
    ch "$1" cli a.md '```bash\napr tui\n```\n'
    printf 'x\n' >"$1/book/src/cli/$(printf '\xff.txt')"
    printf 'x\n' >"$1/book/src/cli/$(printf 'a.md\xff')"
    printf 'x\n' >"$1/book/src/cli/$(printf '\xff.MD')"
}
abe_case "a non-utf-8 name that is not a chapter" abe_badother
# The tree diff must see a planted difference that stdout cannot: one stray file, Rust side only.
abe_plant() {
    ch "$1" cli a.md '```bash\napr tui\n```\n'
    [[ "$1" == */rs ]] && printf 'x\n' >"$1/book/src/cli/stray.txt"
    return 0
}
liar fail abe_case "plant tree" abe_plant
chmod -R u+w "$tmp/abe"


ran=$((pass + fail))
echo "ci_tools_py_parity: $pass/$ran identical (declared $EXPECTED_CASES)"
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
