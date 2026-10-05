#!/usr/bin/env bash
# ci_tools_py_parity_test.sh -- PY-PORT-2: the identical-output case table for the
# ports into crates/aprender-ci-tools (C301: no Python in the build).
#
# Every case runs through BOTH the .py original and the Rust subcommand; stdout must
# be byte-identical and both must agree on success vs failure. The .py files are the
# EXTERNAL VALIDATOR here (C301 allows that), never a build step.
#
#   publishable-crates      vs scripts/lib/publishable_crates.py
#   package-include-diff    vs scripts/lib/package_include_diff.py
#   coverage-report-scope   vs scripts/coverage_report_scope.py
#   tarball-workspace       vs scripts/lib/tarball_workspace.py, read from git (the file is
#                           deleted; its callers are switched). Its written Cargo.toml is
#                           compared too, mapping only the one header line that names it.
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
EXPECTED_CASES=82

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
pid_case() { # NAME LISTING_PRINTF INCLUDES_PRINTF
    printf "$2" >"$tmp/l"
    printf "$3" >"$tmp/i"
    check "package-include-diff $1" "$tmp/empty" \
        "$PY" scripts/lib/package_include_diff.py "$tmp/l" "$tmp/i" -- \
        "$BIN" package-include-diff "$tmp/l" "$tmp/i"
}
pid_case "all present" 'a.rs\nb.rs\n' 'a.rs\tm.rs\nb.rs\tm.rs\n'
pid_case "one missing" 'a.rs\n' 'a.rs\tm.rs\nb.rs\tn.rs\n'
pid_case "empty listing" '' 'a.rs\tm.rs\n'
pid_case "empty includes" 'a.rs\n' ''
pid_case "duplicates kept" '' 'x\ty\nx\ty\n'
pid_case "no tab" '' 'lonely\n'
pid_case "second tab" '' 't\ts\tz\n'
pid_case "CRLF" 'a.rs\r\n' 'a.rs\tm.rs\r\nq\tr\r\n'
pid_case "lone CR" 'a.rs\rb.rs' 'b.rs\tm\ra.rs\tm\rc\td'
pid_case "blank includes" '' '\n\n'
pid_case "whitespace include row" '' '  \n'
pid_case "whitespace listing" '  \n\x1c\n' '  \tm\n\x1c\tn\n'
pid_case "inner spaces" ' a.rs \n' ' a.rs \tm\n'
pid_case "no final newline" 'a.rs' 'a.rs\tm'
pid_case "bad utf8" '\xff.rs\n' '\xff.rs\tm\n\xfe\tq\n\xe2\x82\tr\n'
check "package-include-diff missing listing" "$tmp/empty" \
    "$PY" scripts/lib/package_include_diff.py "$tmp/nope" "$tmp/i" -- \
    "$BIN" package-include-diff "$tmp/nope" "$tmp/i"
git ls-files crates/aprender-ci-tools >"$tmp/live-l"
{ sed 's/$/\tsrc.rs/' "$tmp/live-l"; printf 'crates/aprender-ci-tools/src/planted.rs\tlib.rs\n'; } >"$tmp/live-i"
check "package-include-diff LIVE+plant" "$tmp/empty" \
    "$PY" scripts/lib/package_include_diff.py "$tmp/live-l" "$tmp/live-i" -- \
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

# --- 4. tarball-workspace (its .py is deleted: the validator is read from git) ---------
TW_REV=eb60aef156667e2da9aab6e3a78ed1ae1540d648
mkdir -p "$tmp/pyval" "$tmp/twfix"
if ! git cat-file -e "$TW_REV:scripts/lib/tarball_workspace.py" 2>/dev/null; then
    echo "FAIL: not measured: $TW_REV is not in this clone, so the original cannot run (fetch it)" >&2
    exit 1
fi
git show "$TW_REV:scripts/lib/tarball_workspace.py" >"$tmp/pyval/tarball_workspace.py"
git show "$TW_REV:scripts/lib/toml_compat.py" >"$tmp/pyval/toml_compat.py"
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

ran=$((pass + fail))
echo "ci_tools_py_parity: $pass/$ran identical (declared $EXPECTED_CASES)"
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
