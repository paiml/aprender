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
#   cascade-universe        vs scripts/lib/cascade_universe.py (kept: N-1), exact exit code and
#                           stderr too; a fake `cargo` serves one fixture per workspace
#   tarball-shrink-report   vs scripts/lib/tarball_shrink_report.py
#   tarball-workspace       vs scripts/lib/tarball_workspace.py, read from git (the file is
#                           deleted; its callers are switched). Its written Cargo.toml is
#                           compared too, mapping only the one header line that names it.
#   tarball-build-errors    vs scripts/lib/tarball_build_errors.py
#   annotate-book-examples  vs scripts/annotate-book-examples.py (no caller; it rewrites
#                           book chapters, so each case also compares the two rewritten trees)
#   extract-book-examples   vs scripts/extract_book_examples.py
#   llama-fit-verdict       vs scripts/lib/llama_fit_verdict.py (kept: model_ladder.sh's
#                           certification path still calls it), exact exit code too, run
#                           under CPython 3.12/3.13 (LFV_PYTHON=) whose Unicode it follows.
#   complexity-rows         vs scripts/lib/complexity_rows.py (kept: check_complexity_ratchet.sh
#                           still calls it, N-1), exact exit code too.
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
EXPECTED_CASES=365

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

# --- 3b. cascade-universe (fake cargo serves a fixture per --manifest-path) ---------
# The original's exit 2 (a refusal) and exit 1 (a raise) mean different things to its
# callers, so these cases compare the exact code, and stderr too unless it raised.
mkdir -p "$tmp/cubin" "$tmp/cu"
cat >"$tmp/cubin/cargo" <<'STUB'
#!/usr/bin/env bash
m=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == --manifest-path ]]; then m="${2:-}"; fi
    shift
done
case "$m" in
    */crates/facades/Cargo.toml | crates/facades/Cargo.toml) f="$FAKE_FAC" ;;
    *) f="$FAKE_ROOT" ;;
esac
if [[ "$f" == FAIL ]]; then printf 'error: no manifest at %s\n' "$m" >&2; exit 101; fi
cat "$f"
STUB
chmod +x "$tmp/cubin/cargo"
cupkgs() { # N [EXTRA_JSON]: N publishable crates c00.., then EXTRA, as one metadata doc
    local i sep="" out='{"packages":['
    for ((i = 0; i < $1; i++)); do
        out+="$sep$(printf '{"name":"c%02d","version":"0.%d.0","manifest_path":"/r/c%02d/Cargo.toml","publish":null}' "$i" "$i" "$i")"
        sep=","
    done
    if [[ -n "${2:-}" ]]; then out+="$sep$2"; fi
    printf '%s]}' "$out"
}
cupkgs 70 '{"name":"zz-private","version":"1.0.0","manifest_path":"/r/zz/Cargo.toml","publish":[]},{"name":"Zed","version":7,"manifest_path":"/r/Zed/Cargo.toml","publish":["crates-io"]},{"name":"ñame","version":true,"manifest_path":"/r/n/Cargo.toml"},{"name":"a-null","version":null,"manifest_path":"/r/a/Cargo.toml"}' >"$tmp/cu/root.json"
printf '{"packages":[{"name":"fa","version":"0.4.0","manifest_path":"/r/crates/facades/fa/Cargo.toml","publish":null},{"name":"fb","version":"0.4.1","manifest_path":"/r/crates/facades/fb/Cargo.toml","publish":["crates-io"]},{"name":"fpriv","version":"0.4.0","manifest_path":"/r/f/Cargo.toml","publish":[]}],"workspace_root":"/r/crates/facades"}' >"$tmp/cu/fac.json"
printf '{"packages":[{"name":"c00","version":"9.9.9","manifest_path":"/r/c00/Cargo.toml"}]}' >"$tmp/cu/fac-dup-same.json"
printf '{"packages":[{"name":"c00","version":"0.4.0","manifest_path":"/r/crates/facades/c00/Cargo.toml"}]}' >"$tmp/cu/fac-dup-diff.json"
printf '{"packages":[{"name":"c01","version":"0.4.0","manifest_path":"/r/c01/Cargo.toml","publish":[]}]}' >"$tmp/cu/fac-dup-private.json"
printf '{"packages":{}}' >"$tmp/cu/fac-dict.json"
printf '{"packages":[]}' >"$tmp/cu/fac-empty.json"
cupkgs 69 >"$tmp/cu/root69.json"
cupkgs 69 '{"name":"c00","version":"0.0.0","manifest_path":"/r/c00/Cargo.toml"}' >"$tmp/cu/root69dup.json"
cupkgs 70 '{"name":"nov","manifest_path":"/r/nov/Cargo.toml"}' >"$tmp/cu/noversion.json"
cupkgs 70 '"c99"' >"$tmp/cu/strpkg.json"
printf ' \t\n' >"$tmp/cu/blank.json"
printf '{"packages":[' >"$tmp/cu/bad.json"
printf '{"workspace_root":"/r"}' >"$tmp/cu/nopkgs.json"
CU_PY=scripts/lib/cascade_universe.py
cu() { # NAME ROOT_FIXTURE FACADES_FIXTURE ARGS... (a fixture may be FAIL)
    local name="$1" prc rrc
    local e=(env PATH="$tmp/cubin:$PATH" FAKE_ROOT="$2" FAKE_FAC="$3")
    shift 3
    "${e[@]}" "$PY" "$CU_PY" "$@" </dev/null >/dev/null 2>"$tmp/cu.py.err" && prc=0 || prc=$?
    "${e[@]}" "$BIN" cascade-universe "$@" </dev/null >/dev/null 2>"$tmp/cu.rs.err" && rrc=0 || rrc=$?
    if [[ "$prc" -ne "$rrc" ]]; then
        fail=$((fail + 1))
        echo "MISMATCH: cascade-universe $name exit code (py $prc, rust $rrc)" >&2
        return
    fi
    if [[ "$prc" -ne 1 ]] && ! cmp -s "$tmp/cu.py.err" "$tmp/cu.rs.err"; then
        fail=$((fail + 1))
        echo "MISMATCH: cascade-universe $name stderr" >&2
        diff "$tmp/cu.py.err" "$tmp/cu.rs.err" | head -6 >&2 || true
        return
    fi
    check "cascade-universe $name" "$tmp/empty" \
        "${e[@]}" "$PY" "$CU_PY" "$@" -- "${e[@]}" "$BIN" cascade-universe "$@"
}
c="$tmp/cu"
cu "default" "$c/root.json" "$c/fac.json"
cu "names" "$c/root.json" "$c/fac.json" --names
cu "names after the root" "$c/root.json" "$c/fac.json" . --names
cu "relative root, normalised" "$c/root.json" "$c/fac.json" "x/../y//./"
cu "absolute root, normalised" "$c/root.json" "$c/fac.json" "/tmp/../a//b/"
cu "double-slash root" "$c/root.json" "$c/fac.json" "//srv/r"
cu "root above cwd" "$c/root.json" "$c/fac.json" "../../.."
cu "empty-string root" "$c/root.json" "$c/fac.json" ""
cu "single-dash arg is the root" "$c/root.json" "$c/fac.json" -x
cu "unknown flags ignored" "$c/root.json" "$c/fac.json" --frob --names=1 r
cu "extra positionals ignored" "$c/root.json" "$c/fac.json" r1 r2 r3
cu "--help is not help" "$c/root.json" "$c/fac.json" --help
cu "facades empty dict" "$c/root.json" "$c/fac-dict.json"
cu "facades empty list" "$c/root.json" "$c/fac-empty.json"
cu "same crate, same manifest: facades row wins" "$c/root.json" "$c/fac-dup-same.json"
cu "same crate, two manifests (2)" "$c/root.json" "$c/fac-dup-diff.json"
cu "a private duplicate is skipped" "$c/root.json" "$c/fac-dup-private.json"
cu "69 crates (2)" "$c/root69.json" "$c/fac-empty.json"
cu "69 plus facades clears the floor" "$c/root69.json" "$c/fac.json" --names
cu "duplicate within one workspace counts once (2)" "$c/root69dup.json" "$c/fac-empty.json"
cu "root cargo fails (2)" FAIL "$c/fac.json"
cu "facades cargo fails (2)" "$c/root.json" FAIL
cu "blank stdout (2)" "$c/blank.json" "$c/fac.json"
cu "bad json (raises)" "$c/bad.json" "$c/fac.json"
cu "no packages key (raises)" "$c/nopkgs.json" "$c/fac.json"
cu "package without version (raises)" "$c/noversion.json" "$c/fac.json"
cu "package is a string (raises)" "$c/strpkg.json" "$c/fac.json"
cu "facades fixture missing (2)" "$c/root.json" "$c/no-such.json"
if ! cu_out=$(env PATH="$tmp/cubin:$PATH" FAKE_ROOT="$c/root.json" FAKE_FAC="$c/fac.json" "$BIN" cascade-universe) ||
    [[ "$(grep -c . <<<"$cu_out")" -ne 75 ]] || ! grep -q "^fa	0.4.0	/r/crates/facades/fa/Cargo.toml	$ROOT/crates/facades\$" <<<"$cu_out" ||
    grep -q 'zz-private\|fpriv' <<<"$cu_out"; then
    echo "FAIL: cascade-universe fixture did not print the 75 planted rows (vacuous)" >&2
    fail=$((fail + 1))
fi
check "cascade-universe LIVE" "$tmp/empty" "$PY" "$CU_PY" "$ROOT" -- "$BIN" cascade-universe "$ROOT"
check "cascade-universe LIVE names" "$tmp/empty" "$PY" "$CU_PY" --names -- "$BIN" cascade-universe --names
if ! cu_live=$("$BIN" cascade-universe --names) || [[ "$(grep -c . <<<"$cu_live")" -lt 70 ]]; then
    echo "FAIL: live cascade-universe enumerated fewer than 70 crates" >&2
    fail=$((fail + 1))
fi

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
# --- 6. perf041-report ---
# No caller anywhere in the tree, so no gate path; the .py stays as this table's validator
# until the port is released. A case is a directory of band records; exit 0 report, 1 no
# band / no fast c=1 band, 1 where the original raised (after the lines it had printed).
P41_PY=("$PY" scripts/perf041_report.py)
p41="$tmp/p41"
rec() { # DIR FILE JSON
    mkdir -p "$p41/$1"
    printf '%s\n' "$3" >"$p41/$1/$2"
}
band() { # DIR FILE LABEL C LAT AGG TOK_P50 TOK_MIN TOK_MAX
    rec "$1" "$2" "{\"label\": \"$3\", \"c\": $4, \"latency_p50_s\": $5, \"agg_tok_s\": $6, \"tokens_p50\": $7, \"tokens_min\": $8, \"tokens_max\": $9}"
}
check_p41() { # NAME ARGS... (the same args go to both sides)
    local name="$1"
    shift
    check_exact "perf041-report $name" "$tmp/empty" \
        "${P41_PY[@]}" "$@" -- "$BIN" perf041-report "$@"
}
# A full sweep: fast c=1,2,4 and forced c=1, two replicates each (even medians).
for r in 1 2; do
    band full "fast-c1-r$r.json" fast-c1 1 "0.5$r" "40.$r" 64 60 "6$r"
    band full "fast-c2-r$r.json" fast-c2 2 "0.7$r" "71.$r" 64 58 66
    band full "fast-c4-r$r.json" fast-c4 4 "1.3$r" "110.$r" 63 50 70
    band full "forced-c1-r$r.json" forced-c1 1 "0.6$r" "35.$r" 64 61 64
done
mkdir -p "$p41/noforced" "$p41/nofast1" "$p41/empty"
cp "$p41"/full/fast-* "$p41/noforced/"
cp "$p41"/full/fast-c2-* "$p41"/full/fast-c4-* "$p41"/full/forced-* "$p41/nofast1/"
# Skips: not JSON, an error record (any value type), a BOM, a hidden file, other extensions.
cp -r "$p41/full" "$p41/skips"
rec skips a-bad.json '{"label": '
rec skips b-err.json '{"error": "connection refused", "c": 1}'
rec skips c-err.json '{"error": {"code": 7, "why": [1.5, null, true, 1e16, 1e-05, "it'"'"'s"]}}'
rec skips d-err.json '{"error": null}'
rec skips e-bom.json $'\xef\xbb\xbf{"c": 1}'
rec skips .hidden.json '[]'
rec skips notes.txt '[]'
rec skips x.JSON '[]'
# Medians: ints and bools, odd and even counts, an even median rounding half to even.
band ints m1.json fast-1 1 1 true 2 2 9
band ints m2.json fast-1 1 2 2 3 2.0 9
band ints m3.json fast-1 1 4 3 3 5 9
band ints m4.json fast-1 1 3 4 2 3 9
# Modes: no dash, empty, absent, null, false, 0, leading dash, sorting by code point.
band modes a.json fastest 1 0.5 10 1 1 1
band modes b.json "" 3 0.5 10 1 1 1
rec modes c.json '{"c": 10, "latency_p50_s": 1, "agg_tok_s": 2, "tokens_p50": 3, "tokens_min": 1, "tokens_max": 1}'
rec modes d.json '{"label": null, "c": 9, "latency_p50_s": 1, "agg_tok_s": 2, "tokens_p50": 3, "tokens_min": 1, "tokens_max": 1}'
rec modes e.json '{"label": false, "c": 9, "latency_p50_s": 1, "agg_tok_s": 2, "tokens_p50": 3, "tokens_min": 1, "tokens_max": 1}'
rec modes f.json '{"label": 0, "c": 9, "latency_p50_s": 1, "agg_tok_s": 2, "tokens_p50": 3, "tokens_min": 1, "tokens_max": 1}'
band modes g.json -fast 2 0.5 10 1 1 1
band modes h.json "Zeta-1" 1 0.5 10 1 1 1
band modes i.json "épsilon-1" 1 0.5 10 1 1 1
band modes j.json fast-1 1 0.25 20 1 1 1
band modes k.json fast-10 10 2.5 60 1 1 1
band modes l.json fast-9 9 2 50 1 1 1
# c as Python int() takes it: padded, signed and underscored strings, floats, a bool, a
# repeated key (the last one wins).
band cforms a.json fast-1 '" 1 "' 0.5 10 1 1 1
band cforms b.json fast-10 '"1_0"' 1.5 30 1 1 1
band cforms c.json fast-2 2.9 1 18 1 1 1
band cforms d.json fast-1 true 0.4 11 1 1 1
band cforms e.json fast-3 '"+3"' 1.2 25 1 1 1
band cforms f.json fast-7 '"-7"' 1.2 25 1 1 1
rec cforms g.json '{"label": "fast-4", "c": 1, "c": 4, "latency_p50_s": 1, "agg_tok_s": 2, "tokens_p50": 3, "tokens_min": 1, "tokens_max": 1}'
band cforms h.json fast-0 -0.5 1 2 1 1 1
# tokens_min/max: the FIRST extreme decides whether :d sees an int.
band tokint a.json fast-1 1 0.5 10 1 2 3
band tokint b.json fast-1 1 0.5 10 1 2.0 3.0
band tokflt a.json fast-1 1 0.5 10 1 2.0 3
band tokflt b.json fast-1 1 0.5 10 1 2 3
band tokmax a.json fast-1 1 0.5 10 1 2 3
band tokmax b.json fast-1 1 0.5 10 1 2 3.5
# Formatting: rounding ties, negative zero, overflow to inf, inf/inf = nan.
band fmt a.json fast-1 1 0.0005 0.25 2.5 1 1
band fmt b.json fast-2 2 -0.0 -0.04 -0.4 1 1
band fmt c.json forced-1 1 0.00049 1e308 0.5 1 1
band fmt d.json fast-4 4 1e308 1e308 1 1 1
band nan a.json fast-1 1 1.7e308 5 1 1 1
band nan b.json fast-1 1 1.7e308 5 1 1 1
band nan c.json fast-2 2 1.7e308 5 1 1 1
band nan d.json fast-2 2 1.7e308 5 1 1 1
# Zero divisors (Python raises), and a zero penalty (no division: scaling is nan).
band z1 a.json fast-1 1 0 10 1 1 1
band z1 b.json forced-1 1 0.5 10 1 1 1
band z2 a.json fast-1 1 0 10 1 1 1
band z3 a.json fast-1 1 0.5 10 1 1 1
band z3 b.json fast-2 2 0.5 0 1 1 1
band z4 a.json fast-1 1 0.5 10 1 1 1
band z4 b.json forced-1 1 0 10 1 1 1
band z4 c.json fast-2 2 0.9 18 1 1 1
# Where the original raised, after what it had printed.
cp -r "$p41/skips" "$p41/notobj" && rec notobj z.json '[1, 2]'
cp -r "$p41/skips" "$p41/strobj" && rec strobj z.json '"error"'
cp -r "$p41/skips" "$p41/noc" && rec noc z.json '{"label": "fast-1"}'
band badc a.json fast-1 '"2.0"' 0.5 10 1 1 1
band nullc a.json fast-1 null 0.5 10 1 1 1
rec lblint a.json '{"label": 5, "c": 1}'
cp -r "$p41/full" "$p41/nolat" && rec nolat fast-c4-r3.json '{"label": "fast-4", "c": 4}'
cp -r "$p41/full" "$p41/strlat" && band strlat fast-c4-r1.json fast-4 4 '"1.3"' 1 1 1 1
cp -r "$p41/full" "$p41/listtok" && band listtok fast-c2-r1.json fast-2 2 1 1 1 '[1]' 1
mkdir -p "$p41/isdir/x.json" && band isdir a.json fast-1 1 0.5 10 1 1 1
band badutf8 a.json fast-1 1 0.5 10 1 1 1 && printf '{"label": "\xff"}\n' >"$p41/badutf8/b.json"
# Literals json reads: NaN and the infinities in bands of one or two replicates (Python's
# sort leaves those in place), a float beyond the double range, NaN as a skipped error.
band nanlit a.json fast-1 1 0.5 40 1 1 1
band nanlit b.json fast-1 1 NaN Infinity 2 1 NaN
band nanlit c.json fast-2 2 1e400 -Infinity NaN 2 3
band nanlit d.json forced-1 1 Infinity 5 1 1 1
rec nanlit e.json '{"error": [NaN, Infinity, -Infinity]}'
# Integers beyond 64 bits stay exact: min/max compare an int with a double exactly (the
# double 2^64 first, the int 2^64+1 wins), c beyond 64 bits as an int and as a str.
band bigint a.json fast-1 1 0.5 40.5 18446744073709551617 18446744073709551617 18446744073709551616.0
band bigint b.json fast-1 1 0.5 40.5 18446744073709551618 18446744073709551616 18446744073709551617
band bigint c.json fast-36893488147419103232 36893488147419103232 1.0 2.5 1 1 170141183460469231731687303715884105727
band bigint d.json fast-x '"-36893488147419103232"' 1.0 2.5 1 -170141183460469231731687303715884105728 1
# Only error records: their skip lines, then no band.
rec erronly a.json '{"error": "x"}'
rec erronly b.json '{"error": 1e400}'
# File names that are not UTF-8: read, and sorted as Python's str (\xff is U+DCFF, before
# U+E000); the first of a tie decides int or float.
band badname a.json fast-1 1 0.5 40 1 1 1
band badname $'\xff.json' fast-3 3 1.5 90 3 5 7
band badname $'\xee\x80\x80.json' fast-3 3 1.6 91 3 5.0 7.0

for d in full noforced nofast1 empty skips ints modes cforms tokint tokflt tokmax fmt nan \
    z1 z2 z3 z4 notobj strobj noc badc nullc lblint nolat strlat listtok isdir badutf8 \
    nanlit bigint erronly badname; do
    check_p41 "$d" "$p41/$d"
done
check_p41 "missing dir" "$p41/nope"
check_p41 "a file, not a dir" "$p41/full/fast-c1-r1.json"
check_p41 "trailing slash" "$p41/full/"
check_p41 "extra arguments ignored" "$p41/full" extra --more
check_p41 "hyphen argument" "-x"
check_p41 "no argument (the default directory)"
check_exact "perf041-report empty argument is the cwd" "$tmp/empty" \
    env -C "$p41/full" "$PY" "$ROOT/scripts/perf041_report.py" "" -- \
    env -C "$p41/full" "$BIN" perf041-report ""
# --- 7. annotate-book-examples ---
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

# --- 8. extract-book-examples ---
# The .py reads the book under its OWN repo (ROOT = the script's grandparent), so each case
# builds one tree, copies the .py into its scripts/ and points the port at the same tree.
# Neither side writes, so check_exact (stdout, exact status, stderr present) is the whole
# comparison.
EBE_PY_SRC="$ROOT/scripts/extract_book_examples.py"
ebe_n=0
ebe_case() { # NAME SETUP_FN: SETUP_FN DIR builds book/src under DIR
    local name="$1" setup="$2" d
    ebe_n=$((ebe_n + 1))
    d="$tmp/ebe/$ebe_n"
    mkdir -p "$d/scripts"
    cp "$EBE_PY_SRC" "$d/scripts/extract_book_examples.py"
    "$setup" "$d"
    check_exact "extract-book-examples $name" "$tmp/empty" \
        "$PY" "$d/scripts/extract_book_examples.py" -- "$BIN" extract-book-examples "$d"
}
ebe_ch() { # DIR SUB NAME PRINTF-FORMAT [ARGS...]: one chapter, its bytes from printf
    local dir="$1" sub="$2" name="$3" fmt="$4"
    shift 4
    mkdir -p "$dir/book/src/$sub"
    # shellcheck disable=SC2059 # the format IS the fixture
    printf "$fmt" "$@" >"$dir/book/src/$sub/$name"
}
ebe_none() { :; }
ebe_clifile() { mkdir -p "$1/book/src" && printf 'x\n' >"$1/book/src/cli"; }
ebe_emptydirs() { mkdir -p "$1/book/src/cli" "$1/book/src/lib"; }
ebe_live() { mkdir -p "$1/book/src" && cp -r "$ROOT/book/src/cli" "$ROOT/book/src/lib" "$1/book/src/"; }
ebe_costs() {
    local c
    for c in trivial model-required gpu destructive interactive skip Trivial bogus; do
        ebe_ch "$1" cli "c-$c.md" '<!-- example-cost: %s -->\n```bash\napr x\n```\n' "$c"
    done
    ebe_ch "$1" lib m.md '<!-- example-cost: model-required model: qwen2.5 -->\n```bash\n```\n<!-- example-cost: model: m -->\n```bash\n```\n<!-- example-cost: gpu model: -->\n```bash\n```\n<!-- example-cost: gpu model: a model: b -->\n```bash\n```\n<!-- example-cost: gpu x model: model: -->\n```bash\n```\n<!-- example-cost:\xe3\x80\x80gpu\x1fmodel:\xc2\xa0\xc3\xa9"\\x -->\n```rust\n```\n<!-- example-cost: gpu MODEL: m model:x -->\n```rust\n```\n'
}
ebe_costlines() {
    ebe_ch "$1" cli k.md '<!--example-cost:gpu-->\n```rust\n```\n<!-- \x1c example-cost: gpu -->\n```rust\n```\n<!--example-cost:-->\n```rust\n```\n<!-- example-cost: a>b -->\n```rust\n```\n<!-- example-cost: gpu --> y\n```rust\n```\n<!-- cost: gpu -->\n```rust\n```\nx<!-- example-cost: gpu -->\n```rust\n```\n<!-- example-cost: gpu ->\n```rust\n```\n<!--\xc2\xa0example-cost:\xe3\x80\x80gpu\xe2\x80\x80-->\n```rust\n```\n<!-- example-cost: gpu -->\xe2\x80\x83\n```rust\n```\n<!-- example-cost: gpu --->\n```rust\n```\n'
}
ebe_emptycost() { # the original's parts[0] IndexError: stops after a.md, b.md prints nothing
    ebe_ch "$1" cli a.md '```rust\n```\n'
    ebe_ch "$1" cli b.md '```rust\n```\n<!-- example-cost: \t -->\n```bash\n```\n'
    ebe_ch "$1" cli c.md '```rust\n```\n'
}
ebe_emptycost_unread() { # an empty cost line no fence looks back to is never parsed
    ebe_ch "$1" cli a.md '<!-- example-cost:  -->\nx\n```rust\n```\n<!-- example-cost:  -->\n\n\n```rust\n```\n<!-- example-cost:  -->\n```sh\n```\n'
}
ebe_lookback() {
    ebe_ch "$1" cli l.md '<!-- example-cost: gpu -->\n\n```bash\n```\n<!-- example-cost: gpu -->\n\n\n```bash\n```\n<!-- example-cost: gpu -->\ntext\n```rust\n```\n   <!-- example-cost: skip -->  \n \t\x1f\n```bash\n```\n```bash\n```\n```rust\n```\n'
}
ebe_fences() {
    ebe_ch "$1" cli f.md '```bash \t\x1f\xc2\xa0\napr gpu\n```\n```Bash\n```\n ```bash\n```\n``` bash\n```\n```bashx\n```\n````bash\n```\n```sh\n```\n```bash\na\n```rust\nb\n``` \xc2\xa0\n```rust\n\n```\n```bash\n ```\n````\n```x\n```\n```bash\nnever closed\n'
    ebe_ch "$1" cli g.md '```bash\n```'
}
ebe_escapes() {
    ebe_ch "$1" cli e.md '```bash\nsay "hi" \\ back\ttab\x7fdel\x01\x08\x1b \xc3\xa9 \xf0\x9f\x98\x80 \xef\xbf\xbf\n\x1f\n```\n'
}
ebe_breaks() {
    ebe_ch "$1" cli crlf.md 'Text\r\n```bash\r\napr tui\r\n\r\n```\r\n'
    ebe_ch "$1" cli cr.md '```rust\rfn f(){}\r```\r'
    ebe_ch "$1" cli u.md '<!-- example-cost: gpu -->\x0b```bash\x1capr\x1dgpu\xe2\x80\xa8x\xc2\x85```\x1ed\xe2\x80\xa9'
}
ebe_order() {
    local n
    for n in b.md a.md A.md .h.md .md y.md.md c.MD 'n\n.md' 'r\r.md' '\xc3\xa9.md' 'z z.md' '\xff.md' '\xee\x80\x80.md' '\xf0\x9f\x98\x80.md' 'a\xc3.md'; do
        # shellcheck disable=SC2059 # the name IS a printf fixture
        ebe_ch "$1" cli "$(printf "$n")" '```rust\n```\n'
    done
    ebe_ch "$1" lib a.md '```bash\napr tui\n```\n'
    ebe_ch "$1" other o.md '```rust\n```\n'
}
ebe_libonly() { ebe_ch "$1" lib only.md 'x\n```bash\napr eval m.apr\n```\n'; }
ebe_bom() { ebe_ch "$1" cli bom.md '\xef\xbb\xbf```bash\n```\n\xef\xbb\xbf<!-- example-cost: gpu -->\n```rust\n```\n'; }
ebe_badutf8() {
    ebe_ch "$1" cli a.md '```rust\n```\n'
    ebe_ch "$1" cli b.md '```rust\n```\nbad \xff utf-8\n'
    ebe_ch "$1" cli c.md '```rust\n```\n'
}
ebe_surrogate() { ebe_ch "$1" cli s.md '```rust\n\xed\xa0\x80\n```\n'; }
ebe_dirmd() { ebe_ch "$1" cli a.md '```rust\n```\n' && mkdir -p "$1/book/src/cli/d.md"; }
ebe_dangling() { ebe_ch "$1" cli a.md '```rust\n```\n' && ln -s nowhere "$1/book/src/cli/l.md"; }
ebe_symlink() { ebe_ch "$1" cli real.txt '```rust\n```\n' && ln -s real.txt "$1/book/src/cli/l.md"; }
ebe_unreadable() {
    ebe_ch "$1" cli a.md '```rust\n```\n'
    ebe_ch "$1" cli b.md '```rust\n```\n'
    chmod 000 "$1/book/src/cli/b.md"
}
ebe_case "no book" ebe_none
ebe_case "cli is a file" ebe_clifile
ebe_case "empty dirs" ebe_emptydirs
ebe_case "live book" ebe_live
ebe_case "cost classes and models" ebe_costs
ebe_case "cost-line forms" ebe_costlines
ebe_case "an empty cost line stops the run" ebe_emptycost
ebe_case "an empty cost line nothing reads" ebe_emptycost_unread
ebe_case "look-back" ebe_lookback
ebe_case "fence forms" ebe_fences
ebe_case "json escapes" ebe_escapes
ebe_case "line breaks" ebe_breaks
ebe_case "order and names" ebe_order
ebe_case "lib only" ebe_libonly
ebe_case "BOM" ebe_bom
ebe_case "invalid utf-8 stops the run" ebe_badutf8
ebe_case "encoded surrogate" ebe_surrogate
ebe_case "a directory named .md" ebe_dirmd
ebe_case "dangling link" ebe_dangling
ebe_case "link to a chapter" ebe_symlink
ebe_case "unreadable chapter" ebe_unreadable
chmod -R u+rw "$tmp/ebe"


# --- 9. llama-fit-verdict ---
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


# --- 8. complexity-rows (scripts/lib/complexity_rows.py, kept: check_complexity_ratchet.sh
# still calls it under N-1). Exact exit code too: the original's 2 (bad threshold, no
# document) and 1 (it raised) are different refusals.
CXR_PY=scripts/lib/complexity_rows.py
c="$tmp/cxr"
mkdir -p "$c"
cx() { printf '%s' "$2" >"$c/$1.json"; }
# cxr NAME CYC COG ARGS... -- a threshold of "-" leaves that variable unset on both sides
cxr() {
    local name="$1" cyc="$2" cog="$3" prc rrc
    shift 3
    local envs=(env -u CX_MAX_CYCLOMATIC -u CX_MAX_COGNITIVE)
    [[ "$cyc" == "-" ]] || envs+=("CX_MAX_CYCLOMATIC=$cyc")
    [[ "$cog" == "-" ]] || envs+=("CX_MAX_COGNITIVE=$cog")
    "${envs[@]}" "$PY" "$CXR_PY" "$@" </dev/null >/dev/null 2>&1 && prc=0 || prc=$?
    "${envs[@]}" "$BIN" complexity-rows "$@" </dev/null >/dev/null 2>&1 && rrc=0 || rrc=$?
    if [[ "$prc" -ne "$rrc" ]]; then
        fail=$((fail + 1))
        echo "MISMATCH: complexity-rows $name exit code (py $prc, rust $rrc)" >&2
        return
    fi
    check "complexity-rows $name" "$tmp/empty" \
        "${envs[@]}" "$PY" "$CXR_PY" "$@" -- "${envs[@]}" "$BIN" complexity-rows "$@"
}
fn() { printf '{"name":%s,"metrics":{"cyclomatic":%s,"cognitive":%s}}' "$1" "$2" "$3"; }
cx main "{\"files_analyzed\":4,\"files\":[
 {\"path\":\"./crates/b/src/x.rs\",\"functions\":[$(fn '"fit"' 12 3),$(fn '"fit"' 4 31),$(fn '"ok"' 10 15)]},
 {\"path\":\"././a.rs\",\"functions\":[$(fn '"z"' 11 0),$(fn '"Z"' 11 0),$(fn '"é"' 11 0),$(fn '"_"' 0 16)]},
 {\"path\":\"tool.py\",\"functions\":[$(fn '"big"' 99 99)]},
 {\"path\":\"x.rs.bak\",\"functions\":[$(fn '"big"' 99 99)]}]}"
cx names "{\"files\":[{\"path\":\"n.rs\",\"functions\":[$(fn 5 20 0),$(fn true 20 0),$(fn '["a",1]' 20 0),$(fn '{"k":null,"f":1e16}' 20 0),$(fn 1e16 20 0),$(fn 0.5 20 0),$(fn 0 20 0),$(fn '""' 20 0),$(fn null 20 0),{\"metrics\":{\"cyclomatic\":20}},$(fn '"a b"' 20 0)]}]}"
cx coerce "{\"files\":[{\"path\":\"c.rs\",\"functions\":[$(fn '"s"' '"12"' '" 1_6 "'),$(fn '"f"' 11.9 -0.5),$(fn '"b"' true false),$(fn '"n"' null null),$(fn '"e"' 1e20 2),{\"name\":\"nometrics\"},{\"name\":\"nullm\",\"metrics\":null},{\"name\":\"emptym\",\"metrics\":{}},$(fn '"neg"' -40 -2)]}]}"
cx falsy '{"files_analyzed":"7","files":[{"path":null,"functions":{"a":1}},{"path":"","functions":"abc"},{"path":"p.py","functions":{"k":[]}},{"path":"e.rs","functions":[]},{"path":"f.rs"},{"functions":[{"name":"q"}]}]}'
cx nofiles1 '{"files":null}'
cx nofiles2 '{"files":0,"files_analyzed":0}'
cx nofiles3 '{"files":""}'
cx dupkeys '{"files":[{"path":"d.rs","functions":[{"name":"first","name":"last","metrics":{"cyclomatic":1,"cyclomatic":50}}]}],"files":[{"path":"d2.rs","functions":[{"name":"w","metrics":{"cognitive":40}}]}]}'
cx second "{\"files\":[{\"path\":\"crates/b/src/x.rs\",\"functions\":[$(fn '"fit"' 30 1)]},{\"path\":\"new.rs\",\"functions\":[$(fn '"n"' 11 0)]}]}"
cx bad_list '[]'
cx bad_files_dict '{"files":{"a.rs":[]}}'
cx bad_files_str '{"files":"a.rs"}'
cx bad_files_num '{"files":3}'
cx bad_entry '{"files":["a.rs"]}'
cx bad_path_num '{"files":[{"path":5,"functions":[]}]}'
cx bad_path_list '{"files":[{"path":["a.rs"]}]}'
cx bad_funcs_num '{"files":[{"path":"a.py","functions":7}]}'
cx bad_funcs_dict_rs '{"files":[{"path":"a.rs","functions":{"f":1}}]}'
cx bad_funcs_str_rs '{"files":[{"path":"a.rs","functions":"f"}]}'
cx bad_func '{"files":[{"path":"a.rs","functions":[3]}]}'
cx bad_metrics_list '{"files":[{"path":"a.rs","functions":[{"metrics":[1]}]}]}'
cx bad_metrics_str '{"files":[{"path":"a.rs","functions":[{"metrics":"m"}]}]}'
cx bad_metric_str '{"files":[{"path":"a.rs","functions":[{"metrics":{"cognitive":"x"}}]}]}'
cx bad_metric_us '{"files":[{"path":"a.rs","functions":[{"metrics":{"cyclomatic":"1__2"}}]}]}'
cx bad_metric_hex '{"files":[{"path":"a.rs","functions":[{"metrics":{"cyclomatic":"0x10"}}]}]}'
cx bad_metric_list '{"files":[{"path":"a.rs","functions":[{"metrics":{"cyclomatic":[1]}}]}]}'
cx bad_analyzed '{"files":[],"files_analyzed":"many"}'
cx bad_json '{"files":['
printf '\xef\xbb\xbf{"files":[]}' >"$c/bom.json"
printf '{"files":[{"path":"\xff.rs"}]}' >"$c/badutf8.json"
: >"$c/empty.json"

cxr "rows: sort, max on collision, ./ once, .rs only" 10 15 "$c/main.json"
cxr "names: str() of every JSON type, falsy -> ?" 10 15 "$c/names.json"
cxr "metrics: int() of str, float, bool, null, missing" 10 15 "$c/coerce.json"
cxr "falsy path/functions, len() of dict and str" 10 15 "$c/falsy.json"
cxr "files null" 10 15 "$c/nofiles1.json"
cxr "files 0" 10 15 "$c/nofiles2.json"
cxr "files empty string" 10 15 "$c/nofiles3.json"
cxr "duplicate keys: last value wins" 10 15 "$c/dupkeys.json"
cxr "two documents: a key collides across them" 10 15 "$c/main.json" "$c/second.json"
cxr "same document twice" 10 15 "$c/main.json" "$c/main.json"
cxr "negative thresholds" -3 -1 "$c/coerce.json"
cxr "zero thresholds" 0 0 "$c/main.json"
cxr "thresholds with whitespace" " 10 " "	15
" "$c/main.json"
cxr "high thresholds: no rows" 1000 1000 "$c/main.json"
for b in bad_list bad_files_dict bad_files_str bad_files_num bad_entry bad_path_num bad_path_list \
    bad_funcs_num bad_funcs_dict_rs bad_funcs_str_rs bad_func bad_metrics_list bad_metrics_str \
    bad_metric_str bad_metric_us bad_metric_hex bad_metric_list bad_analyzed bad_json bom badutf8 empty; do
    cxr "raises: $b, nothing printed" 10 15 "$c/main.json" "$c/$b.json"
done
cxr "missing document" 10 15 "$c/main.json" "$c/no-such.json"
cxr "document is a directory" 10 15 "$c"
cxr "no document" 10 15
cxr "CX_MAX_CYCLOMATIC unset" - 15 "$c/main.json"
cxr "CX_MAX_COGNITIVE unset" 10 - "$c/main.json"
cxr "both unset, no document" - -
cxr "threshold empty" "" 15 "$c/main.json"
cxr "threshold +1" 10 "+1" "$c/main.json"
cxr "threshold 1_0" "1_0" 15 "$c/main.json"
cxr "threshold 1.5" "1.5" 15 "$c/main.json"
cxr "threshold --5 (int() raises)" "--5" 15 "$c/main.json"
cxr "threshold abc beats a missing document" abc 15 "$c/no-such.json"
cxr "--help is a path" 10 15 --help
cxr "-x is a path" 10 15 -x "$c/main.json"

ran=$((pass + fail))
echo "ci_tools_py_parity: $pass/$ran identical (declared $EXPECTED_CASES)"
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
