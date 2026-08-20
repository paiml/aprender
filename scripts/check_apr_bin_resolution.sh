#!/usr/bin/env bash
# check_apr_bin_resolution.sh — apr_bin.sh must resolve the binary built from
# HEAD, whichever profile it happens to live in.
#
# WHY THIS EXISTS
# ---------------
# `apr_bin.sh` used to return `target/release/apr` whenever it existed, then
# hand it to the freshness check, which refused it. Measured on a real
# checkout, with `debug/apr` built from HEAD sitting beside a stale
# `release/apr`:
#
#   STALE apr BINARY
#     resolved : .../target/release/apr
#     reports  : apr 0.60.0 (v0.60.0+no-git)
#     HEAD     : 75d6610d8
#
# It hard-failed and told the caller to `cargo install` while a provably
# correct binary sat in the next directory. Every gate that sources apr_bin.sh
# broke in that state, and the trigger is only "you ran `cargo build --release`
# in this checkout once", which is most of them.
#
# `check_apr_bin_pinned.sh` cannot catch this: it asserts that CALLERS pin the
# binary, never that the resolver picks the right one. Nothing exercised
# resolution ORDER, so this does.
#
# End-to-end on a throwaway git checkout with fabricated `apr` binaries -- shell
# scripts that print a version string. The resolver only ever runs
# `--version` and compares it to HEAD, so a script is a faithful stand-in and
# the test costs no cargo build.
#
#   bash scripts/check_apr_bin_resolution.sh
#
# Rows:
#   1. stale release + fresh debug -> resolves DEBUG        (the regression)
#   2. fresh release + stale debug -> resolves RELEASE      (order still works)
#   3. both stale                  -> FAILS CLOSED          (not made permissive)
#   4. no candidate matches, none exist -> FAILS            (no false success)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APR_BIN_SH="${REPO_ROOT}/scripts/apr_bin.sh"

fails=0

# A throwaway checkout whose HEAD sha we know, with a fake target dir.
make_fixture() {
    local dir="$1" release_ver="$2" debug_ver="$3"
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" config user.email t@t
    git -C "$dir" config user.name t
    # A minimal workspace so `cargo metadata` answers with <dir>/target.
    # Written line by line rather than from a heredoc: bashrs parses an embedded
    # heredoc as shell, so TOML `name = "x"` is reported as SC1007 -- a false
    # positive that would fail the lint gate.
    {
        printf '[package]\n'
        printf 'name = "apr-bin-resolution-fixture"\n'
        printf 'version = "0.0.0"\n'
        printf 'edition = "2021"\n'
    } > "$dir/Cargo.toml"
    mkdir -p "$dir/src"
    echo 'fn main() {}' > "$dir/src/main.rs"
    git -C "$dir" add -A
    # -c core.hooksPath=/dev/null: the developer's global pre-commit hook fires in
    # the fixture repo otherwise and buries this test's output in its own report.
    git -C "$dir" -c core.hooksPath=/dev/null commit -qm fixture

    local head
    head=$(git -C "$dir" rev-parse --short HEAD)

    mkdir -p "$dir/target/release" "$dir/target/debug"
    for pair in "release:$release_ver" "debug:$debug_ver"; do
        local profile="${pair%%:*}" want="${pair#*:}"
        [ "$want" = "none" ] && continue
        # Built in two steps, and the sha kept off any line that also holds a
        # `[ ]` test: bashrs reads the parens of "apr 0.63.0 (sha)" as unescaped
        # parens inside a test expression (SC1028) when they share a line.
        local ver sha
        sha="deadbeef"
        if [ "$want" = "fresh" ]; then
            sha="$head"
            ver="apr 0.63.0"
        else
            ver="apr 0.60.0"
        fi
        ver="$ver ${sha}"
        printf '#!/usr/bin/env bash\necho "%s"\n' "$ver" > "$dir/target/$profile/apr"
        chmod +x "$dir/target/$profile/apr"
    done
    printf '%s\n' "$head"
}

# Resolve inside the fixture, with CARGO_HOME pointed at an empty dir so the
# `cargo install` fallback cannot supply a candidate we did not stage.
resolve_in() {
    local dir="$1"
    ( cd "$dir" && CARGO_HOME="$dir/.empty-cargo" APR_BIN= bash -c '. '"$APR_BIN_SH"' >/dev/null 2>&1 && printf "%s" "$APR"' )
}

row() {
    local name="$1" release_ver="$2" debug_ver="$3" expect="$4"
    local dir; dir="$(mktemp -d)"
    # Refuse to proceed on an empty or absurd temp dir: everything below ends in
    # `rm -rf "$dir"`, and bashrs SEC011 is right that an unvalidated one is a
    # loaded gun. This is a real finding, not one of its false positives.
    case "$dir" in
        /tmp/*|/var/folders/*) : ;;
        *) printf 'FAIL  mktemp -d gave %s, refusing to rm -rf it\n' "${dir:-<empty>}"; fails=1; return ;;
    esac
    make_fixture "$dir" "$release_ver" "$debug_ver" >/dev/null
    local got rc
    got="$(resolve_in "$dir")"; rc=$?

    if [ "$expect" = "FAIL" ]; then
        if [ "$rc" -ne 0 ] || [ -z "$got" ]; then
            printf 'ok    %s -> refused, as it must\n' "$name"
        else
            printf 'FAIL  %s -> accepted %s; the guard was made permissive\n' "$name" "$got"; fails=1
        fi
    else
        case "$got" in
            */target/"$expect"/apr)
                printf 'ok    %s -> %s\n' "$name" "$expect" ;;
            *)
                printf 'FAIL  %s -> got [%s], expected the %s binary\n' "$name" "$got" "$expect"; fails=1 ;;
        esac
    fi
    rm -rf "${dir:?refusing to rm an empty path}"
}

printf '=== apr_bin.sh resolves the HEAD-built binary (check_apr_bin_resolution.sh) ===\n'

row 'stale release + fresh debug' stale fresh debug
row 'fresh release + stale debug' fresh stale release
row 'both stale'                  stale stale FAIL
row 'no binaries at all'          none  none  FAIL

printf '\n'
if [ "$fails" -ne 0 ]; then
    printf 'FAIL: apr_bin.sh does not resolve by freshness.\n'
    exit 1
fi
printf 'PASS: resolution follows the evidence, and still fails closed.\n'
exit 0
