#!/usr/bin/env bash
# check_apr_bin_resolution.sh — apr_bin.sh must resolve the binary built from
# HEAD IN THIS TREE, whichever profile it happens to live in — and must name
# the condition it actually found.
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
# WHAT #2739 ADDED
# ----------------
# The rows above all vary ONE axis: which commit a binary was built from. That
# axis is not sufficient. Every git worktree of this repo shares one target
# directory (the dev shell's `cargo` function exports
# CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/$project for all of them), so
# whichever tree built last owns <target>/release/apr. Measured live from
# worktree `s2-aprbin`:
#
#   /mnt/nvme-raid0/targets/aprender/release/apr    apr 0.64.0 (de0e3e182)
#   /mnt/nvme-raid0/targets/aprender/release/apr.d  .../p3-warmup/src/bin/apr.rs
#
# and, in the same checkout at the same moment:
#
#   ~/.cargo/bin/apr                                apr 0.64.0 (50d2bc2bb)
#   ~/.cargo/.crates.toml                           path+file:///home/noah/
#                                                   actions-runner-lambda/_work/...
#
# That second one is the sharp case: its embedded SHA MATCHED HEAD exactly, so
# the pre-#2739 resolver returned it with no warning at all — a binary built in
# the CI runner's checkout, from an unknown feature set, presented as this
# tree's. An agent working #2730 was handed a sibling worktree's apr built
# WITHOUT --features cuda while working a CUDA-only path, and only noticed
# because the CUDA error text was wrong for a CUDA build.
#
# So the rows below vary a SECOND axis — which tree — and assert the MESSAGE,
# not only the exit code. "STALE" tells the reader to rebuild. That is the
# right advice for an old build of this tree and useless for someone else's
# build, so a guard that only checks "did it refuse" would pass on the wrong
# diagnosis.
#
# End-to-end on a throwaway git checkout with fabricated `apr` binaries -- shell
# scripts that print a version string, and hand-written `.d` dep-info files
# beside them. The resolver only ever runs `--version`, greps the dep-info and
# reads .crates.toml, so fabrications are faithful stand-ins and the test costs
# no cargo build.
#
#   bash scripts/check_apr_bin_resolution.sh
#
# Rows:
#   1. stale release + fresh debug      -> resolves DEBUG      (the regression)
#   2. fresh release + stale debug      -> resolves RELEASE    (order still works)
#   3. both stale                       -> FAILS CLOSED        (not made permissive)
#   4. no candidate matches, none exist -> FAILS               (no false success)
#   5. foreign fresh release, nothing else     -> FAILS, WRONG-TREE
#   6. own stale release                       -> FAILS, STALE (discrimination)
#   7. foreign fresh release + own fresh debug -> resolves DEBUG
#   8. foreign fresh release + own stale debug -> FAILS, STALE (own wins the report)
#   9. binary only in <worktree>/target, target_directory elsewhere -> resolves it
#  10. ~/.cargo/bin/apr installed from ANOTHER path  -> FAILS, WRONG-TREE
#  11. ~/.cargo/bin/apr installed from THIS path     -> resolves it

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APR_BIN_SH="${REPO_ROOT}/scripts/apr_bin.sh"

fails=0

# A throwaway checkout whose HEAD sha we know, with a fake target dir.
#
#   $4 / $5: which TREE the release/debug binary claims to have been built
#            from -- self, other, or none (no dep-info at all, which is what
#            `cargo install` leaves behind and what every pre-#2739 fixture
#            looked like).
make_fixture() {
    local dir="$1" release_ver="$2" debug_ver="$3"
    local release_owner="${4:-none}" debug_owner="${5:-none}"
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" config user.email t@t
    git -C "$dir" config user.name t
    # A minimal workspace so `cargo metadata` answers with <dir>/target, and so
    # it reports a bin target NAMED apr -- the src_path of that target is the
    # anchor apr_bin.sh greps for in the dep-info, so a fixture without one
    # would exercise the unattributable path only and prove nothing about tree
    # attribution.
    # Written line by line rather than from a heredoc: bashrs parses an embedded
    # heredoc as shell, so TOML `name = "x"` is reported as SC1007 -- a false
    # positive that would fail the lint gate.
    {
        printf '[package]\n'
        printf 'name = "apr-bin-resolution-fixture"\n'
        printf 'version = "0.0.0"\n'
        printf 'edition = "2021"\n'
    } > "$dir/Cargo.toml"
    mkdir -p "$dir/src/bin"
    echo 'fn main() {}' > "$dir/src/bin/apr.rs"
    git -C "$dir" add -A
    # -c core.hooksPath=/dev/null: the developer's global pre-commit hook fires in
    # the fixture repo otherwise and buries this test's output in its own report.
    git -C "$dir" -c core.hooksPath=/dev/null commit -qm fixture

    local head
    head=$(git -C "$dir" rev-parse --short HEAD)

    mkdir -p "$dir/target/release" "$dir/target/debug"
    stage_binary "$dir/target/release/apr" "$release_ver" "$release_owner" "$head" "$dir"
    stage_binary "$dir/target/debug/apr" "$debug_ver" "$debug_owner" "$head" "$dir"
    printf '%s\n' "$head"
}

# One fabricated binary, plus the dep-info that attributes it to a tree.
stage_binary() {
    local path="$1" want="$2" owner="$3" head="$4" dir="$5"
    [ "$want" = "none" ] && return 0
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
    printf '#!/usr/bin/env bash\necho "%s"\n' "$ver" > "$path"
    chmod +x "$path"

    # Cargo writes <binary>.d beside every binary it links: a make-style rule
    # whose right-hand side is the absolute path of every source that went into
    # it. `other` uses "$dir-other", a sibling path that does NOT contain
    # "$dir/" as a substring -- so a resolver matching on the anchor cannot
    # accept it by accident.
    case "$owner" in
        self)  printf '%s: %s/src/bin/apr.rs\n' "$path" "$dir" > "$path.d" ;;
        other) printf '%s: %s-other/src/bin/apr.rs\n' "$path" "$dir" > "$path.d" ;;
        *)     rm -f "$path.d" ;;
    esac
}

# The record `cargo install --path DIR` leaves behind. This is the only thing
# that attributes an installed binary to a tree: cargo builds it in a temp dir
# and copies over just the finished file, so there is no dep-info beside it.
write_crates2() {
    local out="$1" from="$2"
    {
        printf '{"installs":{'
        printf '"apr-cli 0.63.0 (path+file://%s)":' "$from"
        printf '{"version_req":null,"bins":["apr","apr-corpus-ingest"],'
        printf '"features":[],"all_features":false,"no_default_features":false,'
        printf '"profile":"release","target":"x86_64-unknown-linux-gnu","rustc":"x"}'
        printf '}}\n'
    } > "$out"
}

# Resolve inside the fixture, with CARGO_HOME pointed at an empty dir so the
# `cargo install` fallback cannot supply a candidate we did not stage.
# stderr is captured so the DIAGNOSIS can be asserted, not just the exit code.
resolve_in() {
    local dir="$1" errlog="$2" ch="${3:-$dir/.empty-cargo}" td="${4:-}"
    # Two spellings rather than CARGO_TARGET_DIR="" in one: an EMPTY
    # CARGO_TARGET_DIR is not the same as an unset one to cargo.
    if [ -n "$td" ]; then
        (
            cd "$dir" || exit 1
            APR_BIN_SH_PATH="$APR_BIN_SH" CARGO_HOME="$ch" CARGO_TARGET_DIR="$td" APR_BIN="" \
            bash -c 'source "$APR_BIN_SH_PATH" && printf "%s" "$APR"'
        ) 2>"$errlog"
        return
    fi
    (
        cd "$dir" || exit 1
        APR_BIN_SH_PATH="$APR_BIN_SH" CARGO_HOME="$ch" APR_BIN="" \
        bash -c 'source "$APR_BIN_SH_PATH" && printf "%s" "$APR"'
    ) 2>"$errlog"
}

# A temp dir this script is willing to `rm -rf`. bashrs SEC011 is right that an
# unvalidated one is a loaded gun; this is a real finding, not a false positive.
safe_tmpdir() {
    local dir
    dir="$(mktemp -d)" || return 1
    case "$dir" in
        /tmp/*|/var/folders/*) printf '%s\n' "$dir" ;;
        *) return 1 ;;
    esac
}

# Assert on the resolved path AND on the diagnosis.
#   $4 expect     : release | debug | FAIL
#   $7 expect_msg : STALE | WRONG-TREE | "" (do not care)
row() {
    local name="$1" release_ver="$2" debug_ver="$3" expect="$4"
    local release_owner="${5:-none}" debug_owner="${6:-none}" expect_msg="${7:-}"
    local dir
    if ! dir="$(safe_tmpdir)"; then
        printf 'FAIL  mktemp -d gave an unusable path, refusing to rm -rf it\n'; fails=1; return
    fi
    make_fixture "$dir" "$release_ver" "$debug_ver" "$release_owner" "$debug_owner" >/dev/null
    local got rc errlog
    errlog="$dir/stderr.log"
    got="$(resolve_in "$dir" "$errlog")"; rc=$?

    check_outcome "$name" "$expect" "$expect_msg" "$got" "$rc" "$errlog"
    rm -rf "${dir:?refusing to rm an empty path}"
}

check_outcome() {
    local name="$1" expect="$2" expect_msg="$3" got="$4" rc="$5" errlog="$6"
    local msg
    msg="$(cat "$errlog" 2>/dev/null)"

    if [ "$expect" = "FAIL" ]; then
        if [ "$rc" -eq 0 ] && [ -n "$got" ]; then
            printf 'FAIL  %s -> accepted %s; the guard was made permissive\n' "$name" "$got"; fails=1; return
        fi
    else
        case "$got" in
            */"$expect"/apr) : ;;
            *) printf 'FAIL  %s -> got [%s], expected the %s binary\n' "$name" "$got" "$expect"; fails=1; return ;;
        esac
    fi

    # The DIAGNOSIS. Asserted with a here-string, never `printf | grep -q`:
    # grep exits on the first match, the producer takes SIGPIPE, and under
    # `pipefail` the pipeline reports 141 though grep MATCHED.
    if [ -n "$expect_msg" ]; then
        if ! grep -qF -- "$expect_msg" <<<"$msg"; then
            printf 'FAIL  %s -> refused, but never said %s\n' "$name" "$expect_msg"; fails=1; return
        fi
        # And it must not say the OTHER thing. "STALE" and "WRONG-TREE" carry
        # different remedies; emitting both is emitting neither.
        local other="STALE"
        [ "$expect_msg" = "STALE" ] && other="WRONG-TREE"
        if grep -qF -- "$other" <<<"$msg"; then
            printf 'FAIL  %s -> said %s AND %s\n' "$name" "$expect_msg" "$other"; fails=1; return
        fi
        printf 'ok    %s -> refused, and diagnosed %s\n' "$name" "$expect_msg"; return
    fi

    if [ "$expect" = "FAIL" ]; then
        printf 'ok    %s -> refused, as it must\n' "$name"
    else
        printf 'ok    %s -> %s\n' "$name" "$expect"
    fi
}

printf '=== apr_bin.sh resolves THIS tree at HEAD (check_apr_bin_resolution.sh) ===\n'

# --- axis 1: which commit (pre-#2739 rows, unchanged semantics) -------------
row 'stale release + fresh debug' stale fresh debug
row 'fresh release + stale debug' fresh stale release
row 'both stale'                  stale stale FAIL
row 'no binaries at all'          none  none  FAIL

# --- axis 2: which tree (#2739) --------------------------------------------
# The registered falsification: with the only build belonging to another tree,
# refuse -- and do not call it staleness.
row 'foreign fresh release, no own build' fresh none FAIL other none 'WRONG-TREE'
# Discrimination. An honest old build of THIS tree must still read STALE, or
# the fix has simply relabelled every failure.
row 'own stale release'                   stale none FAIL self  none 'STALE'
# The SHA alone cannot decide: both candidates match HEAD, and only one of them
# is this tree's.
row 'foreign fresh + own fresh debug'     fresh fresh debug other self
# A foreign candidate is skipped even when the only own one is stale, and the
# report is about the own one.
row 'foreign fresh + own stale debug'     fresh stale FAIL other self 'STALE'

# --- defect 1, other direction: the tree's own target dir -------------------
# `cargo metadata` answers for the shell asking. The dev shell's `cargo`
# function moves the target dir; a build launched from a plain bash script does
# not get that wrapper and lands in <worktree>/target. Searching only the
# metadata answer makes the tree's own binary invisible.
row_local_target_dir() {
    local name='binary only in <worktree>/target, CARGO_TARGET_DIR elsewhere'
    local dir
    if ! dir="$(safe_tmpdir)"; then
        printf 'FAIL  mktemp -d gave an unusable path\n'; fails=1; return
    fi
    make_fixture "$dir" fresh none self none >/dev/null
    local elsewhere="$dir/elsewhere"
    mkdir -p "$elsewhere/release" "$elsewhere/debug"
    local got rc errlog
    errlog="$dir/stderr.log"
    got="$(resolve_in "$dir" "$errlog" "$dir/.empty-cargo" "$elsewhere")"; rc=$?
    check_outcome "$name" release '' "$got" "$rc" "$errlog"
    rm -rf "${dir:?refusing to rm an empty path}"
}
row_local_target_dir

# --- the `cargo install` candidate -----------------------------------------
# It leaves no dep-info, so it is attributed from $CARGO_HOME/.crates2.json,
# which records the path it was installed FROM. This is the row that fires in
# real life from a bash shell: ~/.cargo/bin/apr matched HEAD exactly and had
# been installed out of the CI runner's checkout.
row_cargo_install() {
    local name="$1" from="$2" expect="$3" expect_msg="${4:-}"
    local dir
    if ! dir="$(safe_tmpdir)"; then
        printf 'FAIL  mktemp -d gave an unusable path\n'; fails=1; return
    fi
    make_fixture "$dir" none none >/dev/null
    local head ch
    head=$(git -C "$dir" rev-parse --short HEAD)
    ch="$dir/cargo-home"
    mkdir -p "$ch/bin"
    printf '#!/usr/bin/env bash\necho "apr 0.63.0 %s"\n' "$head" > "$ch/bin/apr"
    chmod +x "$ch/bin/apr"
    # Real .crates2.json shape, including a second installed binary, so the
    # `bins` membership test is exercised rather than a whole-string compare.
    write_crates2 "$ch/.crates2.json" "$from"
    local got rc errlog
    errlog="$dir/stderr.log"
    got="$(resolve_in "$dir" "$errlog" "$ch")"; rc=$?
    if [ "$expect" = "FAIL" ]; then
        check_outcome "$name" FAIL "$expect_msg" "$got" "$rc" "$errlog"
    elif [ "$got" = "$ch/bin/apr" ]; then
        printf 'ok    %s -> the installed binary\n' "$name"
    else
        printf 'FAIL  %s -> got [%s], expected %s\n' "$name" "$got" "$ch/bin/apr"; fails=1
    fi
    rm -rf "${dir:?refusing to rm an empty path}"
}
row_cargo_install 'cargo install from ANOTHER checkout' /some/other/checkout/crates/apr-cli FAIL 'WRONG-TREE'

# The self-install case is built against the fixture's OWN path, so it cannot
# reuse the literal above.
row_cargo_install_from_self() {
    local name='cargo install from THIS checkout'
    local dir
    if ! dir="$(safe_tmpdir)"; then
        printf 'FAIL  mktemp -d gave an unusable path\n'; fails=1; return
    fi
    make_fixture "$dir" none none >/dev/null
    local head ch
    head=$(git -C "$dir" rev-parse --short HEAD)
    ch="$dir/cargo-home"
    mkdir -p "$ch/bin"
    printf '#!/usr/bin/env bash\necho "apr 0.63.0 %s"\n' "$head" > "$ch/bin/apr"
    chmod +x "$ch/bin/apr"
    write_crates2 "$ch/.crates2.json" "$dir/crates/apr-cli"
    local got rc errlog
    errlog="$dir/stderr.log"
    got="$(resolve_in "$dir" "$errlog" "$ch")"; rc=$?
    if [ "$rc" -eq 0 ] && [ "$got" = "$ch/bin/apr" ]; then
        printf 'ok    %s -> the installed binary\n' "$name"
    else
        printf 'FAIL  %s -> got [%s] rc=%s, expected the installed binary\n' "$name" "$got" "$rc"; fails=1
    fi
    rm -rf "${dir:?refusing to rm an empty path}"
}
row_cargo_install_from_self

printf '\n'
if [ "$fails" -ne 0 ]; then
    printf 'FAIL: apr_bin.sh does not resolve this tree at HEAD.\n'
    exit 1
fi
printf 'PASS: resolution follows the evidence, names the condition, and still fails closed.\n'
exit 0
