#!/usr/bin/env bash
#
# check_orphaned_source_files.sh — a .rs file under src/ that nothing declares is dead.
#
# WHY THIS EXISTS (#2473)
# -----------------------
# 15 of the 16 *_tests.rs files in crates/aprender-test-lib were wired into no
# `mod` declaration, no `include!()` and no `#[path]`. ~26,000 lines and 1,741
# test functions sat in the tree looking exactly like tests, and the compiler
# never read a byte of them. Appending `this is not rust;` to any of them left
# `cargo check` exiting 0.
#
# Nothing detected that for the 15 months they sat there, because nothing looks.
# A dead source file is invisible to every gate we run: fmt skips it, clippy
# skips it, the test count does not move, and coverage cannot report on a file
# that was never compiled. The only signal is that a mutation to it does not
# turn the build RED — which no gate was asking.
#
# WHAT IT CHECKS
# --------------
# For every .rs file under a crate's src/, some file in that same crate must
# claim it, via one of:
#
#     mod <stem>;                 (incl. pub / pub(crate) / pub(in path) forms)
#     include!("…/<name>.rs")
#     #[path = "…/<name>.rs"]
#     path = "…/<name>.rs"        in that crate's Cargo.toml
#
# Unclaimed means no module path reaches it, which means it does not compile.
#
# This is a SYNTACTIC reachability check, one level deep: it proves a file is
# named somewhere, not that the namer is itself reachable from the crate root.
# A cluster of orphans that all declare each other would still pass. That is a
# deliberate trade — the check is cheap, runs on every PR, and catches the
# defect class that actually occurs (a file nobody ever wired up). Proving true
# reachability needs the compiler, and the compiler's answer is the mutation
# test described above, which cannot run in a gate.
#
# EXEMPTIONS, AND WHY EACH IS SAFE
# --------------------------------
#   lib.rs, main.rs, mod.rs, build.rs   cargo/rustc entry points, claimed by
#                                       construction, never by a `mod` line.
#   src/bin/**                          auto-discovered binary targets. Verified
#                                       LIVE by mutation: corrupting
#                                       src/bin/apr.rs exits 101.
#   Cargo.toml `path = "…"` targets     explicitly claimed by the manifest.
#
# BASELINE RATCHET
# ----------------
# This guard was written AFTER the debt existed. Pre-existing orphans live in
# scripts/orphaned_source_files_baseline.txt, one path per line. The guard fails
# on any orphan NOT in that file, so new ones are blocked from day one, and it
# also fails when a baseline entry stops being an orphan — the committed list
# can only shrink. Regenerate with --update-baseline, which refuses to add
# entries.
#
# VACUITY GUARD
# -------------
# A guard whose universe is built from the wrong side passes while scanning
# nothing (aprender: the route index, and FALSIFY-MONO-011 which never scanned
# one crate). This one refuses to pass unless it scanned at least MIN_FILES
# files across at least MIN_CRATES crates. Note the universe here is built from
# `find` over the filesystem, and the defect — a missing `mod` line — cannot
# remove a file from that universe. That is the correct side to build it from.
#
# SELF-TEST
# ---------
#   bash scripts/check_orphaned_source_files.sh --self-test
# builds a hermetic fixture crate with an eleven-case must-flag / must-not-flag
# table covering every claim form and the regex's known confusables (commented
# out `mod`, inline `mod x { }` with no semicolon, a stem that is a substring of
# another). It then mutates the ratchet's own input to prove the comparison
# turns RED. Verification Discipline #7: re-run the table, never re-read the
# pattern.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Absolute, because the self-test re-invokes this script and scan_tree runs from
# a different directory. A relative $0 would not resolve there.
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_BASELINE="${REPO_ROOT}/scripts/orphaned_source_files_baseline.txt"
BASELINE="${BASELINE:-$DEFAULT_BASELINE}"
MIN_FILES="${MIN_FILES:-4000}"
MIN_CRATES="${MIN_CRATES:-40}"

GUARD_TMP=""
cleanup() {
    if [ -n "$GUARD_TMP" ]; then
        rm -rf "$GUARD_TMP"
    fi
}
trap cleanup EXIT

# Scratch for the per-crate scan. DELIBERATELY GLOBAL: an EXIT trap fires after
# the function's frame is gone, so a `local` here leaves the cleanup itself
# aborting with `unbound variable` under `set -u`. The sibling guard
# check_contract_test_binding.sh carries the same note; this script reproduced
# the bug before heeding it.
SCAN_SCRATCH=""
scan_cleanup() {
    if [ -z "$SCAN_SCRATCH" ]; then
        return 0
    fi
    if [ "$SCAN_SCRATCH" = "/" ]; then
        return 0
    fi
    rm -rf "$SCAN_SCRATCH"
}

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Emit every orphan under $1 (a tree root), one relative path per line, sorted.
# Also writes "<files> <crates>" to the file named by $2 so the caller can
# enforce the vacuity floor.
# ---------------------------------------------------------------------------
scan_tree() {
    # Runs entirely in a SUBSHELL: the `cd` below must not leak into the caller,
    # which still needs its own cwd (and, in the self-test, re-invokes this
    # script afterwards).
    ( _scan_tree_impl "$@" )
}

_scan_tree_impl() {
    local root="$1" stats_out="$2"
    local scanned=0 crates=0
    local crate_src crate_root base stem f manifest
    local claimed_f cands_f
    SCAN_SCRATCH="$(mktemp -d)"
    trap scan_cleanup EXIT
    claimed_f="$SCAN_SCRATCH/claimed.txt"
    cands_f="$SCAN_SCRATCH/cands.txt"

    cd "$root" || die "check_orphaned_source_files: cannot enter $root"

    # A crate is any directory holding a Cargo.toml with a src/ beside it.
    while IFS= read -r manifest; do
        crate_root="$(dirname "$manifest")"
        crate_src="$crate_root/src"
        [ -d "$crate_src" ] || continue
        crates=$((crates + 1))

        # ---- Build the claimed set for THIS crate -------------------------
        # Streams into a sorted file rather than an associative array: `join`
        # below is O(n) on sorted input, and it keeps the script free of bash-4
        # array syntax that shell linters parse as a `[` test.
        #
        # `mod NAME;` — anchored at line start after optional whitespace, so a
        # commented-out `// mod ghost;` never claims ghost.rs. The trailing `;`
        # is required, so an inline `mod ghost { }` never claims it either.
        {
            grep -rhoE '^[[:space:]]*(pub[[:space:]]*(\([^)]*\))?[[:space:]]+)?mod[[:space:]]+[A-Za-z0-9_]+[[:space:]]*;' \
                "$crate_src" --include='*.rs' 2>/dev/null \
                | sed -E 's/^.*[^A-Za-z0-9_]mod[[:space:]]+([A-Za-z0-9_]+)[[:space:]]*;.*$/\1/;s/^mod[[:space:]]+([A-Za-z0-9_]+)[[:space:]]*;.*$/\1/'
            # include!("…") and #[path = "…"] — matched on basename, since the
            # module tree position is irrelevant to whether the file compiles.
            grep -rhoE '(include!|#\[[[:space:]]*path[[:space:]]*=)[^;]*"[^"]+\.rs"' \
                "$crate_src" --include='*.rs' 2>/dev/null \
                | grep -oE '"[^"]+\.rs"' | tr -d '"' \
                | sed -E 's|^.*/||;s|\.rs$||'
            # Manifest-declared targets ([[bin]], [[example]], [[test]], [lib]).
            grep -hoE '^[[:space:]]*path[[:space:]]*=[[:space:]]*"[^"]+\.rs"' "$manifest" 2>/dev/null \
                | grep -oE '"[^"]+\.rs"' | tr -d '"' \
                | sed -E 's|^.*/||;s|\.rs$||'
        } | grep -v '^$' | LC_ALL=C sort -u > "$claimed_f"

        # ---- Test every file in this crate against the claimed set --------
        # Emit "stem<TAB>path" for each candidate, then `join -v1` keeps only
        # the stems with no match in the claimed set. That is the orphan list.
        : > "$cands_f"
        while IFS= read -r f; do
            base="${f##*/}"
            stem="${base%.rs}"
            case "$base" in
                lib.rs|main.rs|mod.rs|build.rs) continue ;;
            esac
            case "$f" in
                "$crate_src"/bin/*) continue ;;
            esac
            scanned=$((scanned + 1))
            printf '%s\t%s\n' "$stem" "${f#./}" >> "$cands_f"
        done < <(find "$crate_src" -name '*.rs' -type f)

        LC_ALL=C sort -k1,1 "$cands_f" \
            | LC_ALL=C join -t "$(printf '\t')" -v1 -1 1 -2 1 - "$claimed_f" \
            | cut -f2
    done < <(find . -name Cargo.toml -not -path '*/target/*' -not -path '*/.git/*' | LC_ALL=C sort)

    printf '%s %s\n' "$scanned" "$crates" > "$stats_out"
}

# ---------------------------------------------------------------------------
# Self-test: hermetic fixture + case table, then a ratchet mutation.
# ---------------------------------------------------------------------------
self_test() {
    GUARD_TMP="$(mktemp -d)"
    local fx="$GUARD_TMP/fixture" fail=0
    mkdir -p "$fx/src/sub" "$fx/src/bin"

    # printf, not a heredoc: shell linters parse heredoc bodies as shell, and
    # this fixture's bodies are TOML and Rust.
    printf '%s\n' \
        '[package]' \
        'name = "fixture"' \
        '[[example]]' \
        'name = "ex"' \
        'path = "src/cargo_pathed.rs"' > "$fx/Cargo.toml"

    printf '%s\n' \
        'mod claimed;' \
        'pub(crate) mod claimed_pub;' \
        'pub (in crate::sub) mod claimed_vis;' \
        'mod sub;' \
        '// mod ghost_commented;' \
        'mod ghost_inline { }' \
        'include!("included.rs");' \
        '#[path = "pathed.rs"]' \
        'mod aliased;' > "$fx/src/lib.rs"

    # MUST NOT FLAG
    : > "$fx/src/claimed.rs"          # plain `mod x;`
    : > "$fx/src/claimed_pub.rs"      # `pub(crate) mod x;`
    : > "$fx/src/claimed_vis.rs"      # `pub (in path) mod x;`
    : > "$fx/src/included.rs"         # include!()
    : > "$fx/src/pathed.rs"           # #[path]
    : > "$fx/src/cargo_pathed.rs"     # Cargo.toml path =
    : > "$fx/src/sub/mod.rs"          # mod.rs is exempt
    : > "$fx/src/bin/tool.rs"         # src/bin/** is exempt
    # MUST FLAG
    : > "$fx/src/orphan.rs"           # declared nowhere
    : > "$fx/src/ghost_commented.rs"  # only a commented-out `mod`
    : > "$fx/src/ghost_inline.rs"     # only an inline `mod x { }`, no `;`

    local must_flag="ghost_commented ghost_inline orphan"
    local must_not_flag="claimed claimed_pub claimed_vis included pathed cargo_pathed tool"

    local got
    got="$(scan_tree "$fx" "$GUARD_TMP/stats" | sed 's|.*/||;s|\.rs$||' | LC_ALL=C sort | tr '\n' ' ')"
    got=" $got "

    local c
    for c in $must_flag; do
        case "$got" in
            *" $c "*) ;;
            *) printf 'SELF-TEST FAIL: %s should have been flagged, was not\n' "$c" >&2; fail=1 ;;
        esac
    done
    for c in $must_not_flag; do
        case "$got" in
            *" $c "*) printf 'SELF-TEST FAIL: %s was flagged, should not have been\n' "$c" >&2; fail=1 ;;
            *) ;;
        esac
    done

    # Ratchet mutation: an orphan absent from the baseline MUST turn it RED.
    local mut_baseline="$GUARD_TMP/baseline.txt"
    printf 'src/ghost_commented.rs\nsrc/ghost_inline.rs\nsrc/orphan.rs\n' > "$mut_baseline"
    if ! BASELINE="$mut_baseline" MIN_FILES=1 MIN_CRATES=1 \
        bash "$SCRIPT_PATH" --tree "$fx" >/dev/null 2>&1; then
        printf 'SELF-TEST FAIL: complete baseline should PASS, it did not\n' >&2
        fail=1
    fi
    printf 'src/ghost_commented.rs\nsrc/ghost_inline.rs\n' > "$mut_baseline"   # drop one
    if BASELINE="$mut_baseline" MIN_FILES=1 MIN_CRATES=1 \
        bash "$SCRIPT_PATH" --tree "$fx" >/dev/null 2>&1; then
        printf 'SELF-TEST FAIL: baseline missing an orphan should FAIL, it passed\n' >&2
        fail=1
    fi
    # And a baseline naming a file that is NOT an orphan must also turn it RED.
    printf 'src/claimed.rs\nsrc/ghost_commented.rs\nsrc/ghost_inline.rs\nsrc/orphan.rs\n' > "$mut_baseline"
    if BASELINE="$mut_baseline" MIN_FILES=1 MIN_CRATES=1 \
        bash "$SCRIPT_PATH" --tree "$fx" >/dev/null 2>&1; then
        printf 'SELF-TEST FAIL: stale baseline entry should FAIL, it passed\n' >&2
        fail=1
    fi

    [ "$fail" -eq 0 ] || die "check_orphaned_source_files: SELF-TEST FAILED"
    printf 'check_orphaned_source_files: SELF-TEST PASSED (11 cases + 3 ratchet mutations)\n'
}

# ---------------------------------------------------------------------------
main() {
    local tree="$REPO_ROOT" update=0

    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; exit 0 ;;
            --update-baseline) update=1; shift ;;
            --tree) tree="$2"; shift 2 ;;
            *) die "check_orphaned_source_files: unknown argument: $1" ;;
        esac
    done

    GUARD_TMP="${GUARD_TMP:-$(mktemp -d)}"
    [ -n "$GUARD_TMP" ] || GUARD_TMP="$(mktemp -d)"
    local found="$GUARD_TMP/found.txt" stats="$GUARD_TMP/stats.txt"
    scan_tree "$tree" "$stats" | LC_ALL=C sort > "$found"

    local scanned crates
    read -r scanned crates < "$stats"

    if [ "$update" -eq 1 ]; then
        local before=0
        [ -f "$BASELINE" ] && before=$(grep -cve '^[[:space:]]*$' -e '^#' "$BASELINE" || true)
        local after
        after=$(wc -l < "$found")
        if [ "$before" -gt 0 ] && [ "$after" -gt "$before" ]; then
            die "check_orphaned_source_files: refusing to RAISE the baseline ($before -> $after). Wire the new file up or delete it."
        fi
        {
            printf '# Orphaned .rs files under src/ that no mod/include!/#[path]/Cargo.toml claims.\n'
            printf '# Pre-existing debt at the time the guard landed (#2473). This list may only SHRINK.\n'
            printf '# Regenerate: bash scripts/check_orphaned_source_files.sh --update-baseline\n'
            cat "$found"
        } > "$BASELINE"
        printf 'check_orphaned_source_files: baseline updated (%s entries, was %s)\n' "$after" "$before"
        exit 0
    fi

    # Vacuity floor — a scan that measured nothing must never read as clean.
    if [ "$scanned" -lt "$MIN_FILES" ] || [ "$crates" -lt "$MIN_CRATES" ]; then
        die "VACUOUS: scanned $scanned files across $crates crates (floor: $MIN_FILES/$MIN_CRATES). The universe collapsed; this is a broken guard, not a clean tree."
    fi

    [ -f "$BASELINE" ] || die "check_orphaned_source_files: missing baseline $BASELINE"
    grep -ve '^[[:space:]]*$' -e '^#' "$BASELINE" | LC_ALL=C sort > "$GUARD_TMP/base.txt"

    local newly stale rc=0
    newly=$(LC_ALL=C comm -23 "$found" "$GUARD_TMP/base.txt")
    stale=$(LC_ALL=C comm -13 "$found" "$GUARD_TMP/base.txt")

    if [ -n "$newly" ]; then
        printf 'FAIL: %s source file(s) under src/ that NOTHING declares:\n\n' "$(printf '%s\n' "$newly" | wc -l)" >&2
        printf '%s\n' "$newly" | sed 's/^/  /' >&2
        printf '\nNothing compiles these. Corrupting one would not turn the build RED.\n' >&2
        printf 'Add a `mod` declaration in the parent module, or delete the file.\n' >&2
        rc=1
    fi
    if [ -n "$stale" ]; then
        printf '\nFAIL: %s baseline entry(s) are no longer orphans (wired up or deleted):\n\n' "$(printf '%s\n' "$stale" | wc -l)" >&2
        printf '%s\n' "$stale" | sed 's/^/  /' >&2
        printf '\nThe ratchet only moves down. Run: bash scripts/check_orphaned_source_files.sh --update-baseline\n' >&2
        rc=1
    fi
    [ "$rc" -eq 0 ] || exit 1

    printf 'OK: %s source files across %s crates; every one is declared (%s known orphans held at baseline)\n' \
        "$scanned" "$crates" "$(wc -l < "$GUARD_TMP/base.txt")"
}

main "$@"
