#!/usr/bin/env bash
# check_src_test_files_wired.sh — every `crates/*/src/**/*.rs` that defines a test
# must be DECLARED by something the compiler reads: a `mod`, a `#[path]`, or an
# `include!`.
#
# WHY THIS EXISTS (aprender#3809)
# ------------------------------
# `crates/apr-cli/src/commands/serve/tests_contract_enforcement.rs` holds the 25
# FALSIFY-SRV/HTTP tests that `contracts/aprender/apr-serve-v1.yaml` cites. It sat
# beside `tests.rs` with no `mod` pointing at it, so it never compiled, never ran,
# and appeared in no test count. `pv validate` checks the YAML, not that the Rust
# behind a FALSIFY id is built. A file rustc never sees is dark by CONSTRUCTION:
# no green run anywhere is evidence about it.
#
# The universe is enumerated from the SOURCE TREE (every file with a test
# attribute), never from the module tree (which can only confirm what someone
# remembered to declare).
#
# DECLARED means one of, inside the same crate:
#   * `mod <stem>` in a file that owns the directory: for D/stem.rs that is
#     D/mod.rs, D/lib.rs, D/main.rs, or the non-mod-rs parent D.rs; for
#     D/mod.rs it is the same set one level up, naming basename(D);
#   * `#[path = "…/<file>.rs"]` or `include!("…/<file>.rs")` anywhere in the crate
#     (matched on the file NAME — resolving `#[path]` relative to a nested inline
#     module is not attempted, so a same-named file elsewhere can mask a miss);
#   * a crate root: src/lib.rs, src/main.rs, src/bin/*.rs, src/bin/*/main.rs.
# The check is one level deep, not transitive: a dark file WITHOUT tests that is
# the only parent of a test file hides it. That is a known limit, not a pass.
#
# Pre-existing dark files are listed in scripts/src_test_files_unwired_baseline.txt
# with a reason. A NEW dark file fails; a baseline entry that is no longer dark
# also fails (stale amnesty), so the list only shrinks.
#
#   bash scripts/check_src_test_files_wired.sh              # check
#   bash scripts/check_src_test_files_wired.sh --list       # print every dark file
#   bash scripts/check_src_test_files_wired.sh --self-test  # case table
#
# Executed, never sourced, so `set` here affects only its own shell.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$REPO_ROOT/scripts/src_test_files_unwired_baseline.txt"

# Every pipe below is a here-string, never `producer | grep -q`: under pipefail,
# grep -q exiting on the first match SIGPIPEs the producer, and the pipeline then
# reports FAILURE for a MATCH (the 0c6932fd5 defect, in this very tree).

# Does file $1 (with `//` comments stripped) define a test?
defines_test() {
    grep -qE '#\[(tokio::)?test(\]|\()|#\[rstest|proptest! *\{' <<< "$(sed 's://.*$::' "$1")"
}

# Does file $1 declare module $2 (`mod x;`, `pub mod x;`, `pub(crate) mod x;`)?
# Comments are stripped first: a commented-out `// mod x;` is a MENTION.
declares_mod() {
    [ -f "$1" ] || return 1
    grep -qE "^[[:space:]]*(pub(\([a-z:]+\))?[[:space:]]+)?mod[[:space:]]+(r#)?$2[[:space:]]*;" \
        <<< "$(sed 's://.*$::' "$1")"
}

# The file NAMES that crate src dir $1 reaches through #[path] or include!,
# comments stripped, one per line. Computed once per crate.
named_files() {
    local src="$1"
    find "$src" -name '*.rs' -type f -exec sed 's://.*$::' {} + \
        | grep -oE '(#\[path[[:space:]]*=[[:space:]]*"|include!\([[:space:]]*")[^"]*"' \
        | sed -E 's/.*"([^"]*)"$/\1/; s:.*/::' | sort -u
    return 0
}

# Is file $1 (absolute, under crate src dir $2) declared? $3 is named_files of $2.
is_declared() {
    local f="$1" src="$2" names="$3" rel dir stem owner
    rel="${f#"$src"/}"
    case "$rel" in
        lib.rs|main.rs|bin/*.rs|bin/*/main.rs) return 0 ;;
    esac
    dir=$(dirname "$f")
    stem=$(basename "$f" .rs)
    if [ "$stem" = "mod" ]; then
        stem=$(basename "$dir")
        dir=$(dirname "$dir")
    fi
    for owner in "$dir/mod.rs" "$dir/lib.rs" "$dir/main.rs" "$dir.rs"; do
        declares_mod "$owner" "$stem" && return 0
    done
    grep -qxF "$(basename "$f")" <<< "$names" && return 0
    # A sibling that is itself include!d/#[path]ed into the directory's module
    # declares at that module's position: `mod x;` inside an include!d
    # D/hashing.rs resolves to D/x.rs (vectorize, brick/graph.rs, gpu/*).
    for owner in "$dir"/*.rs; do
        [ -f "$owner" ] || continue
        grep -qxF "$(basename "$owner")" <<< "$names" || continue
        declares_mod "$owner" "$stem" && return 0
    done
    return 1
}

# Every dark test file under root $1, as repo-relative paths, sorted.
dark_in() {
    local root="$1" src f names
    for src in "$root"/crates/*/src; do
        [ -d "$src" ] || continue
        names=$(named_files "$src")
        while IFS= read -r f; do
            defines_test "$f" || continue
            is_declared "$f" "$src" "$names" || printf '%s\n' "${f#"$root"/}"
        # Raw grep -l is a superset prefilter; defines_test re-checks without comments.
        done < <(grep -rlE --include='*.rs' '#\[(tokio::)?test(\]|\()|#\[rstest|proptest! *\{' "$src")
    done | sort -u
}

baseline_paths() {
    [ -f "$BASELINE" ] || return 0
    sed 's/#.*$//; s/[[:space:]]*$//' "$BASELINE" | grep -v '^$' | sort -u
}

self_test() {
    local fails=0 fx got want s n
    fx=$(mktemp -d) || return 1
    case "$fx" in /tmp/*|"${TMPDIR:-/tmp}"/?*) ;; *) printf 'refusing fixture dir <%s>\n' "$fx"; return 1 ;; esac
    # shellcheck disable=SC2064
    trap "rm -rf '$fx'" EXIT
    s="$fx/crates/c/src"
    mkdir -p "$s/serve/sub" "$s/serve/lost" "$s/deep/inner" "$s/bin"
    printf 'mod serve;\nmod deep;\n#[test] fn t() {}\n' > "$s/lib.rs"
    printf '#[test] fn b() {}\n' > "$s/bin/tool.rs"
    # serve/mod.rs wires tests.rs by `mod`, tests_pp14 by #[path], tests_inc by include!.
    printf '#[cfg(test)]\nmod tests;\n#[path = "tests_pp14.rs"]\nmod p;\n// mod tests_commented;\ninclude!("tests_inc.rs");\nmod sub;\n' > "$s/serve/mod.rs"
    for n in tests tests_pp14 tests_inc tests_dark tests_commented; do
        printf '#[test]\nfn %s() {}\n' "$n" > "$s/serve/$n.rs"
    done
    printf 'fn helper() {}\n' > "$s/serve/no_tests_dark.rs"
    printf '// #[test] is only mentioned here\nfn h() {}\n' > "$s/serve/mentions_test.rs"
    # non-mod-rs parents: deep.rs declares inner, deep/inner.rs declares leaf.
    printf 'pub mod inner;\n' > "$s/deep.rs"
    printf 'pub(crate) mod leaf;\n' > "$s/deep/inner.rs"
    printf '#[tokio::test]\nasync fn l() {}\n' > "$s/deep/inner/leaf.rs"
    printf '#[test] fn o() {}\n' > "$s/deep/inner/orphan.rs"
    # a mod.rs directory module its parent declares, and one it does not.
    printf '#[test] fn s() {}\n' > "$s/serve/sub/mod.rs"
    printf '#[test] fn x() {}\n' > "$s/serve/lost/mod.rs"
    # an include!d sibling declares its neighbour; a NOT-included sibling cannot.
    printf 'include!("helpers.rs");\n' >> "$s/serve/mod.rs"
    printf 'mod via_inc;\n' > "$s/serve/helpers.rs"
    printf '#[test] fn v() {}\n' > "$s/serve/via_inc.rs"
    printf 'mod via_stray;\n' > "$s/serve/stray.rs"
    printf '#[test] fn w() {}\n' > "$s/serve/via_stray.rs"
    got=$(dark_in "$fx" | tr '\n' ' ')
    want="crates/c/src/deep/inner/orphan.rs crates/c/src/serve/lost/mod.rs crates/c/src/serve/tests_commented.rs crates/c/src/serve/tests_dark.rs crates/c/src/serve/via_stray.rs "
    if [ "$got" != "$want" ]; then
        printf 'FAIL: fixture tree\n  want=<%s>\n  got =<%s>\n' "$want" "$got"
        fails=$((fails + 1))
    fi
    # Wiring the dark file must clear it (the guard is not stuck red).
    printf '#[path = "tests_dark.rs"]\nmod d;\n' >> "$s/serve/mod.rs"
    if grep -q 'tests_dark.rs' <<< "$(dark_in "$fx")"; then
        printf 'FAIL: tests_dark.rs still reported after a #[path] wired it\n'
        fails=$((fails + 1))
    fi
    # A #[path] that is COMMENTED OUT wires nothing.
    printf '// #[path = "orphan.rs"]\n' >> "$s/deep/inner.rs"
    if ! grep -q 'orphan.rs' <<< "$(dark_in "$fx")"; then
        printf 'FAIL: a commented-out #[path] was read as wiring orphan.rs\n'
        fails=$((fails + 1))
    fi
    [ -n "$fx" ] && [ "$fx" != / ] && rm -rf "$fx"
    trap - EXIT
    if [ "$fails" -ne 0 ]; then
        printf 'check_src_test_files_wired self-test: %d FAILED\n' "$fails"
        return 1
    fi
    printf 'check_src_test_files_wired self-test: all cases pass\n'
}

main() {
    case "${1:-}" in
        --self-test) self_test; return ;;
        --list) dark_in "$REPO_ROOT"; return 0 ;;
        "") ;;
        *) printf 'usage: %s [--list|--self-test]\n' "$0" >&2; return 2 ;;
    esac
    if ! self_test >/dev/null; then
        self_test
        return 1
    fi
    local dark base new stale n
    dark=$(dark_in "$REPO_ROOT")
    base=$(baseline_paths)
    new=$(comm -23 <(printf '%s\n' "$dark" | grep -v '^$') <(printf '%s\n' "$base" | grep -v '^$'))
    stale=$(comm -13 <(printf '%s\n' "$dark" | grep -v '^$') <(printf '%s\n' "$base" | grep -v '^$'))
    n=$(printf '%s\n' "$dark" | grep -c . || true)
    if [ -n "$new" ]; then
        printf 'FAIL: test files that no mod/#[path]/include! declares (never compiled, never run, #3809):\n'
        printf '%s\n' "$new" | sed 's/^/  /'
        printf 'Wire each one, delete it with a reason, or (last resort) list it in %s with a reason.\n' \
            "${BASELINE#"$REPO_ROOT"/}"
        return 1
    fi
    if [ -n "$stale" ]; then
        printf 'FAIL: baseline entries that are no longer dark; delete them from %s:\n' \
            "${BASELINE#"$REPO_ROOT"/}"
        printf '%s\n' "$stale" | sed 's/^/  /'
        return 1
    fi
    printf 'OK: every src test file is declared (%s baselined dark file(s))\n' "$n"
}

main "$@"
