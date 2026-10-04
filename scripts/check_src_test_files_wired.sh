#!/usr/bin/env bash
# check_src_test_files_wired.sh — every `crates/*/src/**/*.rs` file must be
# COMPILED: reached from a package root through `mod`, `#[path]` and `include!`.
# (The name is from the first version, which looked at test files only.)
#
# WHY THIS EXISTS (aprender#3809, widened by #4700)
# -------------------------------------------------
# `crates/apr-cli/src/commands/serve/tests_contract_enforcement.rs` holds the 25
# FALSIFY-SRV/HTTP tests that `contracts/aprender/apr-serve-v1.yaml` cites. It sat
# beside `tests.rs` with no `mod` pointing at it, so it never compiled, never ran,
# and appeared in no test count. `pv validate` checks the YAML, not that the Rust
# behind a FALSIFY id is built. A file rustc never sees is dark by CONSTRUCTION:
# no green run anywhere is evidence about it.
#
# The first version looked only at files with a test attribute, and only one
# level deep: a file counted as declared when any file named it, compiled or
# not. At 316dee2cd4 a reachability census found 138 dark files in crates that
# hold a package. That guard saw 42 of them (its baseline). 27 more held tests
# but hung below a dark parent, and 69 held no test attribute. Dark library code
# is the same defect as a dark test: a cite, a contract or a reader that lands
# in it describes code that is in no binary.
#
# The universe is every `crates/*/src` .rs file git lists (tracked, or new and
# not ignored), enumerated from the SOURCE TREE, never from the module tree
# (which can only confirm what someone remembered to declare). Reachability is
# computed by scripts/lib/src_reach.py, which documents the model, what it
# over-approximates, and its two gaps (three levels of inline modules, a
# macro-built `mod`). A compiled file listed dark is a defect of that model:
# fix the model, never baseline the file.
#
# Pre-existing dark files are listed in scripts/src_files_dark_baseline.txt with
# the reason. A NEW dark file fails; a baseline entry that is no longer dark also
# fails (stale amnesty), so the list only shrinks.
#
#   bash scripts/check_src_test_files_wired.sh              # check
#   bash scripts/check_src_test_files_wired.sh --list       # print every dark file
#   bash scripts/check_src_test_files_wired.sh --self-test  # case table
#
# Executed, never sourced, so `set` here affects only its own shell.

set -uo pipefail
# sort and comm must order paths as src_reach.py does, by code point.
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$REPO_ROOT/scripts/src_files_dark_baseline.txt"
REACH="$REPO_ROOT/scripts/lib/src_reach.py"

# Every grep -q below reads a here-string, never `producer | grep -q`: under
# pipefail, grep -q exiting on the first match SIGPIPEs the producer, and the
# pipeline then reports FAILURE for a MATCH (the 0c6932fd5 defect, in this tree).

# The dark files under repo root $1, repo-relative, sorted. A nonzero status
# means no answer (nothing listed, or nothing readable), which is never "none".
dark_in() {
    python3 "$REACH" "$1"
}

# The paths listed in baseline file $1: comments and trailing blanks stripped.
baseline_paths() {
    [ -f "$1" ] || return 0
    sed 's/#.*$//; s/[[:space:]]*$//' "$1" | grep -v '^$' | sort -u
}

# judge DARK BASELINE -- the verdict on the dark set DARK against file BASELINE.
judge() {
    local dark="$1" baseline="$2" base new stale n
    base=$(baseline_paths "$baseline")
    new=$(comm -23 <(printf '%s\n' "$dark" | grep -v '^$') <(printf '%s\n' "$base" | grep -v '^$'))
    stale=$(comm -13 <(printf '%s\n' "$dark" | grep -v '^$') <(printf '%s\n' "$base" | grep -v '^$'))
    n=$(printf '%s\n' "$dark" | grep -c . || true)
    if [ -n "$new" ]; then
        printf 'FAIL: crates/*/src files that no package compiles (no mod/#[path]/include! reaches them, #3809):\n'
        printf '%s\n' "$new" | sed 's/^/  /'
        printf 'Wire each one, delete it with a reason, or (last resort) list it in %s with a reason.\n' \
            "${baseline#"$REPO_ROOT"/}"
        printf 'If one IS compiled, scripts/lib/src_reach.py is wrong: fix the model, never list the file.\n'
        return 1
    fi
    if [ -n "$stale" ]; then
        printf 'FAIL: baseline entries that are no longer dark; delete them from %s:\n' \
            "${baseline#"$REPO_ROOT"/}"
        printf '%s\n' "$stale" | sed 's/^/  /'
        return 1
    fi
    printf 'OK: every crates/*/src file is compiled (%s baselined dark file(s))\n' "$n"
}

# check_at ROOT BASELINE -- the whole check, for the tree at ROOT.
check_at() {
    local dark
    if ! dark=$(dark_in "$1"); then
        printf 'FAIL: scripts/lib/src_reach.py gave no answer for %s (its message is above); no answer is no pass\n' "$1"
        return 1
    fi
    judge "$dark" "$2"
}

# ---------------------------------------------------------------------------
# SELF-TEST: a fixture repo (git init, so src_reach.py lists it as it lists the
# real tree) whose every crates/*/src file is a row of CASES, then mutations,
# the baseline verdicts, and the no-answer paths. Rule 7: re-run this table
# rather than re-read the model. WANT is compiled or dark; unlisted is a file
# git does not list, which must be absent from the output like a compiled one.
CASES='
# PATH (under crates/)   WANT      WHY
p/src/lib.rs             compiled  src/lib.rs is a package root
p/src/a.rs               compiled  `mod a;` in lib.rs
p/src/a/b.rs             compiled  `pub(crate) mod b;` in the non-mod-rs a.rs names a/b.rs
p/src/c/mod.rs           compiled  `mod c;` names c/mod.rs
p/src/c/d.rs             compiled  `mod d;` in the mod-rs c/mod.rs names c/d.rs
p/src/e.rs               compiled  include!("e.rs")
p/src/x/f.rs             compiled  #[path = "x/f.rs"]
p/src/outer/g.rs         compiled  `mod g;` inside an inline `mod outer {`
p/src/i.rs               compiled  `mod i;` in lib.rs
p/src/tests.rs           compiled  #[cfg(test)] mod tests;
p/src/type.rs            compiled  mod r#type;
p/src/j.rs               compiled  /* mod j; */ counts: only // comments are stripped, which errs toward compiled
p/src/gen_root.rs        compiled  a "*.rs" literal in build.rs
p/src/bin/x.rs           compiled  src/bin/*.rs is a root
p/src/bin/y/main.rs      compiled  src/bin/*/main.rs is a root
p/src/z.rs               compiled  #[path = "../../z.rs"] in bin/y/main.rs
p/src/w.rs               compiled  #[path] in the tests/it.rs root
p/src/gen/part.rs        compiled  include!("../gen/part.rs") in c/mod.rs
p/src/gen/chain.rs       compiled  include!("chain.rs") in gen/part.rs
p/src/c/y.rs             compiled  `mod y;` in gen/chain.rs resolves from c/, two include!()s up
p/src/c/inl_body.rs      compiled  include!() inside the inline `mod inl {` of c/mod.rs
p/src/c/inl/v.rs         compiled  `mod v;` in c/inl_body.rs resolves from c/inl/
p/src/zz/mod.rs          compiled  `mod zz;`
p/src/aa/mid.rs          compiled  include!("../aa/mid.rs") in zz/mod.rs
p/src/aa/low.rs          compiled  include!("low.rs") in aa/mid.rs
p/src/zz/deepy.rs        compiled  `mod deepy;` in aa/low.rs resolves from zz/ (a chain whose second link sorts first)
p/src/n.rs               compiled  include!()d by the [[bin]] path of the nested package crates/p/fuzz
p/src/rp.rs              compiled  #[path] in src/lib.rs of the root package
p/src/pdir/pt.rs         compiled  #[path = "pt.rs"] inside `#[path = "pdir"] mod pin {` (#4700)
p/src/pdir/pn.rs         compiled  `mod pn;` inside `#[path = "pdir"] mod pin {` (#4700)
p/src/qt.rs              compiled  #[path = "qt.rs"] mod qt;
p/src/qt_cfg.rs          compiled  #[path] inside `#[path = "."] mod tests {`, the nn/quantization_tests.rs form
q/src/q_root.rs          compiled  [lib] path in Cargo.toml
q/src/qa.rs              compiled  `mod qa;` in q_root.rs
q/src/s.rs               compiled  #[path = "s.rs"] in q_root.rs
q/src/qgen.rs            compiled  a "*.rs" literal in the build = "gen/b.rs" script
p/src/h.rs               dark      `// mod h;` is a comment
p/src/sub/i.rs           dark      `mod i;` names src/i.rs only, and a commented-out #[path] names nothing
p/src/c/x/f.rs           dark      #[path = "x/f.rs"] in lib.rs names src/x/f.rs, not a same-named file elsewhere
p/src/k.rs               dark      no `mod k;`; it holds no test, and every src file is in the universe (#4700)
p/src/k/l.rs             dark      named only by the dark k.rs: reach is transitive (#4700)
p/src/m.rs               dark      no `mod m;`
p/src/w2.rs              dark      "unmod w2;" is no `mod w2;`: a match starts at a word boundary
p/src/lost/mod.rs        dark      a mod.rs that no `mod lost;` names
p/src/cyc_a.rs           dark      an include!() cycle between dark files ends, and both stay dark
p/src/cyc_b.rs           dark      the other half of that cycle
p/src/deep/mid/low/s.rs  dark      GAP: three levels of inline modules (see scripts/lib/src_reach.py)
p/src/mac.rs             dark      GAP: a macro-built `mod $n;` (see scripts/lib/src_reach.py)
v/src/lib.rs             dark      its Cargo.toml has no [package], so src/lib.rs is no root
v/src/generated.rs       dark      named only by the dark v/src/lib.rs
nm/src/x.rs              dark      its crate dir has no Cargo.toml
p/src/ignored.rs         unlisted  .gitignore names it, so it is not in the tree
p/src/gone.rs            unlisted  tracked, then deleted from disk: not listed, and no crash
'

# put ROOT PATH [LINE...] -- write ROOT/PATH, one LINE per line.
put() {
    local f="$1/$2"
    shift 2
    mkdir -p "${f%/*}" && printf '%s\n' "$@" > "$f"
}

# make_fixture ROOT -- the fixture tree of CASES, as a git repo.
make_fixture() {
    local r="$1" t='#[test] fn t() {}' f
    git -c init.defaultBranch=main init -q "$r" || return 1
    put "$r" .gitignore 'crates/p/src/ignored.rs'
    put "$r" Cargo.toml '[workspace]' 'members = ["crates/*"]' '' '[package]' 'name = "root"'
    put "$r" src/lib.rs '#[path = "../crates/p/src/rp.rs"] mod rp;'
    put "$r" crates/p/Cargo.toml '[package]' 'name = "p"'
    put "$r" crates/p/build.rs 'fn main() { let _ = "src/gen_root.rs"; }'
    put "$r" crates/p/src/lib.rs 'mod a;' 'mod c;' 'include!("e.rs");' '#[path = "x/f.rs"] mod f;' \
        'mod outer { mod g; }' '// mod h;' '// #[path = "sub/i.rs"] mod si;' 'mod i;' \
        '#[cfg(test)] mod tests;' 'mod r#type;' '/* mod j; */' 'mod deep { mod mid { mod low { mod s; } } }' \
        '#[path = "pdir"]' 'mod pin { #[path = "pt.rs"] mod pt; mod pn; }' '#[path = "qt.rs"] mod qt;' \
        'macro_rules! mk { ($n:ident) => { mod $n; } }' 'mk!(mac);' 'mod zz;' 'const S: &str = "unmod w2;";'
    put "$r" crates/p/src/a.rs 'pub(crate) mod b;'
    put "$r" crates/p/src/c/mod.rs 'mod d;' 'include!("../gen/part.rs");' 'mod inl { include!("inl_body.rs"); }'
    put "$r" crates/p/src/gen/part.rs 'include!("chain.rs");'
    put "$r" crates/p/src/gen/chain.rs 'mod y;'
    put "$r" crates/p/src/c/inl_body.rs 'mod v;'
    put "$r" crates/p/src/zz/mod.rs 'include!("../aa/mid.rs");'
    put "$r" crates/p/src/aa/mid.rs 'include!("low.rs");'
    put "$r" crates/p/src/aa/low.rs 'mod deepy;'
    put "$r" crates/p/src/qt.rs '#[cfg(test)]' '#[path = "."]' 'mod tests {' '    #[path = "qt_cfg.rs"]' \
        '    mod qt_cfg;' '}'
    put "$r" crates/p/src/k.rs 'mod l;'
    put "$r" crates/p/src/cyc_a.rs 'mod ma { include!("cyc_b.rs"); }'
    put "$r" crates/p/src/cyc_b.rs 'mod mb { include!("cyc_a.rs"); }'
    put "$r" crates/p/src/bin/y/main.rs '#[path = "../../z.rs"] mod z;'
    put "$r" crates/p/tests/it.rs '#[path = "../src/w.rs"] mod w;'
    put "$r" crates/p/fuzz/Cargo.toml '[package]' 'name = "p-fuzz"' '' '[[bin]]' 'name = "t"' \
        'path = "fuzz_targets/t.rs"'
    put "$r" crates/p/fuzz/fuzz_targets/t.rs 'include!("../../src/n.rs");'
    put "$r" crates/q/Cargo.toml '[package]' 'name = "q"' 'build = "gen/b.rs"' '' '[lib]' 'path = "src/q_root.rs"'
    put "$r" crates/q/gen/b.rs 'fn main() { let _ = "src/qgen.rs"; }'
    put "$r" crates/q/src/q_root.rs 'mod qa;' '#[path = "s.rs"] mod qs;'
    put "$r" crates/v/Cargo.toml '[workspace]' 'members = []'
    put "$r" crates/v/src/lib.rs 'mod generated;'
    for f in a/b c/d c/x/f c/y c/inl/v zz/deepy qt_cfg k/l e x/f outer/g h i sub/i tests type j \
        deep/mid/low/s gen_root m w2 bin/x z w rp n pdir/pt pdir/pn mac lost/mod ignored gone; do
        put "$r" "crates/p/src/$f.rs" "$t"
    done
    for f in q/src/qa q/src/s q/src/qgen v/src/generated nm/src/x; do
        put "$r" "crates/$f.rs"
    done
    git -C "$r" add -A && rm -f "${r:?}/crates/p/src/gone.rs"
}

# fail MSG -- one failed self-test row (counted in self_test's `fails`).
fail() {
    printf 'FAIL: %s\n' "$1"
    fails=$((fails + 1))
}

# check_cases ROOT DARK -- every row of CASES against DARK, the output for ROOT.
check_cases() {
    local r="$1" got="$2" path want why is rows on_disk
    while read -r path want why; do
        case "$path" in ''|'#'*) continue ;; esac
        is=absent
        grep -qxF "crates/$path" <<< "$got" && is=dark
        case "$want/$is" in
            dark/dark|compiled/absent|unlisted/absent) ;;
            *) fail "crates/$path is $is, want $want: $why" ;;
        esac
    done <<< "$CASES"
    rows=$(awk '$2 == "dark" { print "crates/" $1 }' <<< "$CASES" | sort)
    [ "$got" = "$rows" ] || fail "the output is not exactly the dark rows: <$(tr '\n' ' ' <<< "$got")>"
    # Every fixture src file on disk is a row, so none escapes the table.
    on_disk=$(cd "$r" && find crates -path 'crates/*/src/*' -name '*.rs' | sort)
    rows=$(awk '$2 ~ /^(compiled|dark|unlisted)$/ { print "crates/" $1 }' <<< "$CASES" |
        grep -vxF crates/p/src/gone.rs | sort)
    [ "$on_disk" = "$rows" ] || fail "fixture src files and CASES rows differ: <$(comm -3 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$rows") | tr '\n' ' ')>"
}

self_test() (
    # A git hook exports these for ITS repo; the fixture's git must not reach it.
    # Nor may a user's git config shape the fixture.
    unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX GIT_COMMON_DIR
    export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
    fails=0
    fx=$(mktemp -d) || exit 1
    case "$fx" in /tmp/?*|"${TMPDIR:-/tmp}"/?*) ;; *) printf 'refusing fixture dir <%s>\n' "$fx"; exit 1 ;; esac
    trap 'rm -rf "${fx:?}"' EXIT
    r="$fx/repo"
    make_fixture "$r" || { printf 'FAIL: could not build the fixture repo\n'; exit 1; }
    got=$(dark_in "$r") || { printf 'FAIL: src_reach.py gave no answer for the fixture\n'; exit 1; }
    check_cases "$r" "$got"

    # Mutations (rule 4): the verdict follows the tree, both ways.
    put "$r" crates/p/src/a.rs
    printf 'mod h;\n' >> "$r/crates/p/src/lib.rs"
    want=$(printf '%s\n' "$got" crates/p/src/a/b.rs | grep -vxF crates/p/src/h.rs | sort)
    now=$(dark_in "$r") || now='<no answer>'
    [ "$now" = "$want" ] || fail "emptying a.rs must darken a/b.rs, and \`mod h;\` must wire h.rs: <$(tr '\n' ' ' <<< "$now")>"

    # The baseline is an exact set: equal passes, a missing entry is a NEW dark
    # file, an extra entry is stale amnesty. Comments and blank lines are skipped.
    printf '# a comment, then a blank line\n\n%s\n' "$got" > "$fx/eq"
    sed 1d <<< "$got" > "$fx/short"
    printf '%s\ncrates/p/src/a.rs\n' "$got" > "$fx/extra"
    judge "$got" "$fx/eq" > /dev/null || fail 'judge: a baseline equal to the dark set must pass'
    out=$(judge "$got" "$fx/short") && fail 'judge: a dark file missing from the baseline must fail'
    grep -qxF "  ${got%%$'\n'*}" <<< "$out" || fail 'judge: the FAIL must name the new dark file'
    out=$(judge "$got" "$fx/extra") && fail 'judge: a stale baseline entry must fail'
    grep -qxF '  crates/p/src/a.rs' <<< "$out" || fail 'judge: the FAIL must name the stale entry'

    # No answer is never a pass: an empty tree (rc 3) and a dir git cannot list (rc 4).
    git -c init.defaultBranch=main init -q "$fx/empty"
    python3 "$REACH" "$fx/empty" > /dev/null 2>&1
    [ "$?" = 3 ] || fail 'src_reach.py: a tree with no crates/*/src file must be rc 3, not an answer'
    put "$fx/norepo" crates/p/src/lib.rs
    GIT_CEILING_DIRECTORIES="$fx" python3 "$REACH" "$fx/norepo" > /dev/null 2>&1
    [ "$?" = 4 ] || fail 'src_reach.py: a dir git cannot list must be rc 4, not an answer'
    # Against an EMPTY baseline, so that only the no-answer path can fail it.
    : > "$fx/none"
    check_at "$fx/empty" "$fx/none" > /dev/null 2>&1 && fail 'check_at: no answer from src_reach.py must FAIL'

    if [ "$fails" -ne 0 ]; then
        printf 'check_src_test_files_wired self-test: %d FAILED\n' "$fails"
        exit 1
    fi
    printf 'check_src_test_files_wired self-test: all cases pass\n'
)

main() {
    case "${1:-}" in
        --self-test) self_test; return ;;
        --list) dark_in "$REPO_ROOT"; return ;;
        "") ;;
        *) printf 'usage: %s [--list|--self-test]\n' "$0" >&2; return 2 ;;
    esac
    if ! self_test >/dev/null; then
        self_test
        return 1
    fi
    check_at "$REPO_ROOT" "$BASELINE"
}

main "$@"
