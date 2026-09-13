#!/usr/bin/env bash
# check_tree_reader_tests.sh — the test targets that READ THE TREE (BSE-17,
# PMAT-1077, PMAT-3120): derived from the sources, never hand-listed.
#
# WHY. On 2026-09-07 both of PR #3039's workspace-test reds came from Rust
# tests that read files outside any crate: aprender-contracts' baseline reader
# (a lib unit test reading scripts/contract_test_binding_baseline.txt) and
# aprender-core's readme_contract.rs (reading README.md). Neither PR touched
# those crates. A quick test tier selected by touched crate alone would have
# run nothing and passed. So the quick tier always runs these targets, and the
# set is DERIVED here from an oracle over the test sources and diffed against
# the committed registry on every run — a registry maintained by hand is the
# two-lists defect (bashrs#266's root cause).
#
# GRANULARITY (PMAT-3120, measured: issue #3120). Naming the CRATE when one
# test FILE in it reads the tree cost 88.08 % of ALL test-seconds — the 18
# whole `--lib` crates were the entire PR-tier bill, not the touched-crate
# expansion (~0.26 %). So a lib row now names the MODULE that holds the
# reader: `crate<TAB>--lib<TAB>module::path`. Mapping, from the file path:
#   src/a/b.rs      -> a::b        src/a/mod.rs -> a
#   src/lib.rs      -> <root>      (the crate root module)
#   an include!()-pulled file -> the module of the INCLUDING file (resolved by
#   scanning the crate for `include!("…/<file>")` and for `#[path = "…"] mod N;`)
# A file that no `mod <leaf>;` declares and that nothing includes is
# UNRESOLVABLE: the row falls back to the whole crate (`crate<TAB>--lib`) and a
# `WARN unresolved-include` line goes to stderr — a silent fallback would put
# the 88 % bill back without anyone noticing. Integration rows are unchanged
# (`crate<TAB>--test<TAB>name`), as are bin-only crates (`crate<TAB>--bins`).
#
# ORACLE. A test target reads the tree when its source names a path INTO the
# repository or resolves one from the manifest dir:
#   "scripts/  "docs/  "README  "contracts/  "../../  "../..
#   CARGO_MANIFEST_DIR  workspace_root(  project_root(
# Integration targets: crates/<c>/tests/<t>.rs -> `<c> --test <t>`.
# Lib targets: any crates/<c>/src/**/*.rs that contains `#[cfg(test)]` AND the
# oracle -> `<c> --lib <module>`. Fixture-only readers (tests/fixtures/...) are
# not excluded: an extra target costs seconds, a missing one costs a red main.
#
# Usage:
#   scripts/check_tree_reader_tests.sh            # derive, diff vs registry; exit 1 on drift
#   scripts/check_tree_reader_tests.sh --print    # print the derived set (wired half)
#   scripts/check_tree_reader_tests.sh --derive [ROOT]        # the raw derived set
#   scripts/check_tree_reader_tests.sh --check [ROOT [REG]]   # one registry only
#   scripts/check_tree_reader_tests.sh --update   # rewrite the registry
#   scripts/check_tree_reader_tests.sh --self-test
set -euo pipefail

ORACLE=${ORACLE:-'"scripts/|"docs/|"README|"contracts/|"\.\./\.\./|"\.\./\.\.|CARGO_MANIFEST_DIR|workspace_root\(|project_root\('}
REGISTRY_DEFAULT="scripts/tree_reader_tests.txt"
export ORACLE

# The two mutation switches the --self-test falsifier rows need. They are read
# from the environment ONLY under --self-test: a mutation that a CI run could
# turn on from outside is a hole, not a falsifier.
MUTATE_NO_INCLUDE=0
MUTATE_FLAT=0
if [ "${TREE_READER_SELF_TEST:-0}" = 1 ]; then
    MUTATE_NO_INCLUDE=${TREE_READER_MUTATE_NO_INCLUDE:-0}
    MUTATE_FLAT=${TREE_READER_MUTATE_FLAT:-0}
fi

INDEX_DIR=""

index_of() { # index_of <root> <crate> -> path of that crate's module index (built once)
    local root=$1 c=$2 src="$root/crates/$2/src" idx="$INDEX_DIR/$2.idx"
    [ -f "$idx" ] && { printf '%s\n' "$idx"; return 0; }
    : > "$idx"
    # ONE grep per crate: every `mod NAME;` declaration (with the DIRECTORY that
    # may own it), every include!() target and every #[path = "…"] attribute,
    # both resolved to a repo path. Per-reader greps over a crate the size of
    # aprender-serve cost minutes; this costs one pass.
    if [ -d "$src" ]; then
        grep -rn --include='*.rs' -E 'include!\(|#\[path[[:space:]]*=|mod[[:space:]]+[A-Za-z0-9_]+[[:space:]]*;' "$src" 2>/dev/null | awk -v OFS='\t' '
            function dirof(p,   d) { d = p; if (sub(/\/[^\/]*$/, "", d)) return d; return "." }
            function resolve(dir, rel,   parts, n, i, stack, m, out, abs) {
                abs = (substr(dir, 1, 1) == "/")
                n = split(dir "/" rel, parts, "/"); m = 0
                for (i = 1; i <= n; i++) {
                    if (parts[i] == "" || parts[i] == ".") continue
                    if (parts[i] == "..") { if (m > 0) m--; continue }
                    stack[++m] = parts[i]
                }
                out = ""
                for (i = 1; i <= m; i++) out = out (i > 1 ? "/" : "") stack[i]
                return (abs ? "/" : "") out
            }
            { i = index($0, ":"); file = substr($0, 1, i - 1); rest = substr($0, i + 1)
              j = index(rest, ":"); ln = substr(rest, 1, j - 1) + 0; txt = substr(rest, j + 1)
              sub(/^\.\//, "", file)
              # A COMMENT is not a declaration. The fixture crate proves this row
              # is load-bearing: its doc comments say "no `mod mystery;` declares
              # this file", and without this skip that prose WAS the declaration —
              # the first fixture run resolved both unresolvable files from prose.
              trimmed = txt; sub(/^[[:space:]]+/, "", trimmed)
              if (trimmed ~ /^(\/\/|\/\*|\*)/) next
              rel = ""
              if (match(txt, /"[^"]*\.rs"/)) rel = substr(txt, RSTART + 1, RLENGTH - 2)
              if (txt ~ /include!\(/ && rel != "") print "inc", resolve(dirof(file), rel), file
              if (txt ~ /#\[path[[:space:]]*=/ && rel != "") { pend[file] = resolve(dirof(file), rel); pline[file] = ln }
              if (match(txt, /mod[[:space:]]+[A-Za-z0-9_]+[[:space:]]*;/)) {
                  name = substr(txt, RSTART, RLENGTH); sub(/^mod[[:space:]]+/, "", name); sub(/[[:space:]]*;$/, "", name)
                  if (file in pend && ln - pline[file] <= 3) { print "path", pend[file], file, name; delete pend[file] }
                  else print "mod", name, file, dirof(file)
              } }' >> "$idx" || true
    fi
    printf '%s\n' "$idx"
}

module_of() { # module_of <root> <crate> <file> [depth] -> the module path; rc 1 = unresolvable
    local root=$1 c=$2 f=$3 depth=${4:-0} rel base leaf cand idx owner site name parent fn
    if [ "$depth" -gt 4 ]; then
        printf 'WARN unresolved-include %s: include!() chain deeper than 4 hops — falling back to the whole crate (%s --lib)\n' "$f" "$c" >&2
        return 1
    fi
    fn=${f#./}
    rel=${fn#*"/crates/$c/src/"}
    rel=${rel#"crates/$c/src/"}
    case "$rel" in lib.rs | main.rs) printf '<root>\n'; return 0 ;; esac
    base=${rel##*/}
    # The DIRECTORY whose files may declare this module: `mod b;` for src/a/b.rs
    # must live in src/a/ (src/a/mod.rs or a file include!()-spliced into it, a
    # pattern this repo uses everywhere); for src/a/mod.rs it must live in src/.
    # A crate-wide name match is too loose: `mod tests;` exists in dozens of
    # directories, and matching it resolved aprender-serve's src/cli/tests.rs to
    # `cli::tests` when the real path is `cli::cli_tests` (#[path]) — a filterset
    # atom matching ZERO tests, the silent-miss defect this ticket exists to kill.
    if [ "$base" = "mod.rs" ]; then
        cand=$(printf '%s' "${rel%/mod.rs}" | sed 's|/|::|g'); owner=${fn%/*}; owner=${owner%/*}
    else
        cand=$(printf '%s' "${rel%.rs}" | sed 's|/|::|g'); owner=${fn%/*}
    fi
    leaf=${cand##*::}
    if [ "$MUTATE_FLAT" = 1 ]; then cand=$leaf; fi
    idx=$(index_of "$root" "$c")
    if awk -F'\t' -v n="$leaf" -v d="$owner" '$1 == "mod" && $2 == n && $4 == d { found = 1 } END { exit !found }' "$idx"; then
        printf '%s\n' "$cand"; return 0
    fi
    if [ "$MUTATE_NO_INCLUDE" != 1 ]; then
        # `include!("…")` resolving to this file — its tests live in the INCLUDER's module.
        site=$(awk -F'\t' -v p="$fn" '$1 == "inc" && $2 == p { print $3; exit }' "$idx")
        if [ -n "$site" ] && [ "$site" != "$fn" ]; then module_of "$root" "$c" "$site" "$((depth + 1))"; return $?; fi
        # `#[path = "…"] mod NAME;` — a module of the DECLARING file, named NAME.
        site=$(awk -F'\t' -v p="$fn" '$1 == "path" && $2 == p { print $3; exit }' "$idx")
        name=$(awk -F'\t' -v p="$fn" '$1 == "path" && $2 == p { print $4; exit }' "$idx")
        if [ -n "$site" ] && [ -n "$name" ] && [ "$site" != "$fn" ]; then
            parent=$(module_of "$root" "$c" "$site" "$((depth + 1))") || return 1
            if [ "$parent" = "<root>" ]; then printf '%s\n' "$name"; else printf '%s::%s\n' "$parent" "$name"; fi
            return 0
        fi
    fi
    printf 'WARN unresolved-include %s: no `mod %s;` in %s/ declares it and no include!()/#[path] resolves to it — falling back to the whole crate (%s --lib)\n' "$f" "$leaf" "$owner" "$c" >&2
    return 1
}

derive() { # derive <repo root> -> sorted rows: crate\t--test\tname | crate\t--lib[\tmodule] | crate\t--bins
    local root=$1 f c t m
    INDEX_DIR=$(mktemp -d "${TMPDIR:-/tmp}/tree-reader-idx.XXXXXX")
    trap 'rm -rf "${INDEX_DIR:?}"' RETURN
    (
        for f in "$root"/crates/*/tests/*.rs; do
            [ -f "$f" ] || continue
            grep -qE "$ORACLE" "$f" || continue
            c=$(basename "$(dirname "$(dirname "$f")")"); t=$(basename "$f" .rs)
            printf '%s\t--test\t%s\n' "$c" "$t"
        done
        for f in $(find "$root"/crates/*/src -name '*.rs' 2>/dev/null); do
            grep -q '#\[cfg(test)\]' "$f" || continue
            grep -qE "$ORACLE" "$f" || continue
            c=$(printf '%s' "$f" | sed "s|^$root/crates/||; s|/.*||")
            # `--lib` on a crate with NO library target is a hard error, never a
            # passable gate: `error: no library targets found in package X`.
            # aprender-compute-xtask is bin-only and its src carries a cfg(test)
            # reader, so the first quick-tier trial died on it (2026-09-07).
            # cohete shipped this exact shape as a pre-push hook and every push
            # then used --no-verify. The target follows the crate.
            if [ ! -f "$root/crates/$c/src/lib.rs" ]; then
                printf '%s\t--bins\n' "$c"
            elif m=$(module_of "$root" "$c" "$f"); then
                printf '%s\t--lib\t%s\n' "$c" "$m"
            else
                printf '%s\t--lib\n' "$c"
            fi
        done
    ) | LC_ALL=C sort -u
}


# WIRED vs UNWIRED (BSE-17, measured 2026-09-07).
#
# A tree-reader target belongs in the QUICK TIER only if CI runs it somewhere.
# The first run of this tier ran `apr-cli --test pixel_regression`, which NO
# workflow runs and which fails on main (4 of 6 pixel tests red, golden
# snapshots that no lane has compared in months). A quick tier stricter than
# the full tier is a gate that cannot pass, and this repository has one of
# those on the record already (cohete's pre-push).
#
# So the derived set is SPLIT by an oracle, never a hand list:
#   * `--lib`  — wired: the full tier runs `cargo nextest run --workspace --lib`
#                and the excluded crates have their own named steps.
#   * `--test NAME` — wired iff some file under .github/workflows/ names
#                `--test NAME`.
# Unwired targets go to the ledger below and are NOT in the quick tier. The
# ledger is shrink-only in the sense that matters: a new unwired reader FAILS
# (write the test into a lane, or ledger it deliberately), and a ledger line
# that has since been wired FAILS too (delete it). "Reads the tree and nothing
# runs it" is itself a finding — 36 of 80 targets on the day this was written.
UNWIRED_LEDGER_DEFAULT="scripts/tree_reader_unwired_baseline.txt"

# full_tier_excludes ROOT -- the crates the full tier's `--workspace --lib` run
# EXCLUDES (gpu, cuda-edge, compute: they need hardware or their own step).
# Read from the workflow, never restated: a lib target of an excluded crate is
# not wired by `--workspace --lib`, and the first quick-tier run would have
# tested aprender-gpu --lib on a CPU runner had this not been derived.
full_tier_excludes() { # full_tier_excludes <root> -> one crate per line
    grep -rhoE 'nextest run --profile ci --workspace --lib( --exclude [a-z0-9-]+)+' "$1"/.github/workflows/ 2>/dev/null \
        | head -1 | grep -oE -- '--exclude [a-z0-9-]+' | awk '{print $2}'
}

wired_targets() { # wired_targets <root> -- the derived set, wired half only
    local root=$1 ex c kind name
    # `|| true`: the excludes grep legitimately finds nothing (a tree with no
    # workflows), and under errexit that killed --print with rc=1 and no output.
    ex=$(full_tier_excludes "$root" || true)
    derive "$root" | while IFS=$'\t' read -r c kind name; do
        if [ "$kind" = "--lib" ] || [ "$kind" = "--bins" ]; then
            # excluded from --workspace --lib by the full tier (hardware, own step)?
            # A here-string, never `printf | grep -q`: under pipefail, grep -q exiting on
            # its first match leaves printf writing into a closed pipe (EPIPE, "write
            # error: Broken pipe", run 34634920736 line 227) and the pipeline FAILS on
            # the producer's status although grep MATCHED -- so the excluded crate was
            # NOT skipped and `aprender-gpu --lib driver::memory::transfer` was derived
            # on CI and not locally (a race, box-dependent). The SIGPIPE+pipefail class.
            grep -qxF -- "$c" <<< "$ex" && continue
            if [ -n "$name" ]; then printf '%s\t%s\t%s\n' "$c" "$kind" "$name"
            else printf '%s\t%s\n' "$c" "$kind"; fi
        elif grep -rqF -- "--test $name" "$root"/.github/workflows/ 2>/dev/null; then
            printf '%s\t--test\t%s\n' "$c" "$name"
        fi
    done
}

unwired_targets() { # unwired_targets <root> -- reads the tree, no lane runs it
    local root=$1 c kind name
    derive "$root" | while IFS=$'\t' read -r c kind name; do
        [ "$kind" = "--test" ] || continue
        grep -rqF -- "--test $name" "$root"/.github/workflows/ 2>/dev/null || printf '%s\t--test\t%s\n' "$c" "$name"
    done
}

check_unwired() { # check_unwired <root> <ledger> -> 0 same, 1 drift, 2 env
    local root=$1 ledger=$2 want have
    [ -f "$ledger" ] || { printf 'ENV   %s missing — run --update to derive it\n' "$ledger"; return 2; }
    want=$(unwired_targets "$root"); have=$(registry_body "$ledger")
    if [ "$want" != "$have" ]; then
        printf 'FAIL  %s drifted (<: ledgered but now wired — delete the line, >: new unwired tree-reader — wire it into a lane or ledger it):\n' "$ledger"
        diff <(printf '%s\n' "$have") <(printf '%s\n' "$want") | grep '^[<>]' | sed 's/^/        /' | head -30
        return 1
    fi
    printf 'PASS  %s: %s tree-reader target(s) that no workflow runs\n' "$ledger" "$(printf '%s\n' "$want" | grep -c .)"
}

# `|| true` on both greps: a comments-only registry makes `grep -v` exit 1, and
# under `set -o pipefail` that aborted check() BEFORE it could print anything —
# rc=1 with an empty message, which reads as a crash, not a verdict. (Latent
# until PMAT-3120: the old self-test called check() from a `bash -c` that had no
# pipefail, so the vacuity row passed for the wrong reason.)
registry_body() { { grep -v '^#' "$1" || true; } | { grep -v '^[[:space:]]*$' || true; } | LC_ALL=C sort -u; }

check() { # check <root> <registry> -> 0 same, 1 drift, 2 env
    local root=$1 reg=$2 want have
    [ -f "$reg" ] || { printf 'ENV   %s missing — run --update to derive it; the quick tier refuses to run without it\n' "$reg"; return 2; }
    want=$(wired_targets "$root"); have=$(registry_body "$reg")
    [ -n "$want" ] || { printf 'FAIL  the oracle derived ZERO tree-reader targets under %s — a detector that finds nothing over a real tree is broken, not a pass\n' "$root"; return 1; }
    if [ "$want" != "$have" ]; then
        printf 'FAIL  %s drifted from the sources (<: registry only, >: derived only):\n' "$reg"
        diff <(printf '%s\n' "$have") <(printf '%s\n' "$want") | grep '^[<>]' | sed 's/^/        /' | head -40
        printf '        run: bash scripts/check_tree_reader_tests.sh --update\n'
        return 1
    fi
    printf 'PASS  %s: %s tree-reader target(s), registry equals the derived set\n' "$reg" "$(printf '%s\n' "$want" | wc -l | tr -d ' ')"
}

update() { # update <root> <registry>
    local root=$1 reg=$2
    { printf '# tool_version=none (derived by scripts/check_tree_reader_tests.sh from the test sources; regenerate with --update, never edit)\n'
      printf '# Every test target that reads the tree (BSE-17); the quick tier always runs these.\n'
      printf '# COLUMNS (PMAT-3120): crate<TAB>--lib<TAB>module::path — the module that holds\n'
      printf '#   the reader (src/a/b.rs -> a::b, src/a/mod.rs -> a, src/lib.rs -> <root>, an\n'
      printf '#   include!()-pulled file -> the INCLUDING file'"'"'s module). A 2-column\n'
      printf '#   crate<TAB>--lib is the WHOLE crate: the module was unresolvable and the\n'
      printf '#   derivation said so on stderr (WARN unresolved-include).\n'
      printf '# Also: crate<TAB>--test<TAB>name (integration binary) and crate<TAB>--bins.\n'
      wired_targets "$root"; } > "$reg"
    printf 'ok    wrote %s (%s target(s))\n' "$reg" "$(registry_body "$reg" | wc -l | tr -d ' ')"
}

self_test() {
    local td n=0 red=0 out rc T FX
    export TREE_READER_SELF_TEST=1
    T=$0
    FX="tests/fixtures/tree_reader"
    td=$(mktemp -d "${TMPDIR:-/tmp}/tree-readers.XXXXXX")
    trap 'rm -rf "${td:?}"' RETURN
    mkdir -p "$td/crates/alpha/tests" "$td/crates/alpha/src" "$td/crates/beta/src/lint" "$td/crates/gamma/src" "$td/crates/gamma/tests"
    printf 'use std::path::Path;\n#[test] fn t() { let _ = std::fs::read_to_string(Path::new("README.md")); }\n' > "$td/crates/alpha/tests/reads_readme.rs"
    printf '#[test] fn t() { assert_eq!(1, 1); }\n' > "$td/crates/alpha/tests/pure.rs"
    printf 'pub fn f() {}\n' > "$td/crates/alpha/src/lib.rs"
    printf 'pub fn parse() {}\n#[cfg(test)]\nmod tests { #[test] fn t() { let _ = std::fs::read_to_string("scripts/x_baseline.txt"); } }\n' > "$td/crates/beta/src/lint/mod.rs"
    printf 'pub mod lint;\n' > "$td/crates/beta/src/lib.rs"
    printf 'pub fn load() { let _ = std::fs::read_to_string("scripts/config.txt"); }\n' > "$td/crates/gamma/src/lib.rs"
    printf '#[test] fn t() { let _ = env!("CARGO_MANIFEST_DIR"); }\n' > "$td/crates/gamma/tests/manifest_dir.rs"
    row() { # row WANT_RC LABEL MUST_MATCH -- CMD...
        local want=$1 label=$2 pat=$3; shift 3; n=$((n + 1))
        rc=0; out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && grep -qE -- "$pat" <<< "$out"; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"; printf '%s\n' "$out" | sed 's/^/        /'; red=1; fi
    }
    row 0 "derive: alpha reads_readme (README path), beta --lib lint (cfg(test) + scripts/ path), gamma manifest_dir (CARGO_MANIFEST_DIR); NOT alpha pure, NOT gamma --lib (reader without cfg(test))" \
        '^alpha	--test	reads_readme$' bash "$T" --derive "$td"
    out=$(bash "$T" --derive "$td" 2>/dev/null) || true
    n=$((n + 1))
    if grep -q '^beta	--lib	lint$' <<< "$out" && grep -q '^gamma	--test	manifest_dir$' <<< "$out" && ! grep -q 'pure' <<< "$out" && ! grep -q '^gamma	--lib' <<< "$out"; then
        printf 'ok    row %-2s        derived set is exactly {alpha reads_readme, beta --lib lint, gamma manifest_dir} — the beta row names the MODULE (src/lint/mod.rs -> lint), not the crate\n' "$n"
    else
        printf 'FAIL  row %-2s        derived set wrong:\n%s\n' "$n" "$out"; red=1
    fi
    # delta: a crate with a cfg(test) reader and NO src/lib.rs is bin-only, so
    # the target is --bins. `--lib` there is `error: no library targets found`
    # — a gate that cannot pass, which is how cohete's pre-push trained
    # --no-verify on every push.
    mkdir -p "$td/crates/delta/src"
    printf 'fn main() {}\n#[cfg(test)]\nmod tests { #[test] fn t() { let _ = std::fs::read_to_string("scripts/d.txt"); } }\n' > "$td/crates/delta/src/main.rs"
    n=$((n + 1))
    if grep -q '^delta	--bins$' <<< "$(bash "$T" --derive "$td" 2>/dev/null)" && ! grep -q '^delta	--lib' <<< "$(bash "$T" --derive "$td" 2>/dev/null)"; then
        printf 'ok    row %-2s        a bin-only crate (cfg(test) reader, no src/lib.rs) is --bins, never --lib\n' "$n"
    else
        printf 'FAIL  row %-2s        bin-only crate mis-targeted: %s\n' "$n" "$(bash "$T" --derive "$td" 2>/dev/null | grep '^delta' | tr '\n' ';')"; red=1
    fi
    # A workflow that names one target, so the wired/unwired split is exercised
    # rather than assumed: alpha reads_readme is run by a lane, gamma manifest_dir
    # is not, beta --lib is covered by the full tier's --workspace --lib.
    mkdir -p "$td/.github/workflows"
    printf 'jobs:\n  t:\n    steps:\n      - run: cargo test -p alpha --test reads_readme\n' > "$td/.github/workflows/ci.yml"
    n=$((n + 1))
    w=$(bash "$T" --print "$td" 2>/dev/null); u=$(bash "$T" --print-unwired "$td" 2>/dev/null)
    if grep -q '^alpha' <<< "$w" && grep -q '^beta	--lib	lint$' <<< "$w" && grep -q '^gamma' <<< "$u" && ! grep -q '^alpha' <<< "$u"; then
        printf 'ok    row %-2s        WIRED/UNWIRED split: a lane names alpha reads_readme (quick tier), gamma manifest_dir is run by nothing (ledger), beta --lib lint is covered by --workspace --lib\n' "$n"
    else
        printf 'FAIL  row %-2s        split wrong. wired=[%s] unwired=[%s]\n' "$n" "$(printf '%s' "$w" | tr '\n' ';')" "$(printf '%s' "$u" | tr '\n' ';')"; red=1
    fi
    update "$td" "$td/registry.txt" > /dev/null 2>&1
    row 0 "registry equals derived -> PASS" '^PASS' bash "$T" --check "$td" "$td/registry.txt"
    printf 'zeta\t--lib\n' >> "$td/registry.txt"
    row 1 "a stale registry line -> RED (drift, registry only)" '^FAIL .*drifted' bash "$T" --check "$td" "$td/registry.txt"
    update "$td" "$td/registry.txt" > /dev/null 2>&1; sed -i '/^beta/d' "$td/registry.txt"
    row 1 "a missing registry line -> RED (drift, derived only)" '> beta' bash "$T" --check "$td" "$td/registry.txt"
    row 2 "registry file absent -> ENV (exit 2), never a pass" '^ENV' bash "$T" --check "$td" "$td/absent.txt"
    mkdir -p "$td/empty/crates/x/tests"; printf '#[test] fn t() {}\n' > "$td/empty/crates/x/tests/t.rs"; printf '# h\n' > "$td/empty/reg.txt"
    row 1 "a tree with ZERO readers -> RED (vacuity), never a pass" 'derived ZERO' bash "$T" --check "$td/empty" "$td/empty/reg.txt"
    # MUTANT: drop the scripts/ pattern from the oracle -> beta vanishes from the derived set (the falsifier discriminates)
    row 0 "mutant oracle without the scripts/ pattern loses beta --lib lint — this row proves the oracle is load-bearing" 'MUTANT-LOST-BETA' \
        env ORACLE="${ORACLE/\"scripts\/|/}" bash -c "if bash '$T' --derive '$td' 2>/dev/null | grep -q '^beta'; then echo MUTANT-KEPT-BETA; else echo MUTANT-LOST-BETA; fi"

    # --- PMAT-3120: module granularity, against the COMMITTED fixture crate.
    # Hermetic (no cargo, no workspace): tests/fixtures/tree_reader/crates/** is
    # a tree of .rs files, and the golden is the derived registry text.
    bash "$T" --derive "$FX" > "$td/fx.out" 2> "$td/fx.warn" || true
    row 0 "fixture: the derived rows equal the committed golden (root / nested mod.rs / leaf.rs / include!()-pulled / unresolvable / integration)" \
        '^$' diff "$FX/derived.golden.txt" "$td/fx.out"
    row 0 "  ...src/lib.rs -> <root>" '^reader_mods	--lib	<root>$' cat "$td/fx.out"
    row 0 "  ...src/deep/mod.rs -> deep" '^reader_mods	--lib	deep$' cat "$td/fx.out"
    row 0 "  ...src/deep/leaf.rs -> deep::leaf" '^reader_mods	--lib	deep::leaf$' cat "$td/fx.out"
    row 0 "  ...src/gen/part.rs, pulled by include!() from src/inc.rs -> inc (the INCLUDER's module)" '^reader_mods	--lib	inc$' cat "$td/fx.out"
    row 0 "  ...src/attached.rs, declared #[path] as mod bolted from src/deep/mod.rs -> deep::bolted" '^reader_mods	--lib	deep::bolted$' cat "$td/fx.out"
    row 0 "  ...tests/it.rs -> --test it (integration rows unchanged)" '^reader_mods	--test	it$' cat "$td/fx.out"
    row 0 "  ...an unresolvable reader -> the WHOLE crate, 2 columns (fallback, never a guessed module)" '^reader_orphan	--lib$' cat "$td/fx.out"
    row 0 "  ...and that fallback is VISIBLE on stderr, never silent" 'WARN unresolved-include .*mystery\.rs' cat "$td/fx.warn"
    # MUTATION 1: hide the include-resolver -> the include!()-pulled reader loses
    # its module and the crate falls back whole. The golden CHANGES.
    row 1 "MUTATION: include-resolver hidden (TREE_READER_MUTATE_NO_INCLUDE=1) -> the golden DIFFERS" '^[<>]' \
        bash -c "TREE_READER_MUTATE_NO_INCLUDE=1 bash '$T' --derive '$FX' 2>/dev/null | diff '$FX/derived.golden.txt' -"
    row 0 "  ...and it equals the committed no-include golden: inc/deep::bolted gone, reader_mods --lib (whole crate) instead" '^$' \
        bash -c "TREE_READER_MUTATE_NO_INCLUDE=1 bash '$T' --derive '$FX' 2>/dev/null | diff '$FX/derived.no-include.golden.txt' -"
    # MUTATION 2: flatten the module path (leaf only) -> deep::leaf becomes leaf.
    row 1 "MUTATION: module path flattened to the leaf (TREE_READER_MUTATE_FLAT=1) -> the golden DIFFERS" '^[<>]' \
        bash -c "TREE_READER_MUTATE_FLAT=1 bash '$T' --derive '$FX' 2>/dev/null | diff '$FX/derived.golden.txt' -"
    row 0 "  ...and it equals the committed flat golden (deep::leaf -> leaf)" '^$' \
        bash -c "TREE_READER_MUTATE_FLAT=1 bash '$T' --derive '$FX' 2>/dev/null | diff '$FX/derived.flat.golden.txt' -"
    # The mutations are self-test-only: without TREE_READER_SELF_TEST the switch is inert.
    row 1 "the mutation switches are inert outside --self-test (TREE_READER_SELF_TEST unset -> the real golden)" '^[<>]' \
        env -u TREE_READER_SELF_TEST TREE_READER_MUTATE_FLAT=1 bash -c "bash '$T' --derive '$FX' 2>/dev/null | diff '$FX/derived.flat.golden.txt' -"
    # The real tree: every lib row carries a module, or its 2-column fallback was warned about.
    n=$((n + 1))
    bash "$T" --derive . > "$td/real.out" 2> "$td/real.warn" || true
    local bare warned
    bare=$(awk -F'\t' '$2 == "--lib" && NF == 2 { print $1 }' "$td/real.out" | sort -u | wc -l | tr -d ' ')
    warned=$(grep -c 'WARN unresolved-include' "$td/real.warn" || true)
    if [ "$bare" = 0 ] || [ "$warned" -gt 0 ]; then
        printf 'ok    row %-2s        the real tree: %s whole-crate lib fallback(s), %s WARN unresolved-include line(s) — a fallback without a WARN is the silent 88 %% bill\n' "$n" "$bare" "$warned"
    else
        printf 'FAIL  row %-2s        %s whole-crate lib fallback(s) with ZERO WARN lines\n' "$n" "$bare"; red=1
    fi
    printf '\n%s checks, %s failed\n' "$n" "$red"
    [ "$red" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    --derive) derive "${2:-.}" ;;
    --print) wired_targets "${2:-.}" ;;
    --print-unwired) unwired_targets "${2:-.}" ;;
    --check) check "${2:-.}" "${3:-$REGISTRY_DEFAULT}" ;;
    --check-unwired) check_unwired "${2:-.}" "${3:-$UNWIRED_LEDGER_DEFAULT}" ;;
    --update) update "${2:-.}" "${3:-$REGISTRY_DEFAULT}"; { printf '# tool_version=none (derived by scripts/check_tree_reader_tests.sh --print-unwired; regenerate with --update, never edit)\n'; printf '# Test targets that READ THE TREE and that no workflow runs. Not in the quick\n# tier (it must never be stricter than the full tier), and each line is a\n# finding: a test nothing executes (BSE-17, PMAT-1077).\n'; unwired_targets "${2:-.}"; } > "${4:-$UNWIRED_LEDGER_DEFAULT}"; printf 'ok    wrote %s (%s unwired target(s))\n' "${4:-$UNWIRED_LEDGER_DEFAULT}" "$(unwired_targets "${2:-.}" | grep -c .)" ;;
    "") check . "$REGISTRY_DEFAULT" && check_unwired . "$UNWIRED_LEDGER_DEFAULT" ;;
    *) printf 'usage: %s [--print [ROOT] | --derive [ROOT] | --check [ROOT [REG]] | --update [ROOT [REGISTRY]] | --self-test]\n' "$0" >&2; exit 2 ;;
esac
