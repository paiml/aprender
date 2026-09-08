#!/usr/bin/env bash
# check_tree_reader_tests.sh — the test targets that READ THE TREE (BSE-17,
# PMAT-1077): derived from the sources, never hand-listed.
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
# ORACLE. A test target reads the tree when its source names a path INTO the
# repository or resolves one from the manifest dir:
#   "scripts/  "docs/  "README  "contracts/  "../../  "../..
#   CARGO_MANIFEST_DIR  workspace_root(  project_root(
# Integration targets: crates/<c>/tests/<t>.rs -> `<c> --test <t>`.
# Lib targets: any crates/<c>/src/**/*.rs that contains `#[cfg(test)]` AND the
# oracle -> `<c> --lib`. Fixture-only readers (tests/fixtures/...) are not
# excluded: an extra target costs seconds, a missing one costs a red main.
#
# Usage:
#   scripts/check_tree_reader_tests.sh            # derive, diff vs registry; exit 1 on drift
#   scripts/check_tree_reader_tests.sh --print    # print the derived set
#   scripts/check_tree_reader_tests.sh --update   # rewrite the registry
#   scripts/check_tree_reader_tests.sh --self-test
set -euo pipefail

ORACLE='"scripts/|"docs/|"README|"contracts/|"\.\./\.\./|"\.\./\.\.|CARGO_MANIFEST_DIR|workspace_root\(|project_root\('
REGISTRY_DEFAULT="scripts/tree_reader_tests.txt"
export ORACLE

derive() { # derive <repo root> -> sorted "crate\t--test\tname" / "crate\t--lib" lines
    local root=$1 f c t
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
            if [ -f "$root/crates/$c/src/lib.rs" ]; then
                printf '%s\t--lib\n' "$c"
            else
                printf '%s\t--bins\n' "$c"
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
    local root=$1 line c kind name
    derive "$root" | while IFS=$'\t' read -r c kind name; do
        if [ "$kind" = "--lib" ] || [ "$kind" = "--bins" ]; then
            # excluded from --workspace --lib by the full tier (hardware, own step)?
            full_tier_excludes "$root" | grep -qxF "$c" && continue
            printf '%s\t%s\n' "$c" "$kind"
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

registry_body() { grep -v '^#' "$1" | grep -v '^[[:space:]]*$' | LC_ALL=C sort -u; }

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
      printf '# crate<TAB>--test<TAB>name | crate<TAB>--lib — every test target that reads the tree (BSE-17); the quick tier always runs these\n'
      wired_targets "$root"; } > "$reg"
    printf 'ok    wrote %s (%s target(s))\n' "$reg" "$(registry_body "$reg" | wc -l | tr -d ' ')"
}

self_test() {
    local td n=0 red=0 out rc
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
        if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -qE -- "$pat"; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"; printf '%s\n' "$out" | sed 's/^/        /'; red=1; fi
    }
    row 0 "derive: alpha reads_readme (README path), beta --lib (cfg(test) + scripts/ path), gamma manifest_dir (CARGO_MANIFEST_DIR); NOT alpha pure, NOT gamma --lib (reader without cfg(test))" \
        '^alpha	--test	reads_readme$' bash -c "$(declare -f derive); derive '$td'"
    out=$(derive "$td") || true; printf '%s\n' "$out" | grep -q '^beta	--lib$' && printf '%s\n' "$out" | grep -q '^gamma	--test	manifest_dir$' && ! printf '%s\n' "$out" | grep -q 'pure' && ! printf '%s\n' "$out" | grep -q '^gamma	--lib$' && printf 'ok    row %-2s        derived set is exactly {alpha reads_readme, beta --lib, gamma manifest_dir}\n' "$((n + 1))" || { printf 'FAIL  row %-2s        derived set wrong:\n%s\n' "$((n + 1))" "$out"; red=1; }; n=$((n + 1))
    # delta: a crate with a cfg(test) reader and NO src/lib.rs is bin-only, so
    # the target is --bins. `--lib` there is `error: no library targets found`
    # — a gate that cannot pass, which is how cohete's pre-push trained
    # --no-verify on every push.
    mkdir -p "$td/crates/delta/src"
    printf 'fn main() {}\n#[cfg(test)]\nmod tests { #[test] fn t() { let _ = std::fs::read_to_string("scripts/d.txt"); } }\n' > "$td/crates/delta/src/main.rs"
    n=$((n + 1))
    if derive "$td" | grep -q '^delta	--bins$' && ! derive "$td" | grep -q '^delta	--lib$'; then
        printf 'ok    row %-2s        a bin-only crate (cfg(test) reader, no src/lib.rs) is --bins, never --lib\n' "$n"
    else
        printf 'FAIL  row %-2s        bin-only crate mis-targeted: %s\n' "$n" "$(derive "$td" | grep '^delta' | tr '\n' ';')"; red=1
    fi
    # A workflow that names one target, so the wired/unwired split is exercised
    # rather than assumed: alpha reads_readme is run by a lane, gamma manifest_dir
    # is not, beta --lib is covered by the full tier's --workspace --lib.
    mkdir -p "$td/.github/workflows"
    printf 'jobs:\n  t:\n    steps:\n      - run: cargo test -p alpha --test reads_readme\n' > "$td/.github/workflows/ci.yml"
    n=$((n + 1))
    w=$(wired_targets "$td"); u=$(unwired_targets "$td")
    if printf '%s\n' "$w" | grep -q '^alpha	test' 2>/dev/null || printf '%s\n' "$w" | grep -q '^alpha' && printf '%s\n' "$w" | grep -q '^beta	--lib$' && printf '%s\n' "$u" | grep -q '^gamma' && ! printf '%s\n' "$u" | grep -q '^alpha'; then
        printf 'ok    row %-2s        WIRED/UNWIRED split: a lane names alpha reads_readme (quick tier), gamma manifest_dir is run by nothing (ledger), beta --lib is covered by --workspace --lib\n' "$n"
    else
        printf 'FAIL  row %-2s        split wrong. wired=[%s] unwired=[%s]\n' "$n" "$(printf '%s' "$w" | tr '\n' ';')" "$(printf '%s' "$u" | tr '\n' ';')"; red=1
    fi
    update "$td" "$td/registry.txt" > /dev/null
    row 0 "registry equals derived -> PASS" '^PASS' bash -c "$(declare -f derive full_tier_excludes wired_targets unwired_targets registry_body check check_unwired); check '$td' '$td/registry.txt'"
    printf 'zeta\t--lib\n' >> "$td/registry.txt"
    row 1 "a stale registry line -> RED (drift, registry only)" '^FAIL .*drifted' bash -c "$(declare -f derive full_tier_excludes wired_targets unwired_targets registry_body check check_unwired); check '$td' '$td/registry.txt'"
    update "$td" "$td/registry.txt" > /dev/null; sed -i '/^beta/d' "$td/registry.txt"
    row 1 "a missing registry line -> RED (drift, derived only)" '> beta' bash -c "$(declare -f derive full_tier_excludes wired_targets unwired_targets registry_body check check_unwired); check '$td' '$td/registry.txt'"
    row 2 "registry file absent -> ENV (exit 2), never a pass" '^ENV' bash -c "$(declare -f derive full_tier_excludes wired_targets unwired_targets registry_body check check_unwired); check '$td' '$td/absent.txt'"
    mkdir -p "$td/empty/crates/x/tests"; printf '#[test] fn t() {}\n' > "$td/empty/crates/x/tests/t.rs"; printf '# h\n' > "$td/empty/reg.txt"
    row 1 "a tree with ZERO readers -> RED (vacuity), never a pass" 'derived ZERO' bash -c "$(declare -f derive full_tier_excludes wired_targets unwired_targets registry_body check check_unwired); check '$td/empty' '$td/empty/reg.txt'"
    # MUTANT: drop the scripts/ pattern from the oracle -> beta --lib vanishes from the derived set (the falsifier discriminates)
    row 0 "mutant oracle without the scripts/ pattern loses beta --lib — this row proves the oracle is load-bearing" 'MUTANT-LOST-BETA' bash -c "ORACLE=$(printf %q "${ORACLE/\"scripts\/|/}"); $(declare -f derive); if derive '$td' | grep -q '^beta'; then echo MUTANT-KEPT-BETA; else echo MUTANT-LOST-BETA; fi"
    printf '\n%s checks, %s failed\n' "$n" "$red"
    [ "$red" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    --print) wired_targets "${2:-.}" ;;
    --print-unwired) unwired_targets "${2:-.}" ;;
    --update) update "${2:-.}" "${3:-$REGISTRY_DEFAULT}"; { printf '# tool_version=none (derived by scripts/check_tree_reader_tests.sh --print-unwired; regenerate with --update, never edit)\n'; printf '# Test targets that READ THE TREE and that no workflow runs. Not in the quick\n# tier (it must never be stricter than the full tier), and each line is a\n# finding: a test nothing executes (BSE-17, PMAT-1077).\n'; unwired_targets "${2:-.}"; } > "${4:-$UNWIRED_LEDGER_DEFAULT}"; printf 'ok    wrote %s (%s unwired target(s))\n' "${4:-$UNWIRED_LEDGER_DEFAULT}" "$(unwired_targets "${2:-.}" | grep -c .)" ;;
    "") check . "$REGISTRY_DEFAULT" && check_unwired . "$UNWIRED_LEDGER_DEFAULT" ;;
    *) printf 'usage: %s [--print [ROOT] | --update [ROOT [REGISTRY]] | --self-test]\n' "$0" >&2; exit 2 ;;
esac
