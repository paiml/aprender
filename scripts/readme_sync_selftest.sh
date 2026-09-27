#!/usr/bin/env bash
# readme_sync_selftest.sh -- the case table for scripts/readme_sync.sh, the ONE writer of
# every derived README count (GEN-001, aprender#4526). check_readme_claims.sh runs it on
# its normal path (no workflow runs a --self-test, and a new check_*.sh would need one).
#
# Each row plants a throwaway tree (README_SYNC_ROOT) and README (README_PATH), runs the
# generator, and asserts an exit code and an effect. Rows that would read GREEN on a
# broken generator carry an effect check, not only an exit code.
#
#   bash scripts/readme_sync_selftest.sh [SUBJECT]      SUBJECT defaults to scripts/readme_sync.sh
# Exit: 0 every row green · 1 any row red · 2 the fixture could not be built.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBJ="${1:-$REPO_ROOT/scripts/readme_sync.sh}"

TD=$(mktemp -d "${TMPDIR:-/tmp}/check-readme-sync.XXXXXX") || exit 2
_rm_td() { case "${TD:-}" in /tmp/?*|"${TMPDIR:-/tmp}"/?*) rm -rf -- "${TD:?}" ;; esac; }
trap _rm_td EXIT

n=0 red=0
row() { # row <label> <test-exit>
    n=$((n + 1))
    if [ "$2" = 0 ]; then printf 'ok    row %-2s %s\n' "$n" "$1"
    else printf 'FAIL  row %-2s %s\n        output: %s\n' "$n" "$1" "$(tr '\n' '|' < "$TD/out" | cut -c1-300)"; red=1; fi
}
gen() { # gen <args...> -> RC, output in $TD/out
    README_PATH="$TD/README.md" README_SYNC_ROOT="$TD/t" bash "$SUBJ" "$@" > "$TD/out" 2>&1; RC=$?
}

# The tree: 2 workspace crates, 3 contracts in the census, 2 CLI commands (+ `help`,
# which is not one), 3 book cli chapters, 1 book lib chapter, 3 crates/ dirs.
plant() {
    rm -rf -- "${TD:?}/t"
    mkdir -p "$TD/t/contracts" "$TD/t/book/src/cli" "$TD/t/book/src/lib" "$TD/t/crates/a/src" "$TD/t/crates/b/src" "$TD/t/crates/stray" || return 2
    printf '[workspace]\nmembers = ["crates/a", "crates/b"]\nresolver = "2"\n' > "$TD/t/Cargo.toml"
    for c in a b; do
        printf '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n' "$c" > "$TD/t/crates/$c/Cargo.toml"
        : > "$TD/t/crates/$c/src/lib.rs"
    done
    printf '{"n_files": 3}\n' > "$TD/t/contracts/census.json"
    printf 'commands:\n  - name: run\n  - name: serve\n  - name: help\n' > "$TD/t/contracts/apr-cli-commands-v1.yaml"
    : > "$TD/t/book/src/cli/run.md"; : > "$TD/t/book/src/cli/serve.md"; : > "$TD/t/book/src/cli/index.md"
    : > "$TD/t/book/src/lib/core.md"
}
readme() { # every block stale, plus authored prose the generator must never touch
    {
        printf '# T\n'
        printf '| **<!-- CRATE_COUNT_START -->9<!-- CRATE_COUNT_END -->** workspace crates |\n'
        printf '| **<!-- CONTRACT_COUNT_START -->9<!-- CONTRACT_COUNT_END -->** provable contracts |\n'
        printf '| **<!-- CLI_COMMAND_COUNT_START -->9<!-- CLI_COMMAND_COUNT_END -->** CLI commands |\n'
        printf '| **<!-- BOOK_CLI_CHAPTER_COUNT_START -->9<!-- BOOK_CLI_CHAPTER_COUNT_END -->** chapters |\n'
        printf '| **<!-- BOOK_LIB_CHAPTER_COUNT_START -->hand-typed prose<!-- BOOK_LIB_CHAPTER_COUNT_END -->** chapters |\n'
        printf '| **<!-- CRATES_DIR_COUNT_START -->9<!-- CRATES_DIR_COUNT_END -->** directories |\n'
        printf 'The tree carries <!-- CONTRACT_COUNT_START -->8<!-- CONTRACT_COUNT_END --> contracts.\n'
        printf 'Both on one line: <!-- CLI_COMMAND_COUNT_START -->7<!-- CLI_COMMAND_COUNT_END --> and <!-- CLI_COMMAND_COUNT_START -->6<!-- CLI_COMMAND_COUNT_END -->.\n'
        printf 'An authored line: **99** chapters.\n'
    } > "$TD/README.md"
}
plant || exit 2

gen --list-blocks
[ "$RC" = 0 ] && [ "$(tr '\n' ' ' < "$TD/out")" = 'CRATE_COUNT CONTRACT_COUNT CLI_COMMAND_COUNT BOOK_CLI_CHAPTER_COUNT BOOK_LIB_CHAPTER_COUNT CRATES_DIR_COUNT ' ]
row "--list-blocks names the six owned blocks, in README order" $?

gen --print-all
[ "$RC" = 0 ] && [ "$(tr '\n' ' ' < "$TD/out")" = 'CRATE_COUNT=2 CONTRACT_COUNT=3 CLI_COMMAND_COUNT=2 BOOK_CLI_CHAPTER_COUNT=3 BOOK_LIB_CHAPTER_COUNT=1 CRATES_DIR_COUNT=3 ' ]
row "--print-all measures each instrument (members not dirs, census not find, help excluded)" $?

readme
gen --check; [ "$RC" = 1 ];                                                row "--check on a stale README is RED (rc=$RC)" $?
gen --write; [ "$RC" = 0 ];                                                row "--write rewrites every block (rc=$RC)" $?
grep -qxF '| **<!-- CRATE_COUNT_START -->2<!-- CRATE_COUNT_END -->** workspace crates |' "$TD/README.md" \
 && grep -qxF '| **<!-- CRATES_DIR_COUNT_START -->3<!-- CRATES_DIR_COUNT_END -->** directories |' "$TD/README.md" \
 && grep -qxF 'The tree carries <!-- CONTRACT_COUNT_START -->3<!-- CONTRACT_COUNT_END --> contracts.' "$TD/README.md"
row "  ...every block, including a second CONTRACT_COUNT site, now carries the measurement" $?
grep -qxF '| **<!-- BOOK_LIB_CHAPTER_COUNT_START -->1<!-- BOOK_LIB_CHAPTER_COUNT_END -->** chapters |' "$TD/README.md"
row "  ...a block body hand-edited to prose is replaced, not skipped" $?
grep -qxF 'Both on one line: <!-- CLI_COMMAND_COUNT_START -->2<!-- CLI_COMMAND_COUNT_END --> and <!-- CLI_COMMAND_COUNT_START -->2<!-- CLI_COMMAND_COUNT_END -->.' "$TD/README.md"
row "  ...two blocks on ONE line are both rewritten" $?
grep -qxF 'An authored line: **99** chapters.' "$TD/README.md"
row "  ...an authored number OUTSIDE a block is never touched" $?
cp "$TD/README.md" "$TD/once"; gen --write; cmp -s "$TD/README.md" "$TD/once"
row "--write twice is byte-identical (a fixpoint)" $?
gen --check; [ "$RC" = 0 ];                                                row "--check after --write is GREEN (rc=$RC)" $?

readme; sed -i 's|<!-- CRATES_DIR_COUNT_START -->9<!-- CRATES_DIR_COUNT_END -->|9|' "$TD/README.md"
gen --write; [ "$RC" = 3 ] && grep -q 'CRATES_DIR_COUNT' "$TD/out"
row "a README missing an owned block is exit 3 naming it -- the count is authored (rc=$RC)" $?
gen --check; [ "$RC" = 3 ] && grep -q 'CRATES_DIR_COUNT' "$TD/out"
row "--check on a README missing an owned block is exit 3 too, never a vacuous ok (rc=$RC)" $?

readme; rm -f -- "${TD:?}/t/book/src/lib/core.md"
gen --write; [ "$RC" = 2 ] && grep -q 'BOOK_LIB_CHAPTER_COUNT: measured 0' "$TD/out" && grep -q 'hand-typed prose' "$TD/README.md"
row "a zero measurement is a broken instrument: exit 2, README untouched (rc=$RC)" $?
plant || exit 2

readme; rm -f -- "${TD:?}/t/contracts/census.json"
gen --write; [ "$RC" = 2 ] && grep -q 'UNMEASURED' "$TD/out"
row "a missing census is UNMEASURED, exit 2, never a zero (rc=$RC)" $?
plant || exit 2

printf '[workspace\n' > "$TD/t/Cargo.toml"
gen --print-all; [ "$RC" = 2 ] && grep -q 'CRATE_COUNT' "$TD/out"
row "a cargo metadata that fails is a failed CRATE_COUNT measurement, exit 2 (rc=$RC)" $?
plant || exit 2

gen --print; [ "$RC" = 0 ] && [ "$(cat "$TD/out")" = '<!-- CONTRACT_COUNT_START -->3<!-- CONTRACT_COUNT_END -->' ]
row "--print keeps its contract-block form (check_readme_claims.sh README-002 reads it)" $?

if [ "$red" = 0 ]; then printf '%s rows, PASS\n' "$n"; exit 0; fi
printf '%s rows, FAIL\n' "$n"; exit 1
