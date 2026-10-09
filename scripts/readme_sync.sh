#!/usr/bin/env bash
# readme_sync.sh — the README's contract count is DERIVED, not authored.
#
# BSE-03 phase A (docs/specifications/build-system-enhancement.md, `paiml/infra`
# §4 wave 4; Pmat-Ticket: PMAT-1068; model:
# docs/audits/threat-model-bse-03-ratchets.md, attack A7 and row P7).
#
# THE DEFECT THIS EXISTS FOR
# --------------------------
# The count lived in README.md as three hand-written literals in three separate
# prose sites, and NOTHING regenerated them: `check_readme_claims.sh --regen`
# only PRINTS numbers "for manual README edit", and `aprender-qa-readme-sync` rewrites
# a certification table between markers README.md does not carry. The literals
# were measured at 1812 against a filesystem carrying 1814 — two behind, and
# green, because the guard lets the README lag. A number a human must copy in
# three places is a number that is wrong between merges by construction.
#
# WHAT THIS REWRITES, AND WHAT IT WILL NOT TOUCH
# ----------------------------------------------
# Exactly the text BETWEEN the markers
#
#     <!-- CONTRACT_COUNT_START -->N<!-- CONTRACT_COUNT_END -->
#
# wherever they occur, and nothing else in the file — no reflow, no other
# number, no line the markers do not delimit. The markers are INLINE (both on
# one line, with the count between them) on purpose: a marker on its own line
# terminates a GFM table and starts an HTML block, so a line-delimited block
# could not sit in README.md's metrics table row or mid-paragraph without
# changing how the file renders.
#
# The count is contracts/census.json `.n_files`: the release train's snapshot,
# written only by `make census` (#3569). Since GEN-001 (#4526) the README states
# that snapshot, not the walked tree, so a PR that adds a contract no longer
# edits README.md and the README count has one writer, like the census. It lags
# the tree between releases by design. check_readme_claims.sh holds every block
# EQUAL to the snapshot in the merge commit.
#
# Modes:
#   --write    rewrite every block in README.md, in place, atomically (default
#              for `make readme-sync`). Idempotent: running it twice produces a
#              byte-identical file, because the substitution is a fixpoint.
#   --print    print the block bytes it WOULD write, to stdout. Writes nothing.
#              This is what check_readme_claims.sh runs (twice, comparing the
#              bytes) when README.md carries no block at all.
#   --check    exit 0 if README.md already equals what --write would produce,
#              1 otherwise, naming both numbers. Writes nothing.
#   --walked   print the WORKING TREE's contract count (contract_files_only).
#              Not a README number: contracts_gate.sh holds it equal to a
#              fresh `pv census`.
#
# A README that carries NO block is exit 3 under --write, never a silent no-op:
# "rewrote 0 blocks" over a file whose count is stale is the vacuous-scan class
# this repository keeps finding in its own guards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="${README_PATH:-$REPO_ROOT/README.md}"   # README_PATH: a fixture, for the tests

CONTRACT_BLOCK_START='<!-- CONTRACT_COUNT_START -->'
CONTRACT_BLOCK_END='<!-- CONTRACT_COUNT_END -->'

usage() {
    cat >&2 <<'USAGE'
usage: bash scripts/readme_sync.sh [--write|--print|--check|--walked]
  --write  rewrite every CONTRACT_COUNT block in README.md (idempotent)
  --print  print the block bytes --write would produce; write nothing
  --check  exit 0 iff README.md already matches; 1 otherwise
  --walked print the working tree's contract count (not a README number)
USAGE
    exit 2
}

# The walked measurement. One instrument, shared with check_readme_claims.sh's
# measured_contract_count(): find over contracts/, *.yaml, any depth.
# ONE definition of "a contract file" — kept byte-identical to
# scripts/check_readme_claims.sh's, and held to `pv census` by `make contracts`.
contract_files_only() {
  grep -E '\.yaml$' \
    | grep -Ev '(^|/)(kaizen|legacy|pipelines|publish-manifests|quarantine)/' \
    | grep -Ev '(^|/)(binding\.yaml|binding\.yml|external-corpora\.yaml|ontology\.yaml)$' \
    | grep -Ev '(^|/)\.[^/]*$'
}

walked_contract_count() {
    # The WORKING TREE's count, with the one definition of "a contract file" that
    # `pv census` (ONT-001 ONT-1) and the lint walker apply (contract_files_only).
    # It is NOT what the README states (see snapshot_contract_count). `--walked`
    # prints it so `contracts_gate.sh census` can prove this listing and a fresh
    # `pv census` still agree.
    local n
    n=$(cd "$REPO_ROOT" && { find contracts -type f -name '*.yaml' 2>/dev/null || true; } | contract_files_only | grep -c .) || true
    case "$n" in
        '' | *[!0-9]*)
            printf 'FAIL readme_sync: the walked contract count read %q, not a number\n' "$n" >&2
            return 1
            ;;
    esac
    [ "$n" -gt 0 ] || {
        printf 'FAIL readme_sync: the walked contract count is 0. A zero count is a broken measurement.\n' >&2
        return 1
    }
    printf '%s' "$n"
}

snapshot_contract_count() {
    # GEN-001 (#4526): the README states the RELEASE SNAPSHOT, not the walked tree.
    # It used to state the walked tree, and FALSIFY-README-002 held that block EQUAL
    # to the merge tree, so every PR that added a contract had to rewrite the same
    # README line and two such PRs conflicted on it (#4502: 1887 vs 1888). The count
    # now comes from contracts/census.json `.n_files`, which has ONE writer -- `make
    # census` on the release train (#3569; scripts/check_census_derived.sh refuses a
    # PR that edits it) -- so a PR never edits the README count either, and the
    # README lags the tree between releases by design, as census.json does.
    # CENSUS_JSON=<file> reads another census (a fixture).
    local n src="${CENSUS_JSON:-"$REPO_ROOT/contracts/census.json"}"
    [ -s "$src" ] || {
        printf 'FAIL readme_sync: %s is missing or empty. The count is UNMEASURED, which is a failure, not a zero.\n' "$src" >&2
        return 1
    }
    n=$(jq -r '.n_files // empty' "$src" 2>/dev/null || true)
    case "$n" in
        '' | *[!0-9]*)
            printf 'FAIL readme_sync: the contract count read %q, not a number\n' "$n" >&2
            return 1
            ;;
    esac
    [ "$n" -gt 0 ] || {
        printf 'FAIL readme_sync: the contract count is 0. A zero count is a broken measurement, not a README to regenerate.\n' >&2
        return 1
    }
    printf '%s' "$n"
}

render_block() { # render_block <count>
    printf '%s%s%s' "$CONTRACT_BLOCK_START" "$1" "$CONTRACT_BLOCK_END"
}

block_occurrences() {
    grep -oF "$CONTRACT_BLOCK_START" "$README" 2>/dev/null | wc -l | tr -d ' '
}

# The rewrite. `[^<]*` is the block body: the markers themselves are the only
# thing in this file that may contain '<', so the body can never swallow the
# closing marker, and a body that has been hand-edited to prose is replaced
# just as a stale number is.
# ONT-4c (B.4): the frontmatter's `contract_count:` is graded `resolves: census` by `extract:readme` — it must
# equal the TRACKED contracts/census.json `n_files`, not the walked tree. The two differ on every branch that adds a
# contract, because census.json is the release train's snapshot (#3569) and lags by design; writing the walked count
# there made readme_sync and the extractor demand different numbers (#4429: 1844 vs 1837, one of them always RED).
# Since GEN-001 (#4526) the body blocks follow the same census, so the README carries ONE contract count.
rewrite_stream() { # rewrite_stream SNAPSHOT_COUNT, README on stdin
    # Only a line that is exactly `contract_count: N` matches; the frontmatter is the only place README.md carries one.
    sed -E -e "s|(${CONTRACT_BLOCK_START})[^<]*(${CONTRACT_BLOCK_END})|\1${1}\2|g" \
        -e "s|^contract_count: [0-9]+\$|contract_count: ${1}|"
}

mode=""
for arg in "$@"; do
    case "$arg" in
        --write) mode=write ;;
        --print) mode=print ;;
        --check) mode=check ;;
        --walked) mode=walked ;;
        -h|--help) usage ;;
        *) printf 'unknown arg: %s\n' "$arg" >&2; usage ;;
    esac
done
[ -n "$mode" ] || usage

if [ ! -f "$README" ]; then
    printf 'FAIL readme_sync: %s not found\n' "$README" >&2
    exit 2
fi
if [ ! -d "$REPO_ROOT/contracts" ]; then
    printf 'FAIL readme_sync: %s/contracts does not exist, so the count is UNMEASURED — that is a failure, not a zero.\n' "$REPO_ROOT" >&2
    exit 2
fi

if [ "$mode" = walked ]; then
    walked_contract_count || exit 2
    printf '\n'
    exit 0
fi

count="$(snapshot_contract_count)" || exit 2

case "$mode" in
    print)
        render_block "$count"
        printf '\n'
        exit 0
        ;;
    check)
        tmp="$(mktemp "${TMPDIR:-/tmp}/readme-sync-check.XXXXXX")"
        # shellcheck disable=SC2064
        trap "rm -f '$tmp'" EXIT
        rewrite_stream "$count" < "$README" > "$tmp"
        if cmp -s "$README" "$tmp"; then
            printf 'ok    readme_sync: README.md already states the release snapshot count %s (contracts/census.json) in every CONTRACT_COUNT block and the frontmatter\n' "$count"
            exit 0
        fi
        printf 'FAIL readme_sync: README.md does not match what the generator produces (contracts/census.json n_files is %s). Run: make readme-sync\n' "$count" >&2
        diff -u "$README" "$tmp" | sed 's/^/  | /' >&2 || true
        exit 1
        ;;
    write)
        n="$(block_occurrences)"
        if [ "$n" -eq 0 ]; then
            printf 'FAIL readme_sync: %s carries no %s marker, so there is nothing to regenerate and the count is AUTHORED.\n' \
                "$README" "$CONTRACT_BLOCK_START" >&2
            printf '     Add the markers around the number, inline:  %s\n' "$(render_block "$count")" >&2
            exit 3
        fi
        tmp="$(mktemp "${TMPDIR:-/tmp}/readme-sync.XXXXXX")"
        # shellcheck disable=SC2064
        trap "rm -f '$tmp'" EXIT
        rewrite_stream "$count" < "$README" > "$tmp"
        cat "$tmp" > "$README"
        # Read the file BACK and assert the effect, rather than reporting that
        # bytes were written: every block must now carry the measured count.
        after="$(grep -oE "${CONTRACT_BLOCK_START}[^<]*${CONTRACT_BLOCK_END}" "$README" | grep -cF "$(render_block "$count")")" || after=0
        if [ "$after" -ne "$n" ]; then
            printf 'FAIL readme_sync: %s of %s block(s) carry the snapshot count %s after the rewrite.\n' "$after" "$n" "$count" >&2
            exit 1
        fi
        printf 'ok    readme_sync: %s CONTRACT_COUNT block(s) now state %s (contracts/census.json n_files, the release snapshot)\n' "$n" "$count"
        exit 0
        ;;
esac
