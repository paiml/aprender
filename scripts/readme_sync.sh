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
# only PRINTS numbers "for manual README edit", and `apr-qa-readme-sync` rewrites
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
# The count is `find contracts/ -name '*.yaml' | wc -l` — the same universe
# `check_readme_claims.sh` measures, so the generator and the guard cannot
# disagree about what "a contract" is.
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
usage: bash scripts/readme_sync.sh [--write|--print|--check]
  --write  rewrite every CONTRACT_COUNT block in README.md (idempotent)
  --print  print the block bytes --write would produce; write nothing
  --check  exit 0 iff README.md already matches; 1 otherwise
USAGE
    exit 2
}

# The measurement. One instrument, shared with check_readme_claims.sh's
# measured_contract_count(): find over contracts/, *.yaml, any depth.
measured_contract_count() {
    find "$REPO_ROOT/contracts" -name '*.yaml' | wc -l | tr -d ' '
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
rewrite_stream() { # rewrite_stream <count> < README
    sed -E "s|(${CONTRACT_BLOCK_START})[^<]*(${CONTRACT_BLOCK_END})|\1${1}\2|g"
}

mode=""
for arg in "$@"; do
    case "$arg" in
        --write) mode=write ;;
        --print) mode=print ;;
        --check) mode=check ;;
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

count="$(measured_contract_count)"
if [ -z "$count" ] || [ "$count" -eq 0 ]; then
    printf 'FAIL readme_sync: contracts/ holds 0 *.yaml files. A zero count is a broken measurement, not a README to regenerate.\n' >&2
    exit 2
fi

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
            printf 'ok    readme_sync: README.md already states the measured count %s in every CONTRACT_COUNT block\n' "$count"
            exit 0
        fi
        printf 'FAIL readme_sync: README.md does not match what the generator produces (contracts/ holds %s *.yaml). Run: make readme-sync\n' "$count" >&2
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
            printf 'FAIL readme_sync: %s of %s block(s) carry the measured count %s after the rewrite.\n' "$after" "$n" "$count" >&2
            exit 1
        fi
        printf 'ok    readme_sync: %s CONTRACT_COUNT block(s) now state %s (find contracts/ -name "*.yaml")\n' "$n" "$count"
        exit 0
        ;;
esac
