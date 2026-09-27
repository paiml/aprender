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
# GEN-001 (#4526): ONE GENERATOR FOR EVERY DERIVED COUNT
# ---------------------------------------------------------
# The contract count was the only block; the crate, CLI-command, book-chapter and
# crate-directory counts were typed by hand, and a second writer
# (session_docs_commit.sh) sed-patched two of them. Measured on main @761d6247de
# the day this landed: 4 of the 5 hand-typed counts were stale (CLI 111 vs 112,
# book cli 113 vs 114, book lib 72 vs 73, crates/ dirs 84 vs 86). Every count is
# now a block this script owns (BLOCKS below, `--list-blocks`), each measured by
# ONE instrument. PRs never edit a block (the G2 guard keys on --list-blocks);
# the fold/regeneration path (batch_fold.sh --regen, `make readme-sync`) runs
# --write once on the integrated tree (G4).
#
# Modes:
#   --write    rewrite every block in README.md, in place, atomically (default
#              for `make readme-sync`). Idempotent: running it twice produces a
#              byte-identical file, because the substitution is a fixpoint.
#   --print    print the block bytes it WOULD write, to stdout. Writes nothing.
#              This is what check_readme_claims.sh runs (twice, comparing the
#              bytes) when README.md carries no block at all.
#   --print-all  print NAME=N for every block, one per line. Writes nothing.
#                This is the comparand readme_contract.rs reads (G3).
#   --list-blocks  print the block NAMEs this script owns, one per line.
#   --check    exit 0 if README.md already equals what --write would produce,
#              1 otherwise, naming both numbers. Writes nothing.
#
# A README missing ANY owned block is exit 3 under --write AND --check, never a silent no-op:
# "rewrote 0 blocks" over a file whose count is stale is the vacuous-scan class
# this repository keeps finding in its own guards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="${README_PATH:-$REPO_ROOT/README.md}"   # README_PATH: a fixture, for the tests
ROOT="${README_SYNC_ROOT:-$REPO_ROOT}"          # README_SYNC_ROOT: the tree measured, for the tests

# The blocks this script owns, in README order. The G2 guard and readme_contract.rs
# read this list through --list-blocks; nothing else may hardcode it.
BLOCKS=(CRATE_COUNT CONTRACT_COUNT CLI_COMMAND_COUNT BOOK_CLI_CHAPTER_COUNT BOOK_LIB_CHAPTER_COUNT CRATES_DIR_COUNT)

CONTRACT_BLOCK_START='<!-- CONTRACT_COUNT_START -->'

usage() {
    cat >&2 <<'USAGE'
usage: bash scripts/readme_sync.sh [--write|--print|--print-all|--list-blocks|--check]
  --write        rewrite every owned block in README.md (idempotent)
  --print        print the CONTRACT_COUNT block bytes --write would produce
  --print-all    print NAME=N for every owned block
  --list-blocks  print the owned block names
  --check        exit 0 iff README.md already matches; 1 otherwise; 3 an owned block is missing
USAGE
    exit 2
}

fail() { printf 'FAIL readme_sync: %s\n' "$*" >&2; return 1; }

# Each instrument prints one positive integer or fails. A zero or an empty parse is a
# broken MEASUREMENT, never a README to regenerate.
measure_CONTRACT_COUNT() {
    # ONT-001 ONT-1 (F-1): the census's `n_files` — the set `pv lint` walks — not a
    # `find`, which counts 51 files the gate never validates.
    local census="$ROOT/contracts/census.json"
    [ -s "$census" ] || { fail "$census is missing or empty — run \`make contracts\`. The count is UNMEASURED, which is a failure, not a zero."; return 1; }
    jq -r '.n_files // empty' "$census" 2>/dev/null || true
}
measure_CRATE_COUNT() {
    # workspace members, which is what "workspace crates" means — NOT the crates/ dirs
    (cd "$ROOT" && "${CARGO:-cargo}" metadata --no-deps --format-version 1 2>/dev/null) | jq -r '.packages | length' 2>/dev/null || true
}
measure_CLI_COMMAND_COUNT() {
    # the registry's §commands, `help` excluded (clap's freebie) — check_readme_claims.sh's instrument
    python3 -c '
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1])) or {}
names = {c.get("name") for c in doc.get("commands") or [] if isinstance(c, dict) and c.get("name")}
names.discard("help")
print(len(names))
' "$ROOT/contracts/apr-cli-commands-v1.yaml" 2>/dev/null || true
}
count_glob() { local n=0 f; for f in "$@"; do [ -e "$f" ] && n=$((n + 1)); done; printf '%s' "$n"; }
measure_BOOK_CLI_CHAPTER_COUNT() { count_glob "$ROOT"/book/src/cli/*.md; }
measure_BOOK_LIB_CHAPTER_COUNT() { count_glob "$ROOT"/book/src/lib/*.md; }
measure_CRATES_DIR_COUNT() { count_glob "$ROOT"/crates/*/; }

measure() { # measure <NAME> -> N, or fail naming the block
    local v
    v="$("measure_$1")" || return 1
    case "$v" in
        '' | *[!0-9]*) fail "$1: the instrument produced '${v}', not a number"; return 1 ;;
    esac
    [ "$v" -gt 0 ] || { fail "$1: measured 0. A zero count is a broken measurement, not a README to regenerate."; return 1; }
    printf '%s' "$v"
}

render_block() { # render_block <count> — the CONTRACT_COUNT block (--print, check_readme_claims.sh README-002)
    printf '%s%s%s' "$CONTRACT_BLOCK_START" "$1" '<!-- CONTRACT_COUNT_END -->'
}

occurrences() { # occurrences <NAME>
    grep -oF "<!-- $1_START -->" "$README" 2>/dev/null | wc -l | tr -d ' '
}

# The rewrite. `[^<]*` is the block body: the markers are the only thing in this file
# that may contain '<' inside a block, so a body can never swallow its closing marker,
# and a body hand-edited to prose is replaced just as a stale number is.
SED_ARGS=()
build_sed() {
    local name
    SED_ARGS=()
    for name in "${BLOCKS[@]}"; do
        SED_ARGS+=(-e "s|(<!-- ${name}_START -->)[^<]*(<!-- ${name}_END -->)|\\1${VAL[$name]}\\2|g")
    done
    # ONT-4c (B.4): the frontmatter's `contract_count:` is the same derived number.
    SED_ARGS+=(-e "s|^contract_count: [0-9]+\$|contract_count: ${VAL[CONTRACT_COUNT]}|")
}
rewrite_stream() { sed -E "${SED_ARGS[@]}"; }

mode=""
for arg in "$@"; do
    case "$arg" in
        --write) mode=write ;;
        --print) mode=print ;;
        --print-all) mode=print-all ;;
        --list-blocks) mode=list ;;
        --check) mode=check ;;
        -h|--help) usage ;;
        *) printf 'unknown arg: %s\n' "$arg" >&2; usage ;;
    esac
done
[ -n "$mode" ] || usage

if [ "$mode" = list ]; then
    printf '%s\n' "${BLOCKS[@]}"
    exit 0
fi

if [ ! -d "$ROOT/contracts" ]; then
    printf 'FAIL readme_sync: %s/contracts does not exist, so the count is UNMEASURED — that is a failure, not a zero.\n' "$ROOT" >&2
    exit 2
fi

declare -A VAL=()
if [ "$mode" = print ]; then
    VAL[CONTRACT_COUNT]="$(measure CONTRACT_COUNT)" || exit 2
    render_block "${VAL[CONTRACT_COUNT]}"
    printf '\n'
    exit 0
fi
for name in "${BLOCKS[@]}"; do
    VAL[$name]="$(measure "$name")" || exit 2
done

if [ "$mode" = print-all ]; then
    for name in "${BLOCKS[@]}"; do printf '%s=%s\n' "$name" "${VAL[$name]}"; done
    exit 0
fi

if [ ! -f "$README" ]; then
    printf 'FAIL readme_sync: %s not found\n' "$README" >&2
    exit 2
fi
build_sed
summary() { local name out=""; for name in "${BLOCKS[@]}"; do out="$out $name=${VAL[$name]}"; done; printf '%s' "${out# }"; }

# A missing block is refused by --check as well as --write: a README with a block deleted
# (markers and all) has nothing for the rewrite to match, so a bare cmp would call it exact.
missing=""
for name in "${BLOCKS[@]}"; do
    [ "$(occurrences "$name")" -gt 0 ] || missing="$missing $name"
done
if [ -n "$missing" ]; then
    printf 'FAIL readme_sync: %s carries no block for:%s — that count is AUTHORED, not derived.\n' "$README" "$missing" >&2
    for name in $missing; do
        printf '     Add the markers around the number, inline:  <!-- %s_START -->%s<!-- %s_END -->\n' "$name" "${VAL[$name]}" "$name" >&2
    done
    exit 3
fi

case "$mode" in
    check)
        tmp="$(mktemp "${TMPDIR:-/tmp}/readme-sync-check.XXXXXX")"
        # shellcheck disable=SC2064
        trap "rm -f '$tmp'" EXIT
        rewrite_stream < "$README" > "$tmp"
        if cmp -s "$README" "$tmp"; then
            printf 'ok    readme_sync: README.md already states every measured count (%s)\n' "$(summary)"
            exit 0
        fi
        printf 'FAIL readme_sync: README.md does not match what the generator produces (%s). Regenerate on the integrated tree: make readme-sync\n' "$(summary)" >&2
        diff -u "$README" "$tmp" | sed 's/^/  | /' >&2 || true
        exit 1
        ;;
    write)
        tmp="$(mktemp "${TMPDIR:-/tmp}/readme-sync.XXXXXX")"
        # shellcheck disable=SC2064
        trap "rm -f '$tmp'" EXIT
        rewrite_stream < "$README" > "$tmp"
        cat "$tmp" > "$README"
        # Read the file BACK and assert the effect: every block of every name carries its value.
        total=0
        for name in "${BLOCKS[@]}"; do
            n="$(occurrences "$name")"
            good="$(grep -oE "<!-- ${name}_START -->[^<]*<!-- ${name}_END -->" "$README" | grep -cxF "<!-- ${name}_START -->${VAL[$name]}<!-- ${name}_END -->")" || good=0
            if [ "$good" -ne "$n" ]; then
                printf 'FAIL readme_sync: %s of %s %s block(s) carry the measured %s after the rewrite.\n' "$good" "$n" "$name" "${VAL[$name]}" >&2
                exit 1
            fi
            total=$((total + n))
        done
        printf 'ok    readme_sync: %s block(s) rewritten (%s)\n' "$total" "$(summary)"
        exit 0
        ;;
esac
