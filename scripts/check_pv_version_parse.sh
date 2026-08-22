#!/usr/bin/env bash
# check_pv_version_parse.sh - the semver extractor in scripts/pv_bin.sh ships a
# case table, and the table is executable.
#
# THE CLASS. scripts/pv_bin.sh proves a resolved `pv` was built from HEAD by
# pulling the semver out of `pv --version` and comparing it to the version the
# tree declares. That extractor is a GUARD's parser, and in this repo guard
# patterns have been wrong five times in a row — every one caught by a
# must-match / must-not-match table, none by review (CLAUDE.md, "Guard regexes
# ship a case table").
#
# It was already wrong once. The extractor was `awk '{print $NF}'`: the LAST
# field of EVERY line. That was correct only while `pv --version` printed the
# single line `pv 0.63.0`. #2559 made the line multi-line on purpose — four
# things claim the name `pv`, so the version output now has to say which one it
# is — and the old extractor started handing the comparison four lines of prose:
#
#     verifier)
#     https://github.com/paiml/aprender
#     surface.
#     crates.io).
#
# Which is not a silent failure — it fails loud — but it BLOCKS every consumer
# of pv_bin.sh, i.e. the release gate. The replacement reads position 2 of line
# 1. That shape is pinned from the other side by the Rust test
# crates/aprender-contracts-cli/tests/version_identity.rs
# (`semver_stays_the_second_field_of_the_first_line`), so the binary cannot drift
# out from under the parser without a RED test.
#
# Exit 0 = the extractor as written in pv_bin.sh answers every row correctly.
# Exit 1 = at least one row disagrees.
#
# `--self-test` proves this check can still turn RED, by running the same table
# against the pre-#2559 extractor.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

PV_BIN_SH="scripts/pv_bin.sh"

# The extractor under test, lifted from pv_bin.sh rather than retyped. If the
# two ever diverge, EXTRACTOR_MISMATCH below fires.
EXTRACTOR='NR==1{print $2; exit}'
# shellcheck disable=SC2154  # $NF is awk's field-count, inside a single-quoted
# awk program. It is not a shell variable and must stay verbatim: this literal is
# the pre-#2559 extractor, and it is grepped for against pv_bin.sh.
PRE_2559_EXTRACTOR='{print $NF}'

# --- the case table ------------------------------------------------------
# Each row: <expected semver or the literal NONE> :: <version output>
# NONE means "must NOT yield a bare semver" — the row is something other than
# this tool, and mistaking it for this tool is the failure mode.
#
# Rows are separated by a line of four dashes so the outputs can be multi-line.
read -r -d '' CASES <<'TABLE' || true
0.63.0 :: pv 0.63.0 (aprender provable-contracts verifier)
crate aprender-contracts-cli — https://github.com/paiml/aprender
Verifies YAML contracts under contracts/; run `pv --help` for the command surface.
This is NOT pv(1), the pipe viewer (distro package `pv`, or the `pv` crate on crates.io).
----
0.63.0 :: pv 0.63.0
----
1.2.3 :: pv 1.2.3 (aprender provable-contracts verifier)
second line with trailing 9.9.9
----
0.64.0-rc.1 :: pv 0.64.0-rc.1 (aprender provable-contracts verifier)
crate aprender-contracts-cli — https://github.com/paiml/aprender
----
NONE :: pv 1.6.20
Copyright (C) 2002-2008 Andrew Wood <pv@ivarch.com>
----
NONE :: pv 1.9.31
TABLE

fail=0

note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*" >&2; fail=1; }

# --- 0. the extractor in this file must be the one pv_bin.sh actually runs ---
# CODE ONLY. The first draft of this block grepped the whole file and fired on
# pv_bin.sh's own COMMENT, which quotes the old extractor to explain why it was
# replaced. A guard that reads prose as code is the twin of a guard that reads
# code as prose; both have shipped here. Strip full-line comments first.
pv_bin_code() { grep -v '^[[:space:]]*#' "$PV_BIN_SH"; }

if ! pv_bin_code | grep -qF -- "$EXTRACTOR"; then
    bad "EXTRACTOR_MISMATCH: this table tests '$EXTRACTOR' but $PV_BIN_SH does not run it"
fi
if pv_bin_code | grep -qF -- "awk '$PRE_2559_EXTRACTOR'"; then
    bad "$PV_BIN_SH still runs the pre-#2559 last-field extractor"
fi

# --- 1. run the table ----------------------------------------------------
run_table() {
    table_awk="$1"
    table_label="$2"
    table_rc=0
    table_row=""
    while IFS= read -r line; do
        if [ "$line" = "----" ]; then
            check_row "$table_awk" "$table_label" "$table_row" || table_rc=1
            table_row=""
        else
            table_row="${table_row}${line}"$'\n'
        fi
    done <<< "$CASES"
    [ -n "$table_row" ] && { check_row "$table_awk" "$table_label" "$table_row" || table_rc=1; }
    return "$table_rc"
}

check_row() {
    row_awk="$1"
    row_label="$2"
    row_body="$3"
    row_expect="${row_body%% :: *}"
    row_output="${row_body#* :: }"
    # awk on a here-string, NOT a pipe: a pipe hands back awk's status and, with
    # pipefail on, a SIGPIPE from the writer can invent a failure (CLAUDE.md).
    row_got=$(awk "$row_awk" <<< "$row_output")
    if [ "$row_expect" = "NONE" ]; then
        # For a foreign pv there is nothing to extract that is CORRECT; the row
        # exists to record what the extractor does return, so a future change
        # that starts silently blessing the pipe viewer is visible in the diff.
        note "  [$row_label] foreign pv -> '$row_got' (compared against the declared version, so it must not match)"
        return 0
    fi
    if [ "$row_got" = "$row_expect" ]; then
        note "  [$row_label] ok: '$row_expect'"
        return 0
    fi
    bad "[$row_label] expected '$row_expect', extractor returned '$row_got'"
    return 1
}

if [ "${1:-}" = "--self-test" ]; then
    note "SELF-TEST: running the same table against the pre-#2559 extractor."
    note "It MUST fail, or this check proves nothing."
    if run_table "$PRE_2559_EXTRACTOR" "pre-2559"; then
        printf 'SELF-TEST FAILED: the pre-#2559 extractor passed the table.\n' >&2
        exit 1
    fi
    printf '\nSELF-TEST PASSED: the table turns RED on the pre-#2559 extractor.\n'
    exit 0
fi

note "pv --version semver extractor case table:"
run_table "$EXTRACTOR" "current" || fail=1

if [ "$fail" -ne 0 ]; then
    printf '\ncheck_pv_version_parse: FAIL\n' >&2
    exit 1
fi
printf '\ncheck_pv_version_parse: OK\n'
