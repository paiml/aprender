#!/usr/bin/env bash
#
# check_duplicate_bin_names.sh — no two crates may declare the same `[[bin]]
# name` without a declared reason.
#
# WHY THIS EXISTS (aprender#2558)
# -------------------------------
# Nothing detected it. MEASURED on crates.io 2026-08-21, FOUR things claimed the
# name `pv`, and this repository owned two of them:
#
#   1. crates.io `pv` — "Rust reimplementation of the unix pipeview (pv)
#      utility", bin_names ["pv"], has_lib false, 7,065 downloads since
#      2019-10-27, latest 0.4.0 on 2025-07-19
#   2. pv(1), the C pipe viewer, at /usr/bin/pv in every distro
#   3. aprender-contracts-cli   — [[bin]] name = "pv"
#   4. provable-contracts-cli   — [[bin]] name = "pv"   (the facade)
#
# All four target ~/.cargo/bin/pv. MEASURED: `cargo install` does NOT overwrite
# across packages -- it fails closed with exit 101 ("binary `pv` already exists
# in destination as part of <package>") and the FIRST binary survives. So a
# duplicate bin name BLOCKS an install rather than clobbering one, which is why
# this guard exists: the obstruction is silent until a user hits it.
# A shadowed artifact is worse than a missing one: edits look effective and
# change nothing. ~/.local/bin/apr shadowing a fresh install cost 24 days here
# once, and a user-scope skill shadowing the repo's cost another investigation.
#
# THE FALSE POSITIVE THIS GUARD MUST NOT RAISE
# --------------------------------------------
# `apr` is declared TWICE ON PURPOSE — by crates/apr-cli (src/main.rs) and by
# the root `aprender` facade (src/bin/apr.rs, required-features = ["cli"]). A
# naive duplicate check reds on main immediately and gets disabled within a
# week. So intent is MODELLED, in scripts/duplicate_bin_names_allowlist.txt,
# with the declared claimant set required to match the observed one EXACTLY:
# an undeclared duplicate FAILS, a declared one passes, a THIRD claimant on a
# declared name FAILS, and a stale entry FAILS.
#
# WHY IT SCANS TWO WORKSPACES
# ---------------------------
# crates/facades is `exclude`d from the root workspace (Cargo.toml), because the
# facades share lib names with the crates they front and would collide on the
# uplifted rlib. `cargo metadata` on the root therefore CANNOT SEE the facade —
# a root-only scan would have been inert against the very collision that
# motivated this guard. That is the pre-filter defect that made a whole guard
# class dead earlier this week, so W1/W2 in scripts/lib/bin_names.py make the
# coverage itself falsifiable: W1 requires 2+ workspace documents and W2
# requires a `provable-contracts*` package to be among the packages scanned.
#
# WHY BASH AND NOT pv
# -------------------
# Same division of labour as check_facade_compat.sh: `pv`'s universe is
# contracts/**.yaml, and the subject here is Cargo manifests. The pv-native half
# is contracts/provable-contracts-facade-v1.yaml, which states the promise and
# binds its falsification tests to this script.
#
#   bash scripts/check_duplicate_bin_names.sh              # check
#   bash scripts/check_duplicate_bin_names.sh --self-test  # case table, no cargo
#   bash scripts/check_duplicate_bin_names.sh --list       # every bin, one per line
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINS="${REPO_ROOT}/scripts/lib/bin_names.py"
ALLOW="${REPO_ROOT}/scripts/duplicate_bin_names_allowlist.txt"
CASES="${REPO_ROOT}/scripts/lib/bin_name_cases"
FACADE_WS="${REPO_ROOT}/crates/facades"

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    # Fixtures are committed JSON under scripts/lib/bin_name_cases/ rather than
    # inline heredocs: bashrs parses an embedded heredoc as shell.
    fails=0

    run_case() {  # name expect_rc allowlist needle metadata...
        local name="$1" want="$2" allow="$3" needle="$4" out rc
        shift 4
        out="$( python3 "$BINS" "$CASES/$allow" "$@" 2>&1 )"; rc=$?
        if [ "$rc" != "$want" ]; then
            printf 'FAIL  %s: exit %s, expected %s\n%s\n' "$name" "$rc" "$want" "$out"
            fails=1; return
        fi
        # -F: the needles contain `[NONE]`, `` ` `` and `{}`; as a REGEX
        # `[NONE]` matches a single character from {N,O,E} and would silently
        # never match the literal text. A needle that cannot match reads exactly
        # like a guard that is fine.
        if [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out"; then
            printf 'FAIL  %s: exit %s as expected but did not name %s\n%s\n' \
                "$name" "$rc" "$needle" "$out"
            fails=1; return
        fi
        printf 'ok    %s\n' "$name"
    }

    # Row 1 pairs the clean tree with an EMPTY allowlist on purpose: pairing it
    # with allow_apr would trip the stale-entry rule (row 6), which is correct
    # behaviour but a different claim.
    run_case 'row 1 a tree with no duplicate bin names passes' \
        0 allow_empty.txt 'D  no bin name is claimed by two packages' \
        "root=$CASES/root_clean.json" "facades=$CASES/facades_libonly.json"

    run_case 'row 2 the INTENTIONAL `apr` duplicate passes when declared' \
        0 allow_apr.txt 'declared intentional' \
        "root=$CASES/root_apr_dup.json" "facades=$CASES/facades_libonly.json"

    run_case 'row 3 the same `apr` duplicate FAILS with an empty allowlist' \
        1 allow_empty.txt 'D  `apr` is declared by 2 packages' \
        "root=$CASES/root_apr_dup.json" "facades=$CASES/facades_libonly.json"

    # Row 4 is the one that only a two-workspace scan can catch: the collision
    # is between a ROOT crate and a FACADE crate, so a root-only guard is blind
    # to it. This is the exact state this branch started in.
    run_case 'row 4 a duplicate SPANNING the two workspaces is REJECTED' \
        1 allow_apr.txt 'D  `pv` is declared by 2 packages' \
        "root=$CASES/root_apr_dup.json" "facades=$CASES/facades_declares_pv.json"

    run_case 'row 5 a THIRD claimant on a declared name is REJECTED' \
        1 allow_apr.txt 'the claimant set CHANGED' \
        "root=$CASES/root_apr_triple.json" "facades=$CASES/facades_libonly.json"

    run_case 'row 6 a STALE allowlist entry is REJECTED' \
        1 allow_apr.txt 'no longer claimed by two packages' \
        "root=$CASES/root_clean.json" "facades=$CASES/facades_libonly.json"

    # Row 7 is the pre-filter control. Scanning ONLY the root workspace must
    # FAIL, not pass quietly — otherwise dropping the facade document from the
    # invocation would silently disarm the guard and look identical to green.
    run_case 'row 7 scanning ONE workspace is REJECTED (facade invisible)' \
        1 allow_apr.txt 'W1 1 workspace document(s) scanned' \
        "root=$CASES/root_apr_dup.json"

    # Row 8: two documents, but the second is not the facade workspace. W1 alone
    # would pass. W2 is what makes the coverage claim falsifiable.
    run_case 'row 8 two documents that do not include the facades is REJECTED' \
        1 allow_apr.txt 'W2 facade workspace present in the scan: [NONE]' \
        "root=$CASES/root_apr_dup.json" "other=$CASES/root_clean.json"

    run_case 'row 9 a scan that finds no bins at all is REJECTED (vacuity)' \
        1 allow_empty.txt 'W3 0 bin target(s) found' \
        "root=$CASES/empty.json" "facades=$CASES/facades_libonly.json"

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (9/9)\n'
    exit 0
fi

# ---------------------------------------------------------------------------
ROOT_MD="$(mktemp)"; FAC_MD="$(mktemp)"
trap 'rm -f "${ROOT_MD:?}" "${FAC_MD:?}"' EXIT
( cd "$REPO_ROOT" && cargo metadata --no-deps --format-version 1 ) > "$ROOT_MD" 2>/dev/null
( cd "$FACADE_WS" && cargo metadata --no-deps --format-version 1 ) > "$FAC_MD" 2>/dev/null

# A missing measurement must be RED, never a silent pass over nothing.
if [ ! -s "$ROOT_MD" ] || [ ! -s "$FAC_MD" ]; then
    printf 'VACUOUS: cargo metadata produced no document; nothing was checked.\n'
    exit 1
fi

if [ "${1:-}" = "--list" ]; then
    python3 "$BINS" --list "root=$ROOT_MD" "facades=$FAC_MD"
    exit 0
fi

printf '=== no two crates may claim one bin name (check_duplicate_bin_names.sh) ===\n\n'
python3 "$BINS" "$ALLOW" "root=$ROOT_MD" "facades=$FAC_MD"; rc=$?

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  every duplicated bin name is declared intentional, and every\n'
    printf '      declaration still describes the tree\n'
else
    # No apostrophe in the prose: bashrs reads the `'"'"'` escape as an
    # unterminated string (SC1078) and the ratchet counts errors, not opinions.
    printf 'FAIL  see rows above. Two packages claiming one ~/.cargo/bin/<name>\n'
    printf '      makes `cargo install` FAIL CLOSED (exit 101) for anyone holding\n'
    printf '      the other -- it blocks the install. Rename the binary on one, or\n'
    printf '      record the intent in %s\n' "$(basename "$ALLOW")"
fi
exit "$rc"
