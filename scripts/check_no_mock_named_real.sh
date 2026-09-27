#!/usr/bin/env bash
# check_no_mock_named_real.sh - a mock is named as a mock (TRACE-001 R-5, TR-04).
#
# THE CLASS. crates/aprender-serve/tests/modality_matrix/common.rs carried
# `pub mod renacer`: a thread-local span recorder that traced nothing. Its
# QA-A08 test printed "renacer::capture() API works correctly", and a reader of
# that line had no way to tell it was not the in-tree renacer. A module or type
# declared outside a workspace crate but named after that crate's library reads
# as the real crate. The module is now `mock_trace`.
#
# The check is Rust (TRACE-001 R-3): scripts/guards/no_mock_named_real.rs, std
# only, compiled here with rustc. Library names are derived from every tracked
# Cargo.toml; declarations that predate the rule are frozen in
# scripts/no_mock_named_real_baseline.txt, which only shrinks (a NEW
# declaration and a STALE baseline line both fail).
#
# REPORT-ONLY (TRACE-001 R-6): a new gate reports until first-green plus 7
# green nights. Until then a NEW or STALE line prints and the exit is 0; set
# NMR_ENFORCE=1 to make it fail. The self-test is not a verdict on the tree
# and always fails loudly: a guard that cannot turn RED is broken.
#
#   bash scripts/check_no_mock_named_real.sh              # check (report-only)
#   bash scripts/check_no_mock_named_real.sh --self-test  # planted `mod trueno` must be RED
#
# Exit 0 = clean, or report-only findings. 1 = self-test failed, or findings
# with NMR_ENFORCE=1. 2 = cannot judge (with NMR_ENFORCE=1).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/scripts/guards/no_mock_named_real.rs"
BASELINE="${REPO_ROOT}/scripts/no_mock_named_real_baseline.txt"
OUT="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}/guards/no_mock_named_real"
BIN="${OUT}/nmr"

case "${1:-}" in
    -h|--help)
        sed -n '2,27p' "$0"
        exit 0
        ;;
esac

if ! command -v rustc >/dev/null 2>&1; then
    printf 'NOT RUN: no rustc on PATH; no-mock-named-real-v1 cannot judge (NotRun, not green)\n'
    if [ "${NMR_ENFORCE:-0}" = "1" ]; then exit 2; fi
    exit 0
fi

mkdir -p "$OUT" || exit 2
if ! rustc --edition 2021 -O -o "$BIN" "$SRC"; then
    printf 'ERROR: %s does not compile\n' "$SRC" >&2
    exit 1
fi

if [ "${1:-}" = "--self-test" ]; then
    # 1. The Rust case table (must-match/must-not-match, planted mod, stale baseline).
    if ! rustc --edition 2021 --test -o "${OUT}/nmr-test" "$SRC"; then
        printf 'SELF-TEST FAILED: the case table does not compile\n' >&2
        exit 1
    fi
    "${OUT}/nmr-test" -q; rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'SELF-TEST FAILED: case table rc=%s\n' "$rc" >&2
        exit 1
    fi

    # 2. End to end over a planted tree: a crate whose library is `trueno`, and a
    #    test outside it declaring `mod trueno`. Files are rewritten in place each
    #    run, so nothing needs deleting.
    plant="${OUT}/plant"
    mkdir -p "${plant}/crates/aprender-compute" "${plant}/crates/aprender-serve/tests" || exit 1
    printf '[package]\nname = "aprender-compute"\n\n[lib]\nname = "trueno"\n' \
        > "${plant}/crates/aprender-compute/Cargo.toml"
    list="${OUT}/plant.list"
    printf '%s\0%s\0' crates/aprender-compute/Cargo.toml crates/aprender-serve/tests/planted.rs > "$list"

    printf 'pub mod trueno {\n    pub fn matmul() {}\n}\n' > "${plant}/crates/aprender-serve/tests/planted.rs"
    "$BIN" "$plant" < "$list" > "${OUT}/plant.out"; rc=$?
    if [ "$rc" -ne 1 ]; then
        printf 'SELF-TEST FAILED: planted mod trueno in a test returned rc=%s, want 1\n' "$rc" >&2
        exit 1
    fi

    printf 'pub mod mock_trueno {\n    pub fn matmul() {}\n}\n' > "${plant}/crates/aprender-serve/tests/planted.rs"
    "$BIN" "$plant" < "$list" > "${OUT}/plant.out"; rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'SELF-TEST FAILED: mod mock_trueno returned rc=%s, want 0\n' "$rc" >&2
        exit 1
    fi
    printf 'self-test OK: planted mod trueno in a test is RED, mod mock_trueno is GREEN\n'
    exit 0
fi

# The baseline may only shrink against origin/main (lib_baseline_ratchet.sh).
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 2
baseline_ratchet_check "${REPO_ROOT}" scripts/no_mock_named_real_baseline.txt set; ratchet_rc=$?

files="${OUT}/files.list"
git -C "$REPO_ROOT" ls-files -z > "$files" || exit 2
"$BIN" "$REPO_ROOT" --baseline "$BASELINE" < "$files"; rc=$?

if [ "$rc" -eq 0 ] && [ "$ratchet_rc" -eq 0 ]; then
    exit 0
fi
[ "$rc" -ne 0 ] || rc=1
if [ "${NMR_ENFORCE:-0}" = "1" ]; then
    exit "$rc"
fi
printf 'REPORT-ONLY (TRACE-001 R-6): no-mock-named-real-v1 rc=%s is reported, not enforced, until first-green + 7 green nights\n' "$rc"
exit 0
