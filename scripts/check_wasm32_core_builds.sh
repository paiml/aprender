#!/usr/bin/env bash
# check_wasm32_core_builds.sh - aprender-core must compile for a 32-bit target.
#
# THE CLASS (aprender#2310). Code written and only ever compiled on x86_64 picks
# up 64-bit assumptions silently. The SGD epoch shuffle in
# crates/aprender-core/src/classification/mod.rs spelled the MMIX LCG constants
# as bare literals in a usize expression:
#
#     let j = (seed * 6364136223846793005 + i * 1442695040888963407) % (i + 1);
#
# On wasm32-unknown-unknown usize is 32 bits, so that is a hard compile error
# ("literal out of range", deny-by-default overflowing_literals) - four of them,
# and aprender-core simply does not build. Nothing in the tree ever compiled for
# a 32-bit target, so it shipped in v0.60.0 and was reported from outside.
#
# This guard runs the reporter's exact command. The getrandom_backend cfg is a
# real precondition, not decoration: getrandom 0.3 refuses
# wasm32-unknown-unknown by default, and the application (not this library)
# supplies __getrandom_v03_custom. Without the flag the build dies inside
# getrandom before it ever reaches aprender-core, which would make this check
# report a failure that has nothing to do with our code.
#
# Exit 0 = aprender-core type-checks for wasm32-unknown-unknown.
# Exit 1 = it does not, or the check could not be run (fail closed - a guard that
#          cannot run must never be mistaken for a guard that passed).
#
# --self-test proves the check can still turn RED: it feeds the toolchain the
# exact #2310 literal and requires a rejection. If a future toolchain ever
# demotes overflowing_literals to a warning, or the "wasm32" target quietly
# becomes 64-bit, the self-test fails instead of this guard passing vacuously.

set -euo pipefail

TARGET="wasm32-unknown-unknown"
PKG="aprender-core"
# One of the two MMIX constants from #2310. Kept as data so the self-test and
# the comment above cannot drift apart.
DEFECT_LITERAL="6364136223846793005"
# The compiler's wording for the #2310 rejection. No backticks: they make the
# pattern ambiguous to shell linters and to anyone quoting this line.
DEFECT_DIAGNOSTIC="literal out of range"

SELF_TEST_DIR=""
CHECK_LOG=""

cleanup() {
    if [ -n "${SELF_TEST_DIR}" ] && [ -d "${SELF_TEST_DIR}" ]; then
        rm -rf "${SELF_TEST_DIR}"
    fi
    if [ -n "${CHECK_LOG}" ] && [ -f "${CHECK_LOG}" ]; then
        rm -f "${CHECK_LOG}"
    fi
}
trap cleanup EXIT

ensure_target() {
    if ! command -v rustup > /dev/null 2>&1; then
        echo "FAIL: rustup not found; cannot prove ${TARGET} is available." >&2
        echo "      Install rustup, or run the cargo command in this file by hand." >&2
        return 1
    fi
    if rustup target list --installed 2> /dev/null | grep -qx "${TARGET}"; then
        return 0
    fi
    echo "note: ${TARGET} not installed; adding it."
    if ! rustup target add "${TARGET}" > /dev/null 2>&1; then
        echo "FAIL: could not install the ${TARGET} std component." >&2
        return 1
    fi
    return 0
}

# Prove the toolchain still rejects the #2310 defect on this target. Without
# this, a green run of the main check could mean "the guard is toothless".
self_test() {
    local rc
    SELF_TEST_DIR="$(mktemp -d)"

    echo "pub const DEFECT: usize = ${DEFECT_LITERAL};" > "${SELF_TEST_DIR}/defect.rs"

    set +e
    rustc --target "${TARGET}" --crate-type lib --emit=metadata \
        -o "${SELF_TEST_DIR}/defect.rmeta" "${SELF_TEST_DIR}/defect.rs" \
        > "${SELF_TEST_DIR}/defect.log" 2>&1
    rc=$?
    set -e

    if [ "${rc}" -eq 0 ]; then
        echo "SELF-TEST FAIL: rustc ACCEPTED ${DEFECT_LITERAL} as a usize on ${TARGET}." >&2
        echo "                This guard can no longer detect the #2310 defect class." >&2
        return 1
    fi
    if ! grep -q "${DEFECT_DIAGNOSTIC}" "${SELF_TEST_DIR}/defect.log"; then
        echo "SELF-TEST FAIL: rustc rejected the defect, but not for the #2310 reason." >&2
        echo "                Expected: ${DEFECT_DIAGNOSTIC}. Got:" >&2
        cat "${SELF_TEST_DIR}/defect.log" >&2
        return 1
    fi
    echo "SELF-TEST PASS: ${TARGET} still rejects the #2310 literal as a usize."
    return 0
}

main_check() {
    local rc
    CHECK_LOG="$(mktemp)"

    echo "Type-checking ${PKG} for ${TARGET} (aprender#2310)..."
    # Append rather than overwrite: CI images bake their own RUSTFLAGS, and
    # clobbering them would silently change what is being compiled.
    export RUSTFLAGS="${RUSTFLAGS:-} --cfg getrandom_backend=\"custom\""

    # Never read the status through a pipe - redirect, then read rc.
    set +e
    cargo check --locked -p "${PKG}" --no-default-features --target "${TARGET}" \
        > "${CHECK_LOG}" 2>&1
    rc=$?
    set -e

    if [ "${rc}" -ne 0 ]; then
        echo "FAIL: ${PKG} does not compile for ${TARGET} (exit ${rc})." >&2
        echo "----- errors -----" >&2
        grep -E "^error" -A 6 "${CHECK_LOG}" >&2 || cat "${CHECK_LOG}" >&2
        echo "------------------" >&2
        return 1
    fi

    echo "PASS: ${PKG} type-checks for ${TARGET}."
    return 0
}

usage() {
    echo "usage: $0 [--self-test]"
    echo "  (no args)    self-test, then type-check ${PKG} for ${TARGET}"
    echo "  --self-test  only prove the toolchain still rejects the #2310 literal"
}

# cargo must run at the workspace root regardless of the caller's cwd.
REPO_ROOT="$(git rev-parse --show-toplevel 2> /dev/null || pwd)"
cd "${REPO_ROOT}" || exit 1

case "${1:-}" in
    --self-test)
        ensure_target || exit 1
        self_test || exit 1
        ;;
    -h | --help)
        usage
        ;;
    "")
        ensure_target || exit 1
        self_test || exit 1
        main_check || exit 1
        ;;
    *)
        echo "unknown argument: ${1}" >&2
        usage >&2
        exit 2
        ;;
esac
