#!/usr/bin/env bash
# check_wasm32_viz_builds.sh - the viz / present / db surface must build for wasm32, and the
# release .wasm artifacts must stay the size committed in scripts/wasm32-viz-size.txt.
#
# APR-MAP-001 row MAP-2 (PMAT-7510). Sibling of check_wasm32_core_builds.sh (aprender#2310),
# which only type-checks aprender-core. This guard runs RELEASE builds of the packages a
# browser page is made from, then pins what they weigh.
#
# Packages built for wasm32-unknown-unknown (release profile):
#   aprender-viz            --features geo
#   aprender-present-core   (default features)
#   aprender-present-layout (default features)
#   aprender-present-widgets (default features)
#   aprender-present-lib    (default features; its cdylib is the browser artifact)
#   aprender-db             --no-default-features --features wasm
#     aprender-db's default feature set (tokio, rayon, parquet-io, server) is native-only; its
#     wasm32 surface is the `wasm` feature, which gates `pub mod wasm` in src/lib.rs.
#
# Size pins. Only the cdylib crates produce a .wasm file: aprender_viz (lib name trueno_viz)
# and aprender_present_lib. Each artifact is measured as `gzip -9n` byte count. `-n` drops the
# gzip header timestamp and file name, so the count depends only on the artifact bytes.
#
# Exit 0 = every package builds for wasm32 AND (default mode) the sizes match the committed file.
# Exit 1 = a build failed, the toolchain could not be proven, or the size file is missing/stale.
#          Fail closed: a guard that cannot run must never be mistaken for a guard that passed.
#
# Modes:
#   (no args)     self-test the literal + classifier, build every package, compare sizes.
#   --write       as above, but rewrite scripts/wasm32-viz-size.txt instead of comparing.
#   --self-test   classifier table, the #2310 literal, and a PLANTED `use std::os::unix` in
#                 crates/aprender-present-core/src/lib.rs that must break the build for that
#                 reason. The planted line is removed by a trap, on every exit path.

set -euo pipefail

TARGET="wasm32-unknown-unknown"
SIZE_FILE_REL="scripts/wasm32-viz-size.txt"
PLANT_REL="crates/aprender-present-core/src/lib.rs"
# The #2310 literal and the compiler's wording for it (see check_wasm32_core_builds.sh).
DEFECT_LITERAL="6364136223846793005"
DEFECT_DIAGNOSTIC="literal out of range"

SELF_TEST_DIR=""
CHECK_LOG=""
PLANT_BACKUP=""
PLANT_TARGET=""

cleanup() {
    # Restore the planted file first: a leftover plant would turn the tree red for every later run.
    if [ -n "${PLANT_BACKUP}" ] && [ -f "${PLANT_BACKUP}" ] && [ -n "${PLANT_TARGET}" ]; then
        cp -p "${PLANT_BACKUP}" "${PLANT_TARGET}" || true
        rm -f "${PLANT_BACKUP}"
    fi
    if [ -n "${SELF_TEST_DIR}" ] && [ -d "${SELF_TEST_DIR}" ]; then
        rm -rf "${SELF_TEST_DIR}"
    fi
    if [ -n "${CHECK_LOG}" ] && [ -f "${CHECK_LOG}" ]; then
        rm -f "${CHECK_LOG}"
    fi
}
trap cleanup EXIT

# cargo must run at the workspace root regardless of the caller's cwd.
REPO_ROOT="$(git rev-parse --show-toplevel 2> /dev/null || pwd)"
cd "${REPO_ROOT}" || exit 1

. "${REPO_ROOT}/scripts/cargo_classify.sh" || exit 1

ensure_target() {
    if ! command -v rustup > /dev/null 2>&1; then
        echo "FAIL: rustup not found; cannot prove ${TARGET} is available." >&2
        echo "      Install rustup, or run the cargo commands in this file by hand." >&2
        return 1
    fi
    if grep -qx "${TARGET}" <<< "$(rustup target list --installed 2> /dev/null)" ; then
        return 0
    fi
    echo "note: ${TARGET} not installed; adding it."
    if ! rustup target add "${TARGET}" > /dev/null 2>&1; then
        echo "FAIL: could not install the ${TARGET} std component." >&2
        return 1
    fi
    return 0
}

# Same RUSTFLAGS handling as check_wasm32_core_builds.sh: appended once, never clobbering the
# CI image's own flags, and never appended twice (that would rebuild every shared crate).
set_wasm_rustflags() {
    case "${RUSTFLAGS:-}" in
        *getrandom_backend*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:-} --cfg getrandom_backend=\"custom\"" ;;
    esac
}

# Prove the toolchain still rejects the #2310 defect on this target.
literal_self_test() {
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
        return 1
    fi
    if ! grep -q "${DEFECT_DIAGNOSTIC}" "${SELF_TEST_DIR}/defect.log"; then
        echo "SELF-TEST FAIL: rustc rejected the literal, but not for the #2310 reason." >&2
        cat "${SELF_TEST_DIR}/defect.log" >&2
        return 1
    fi
    echo "SELF-TEST PASS: ${TARGET} still rejects the #2310 literal as a usize."
    return 0
}

# Plant `use std::os::unix::fs::PermissionsExt;` into a present crate and require the wasm32
# check to FAIL for that reason. std::os::unix does not exist on wasm32-unknown-unknown, so
# a green result here would mean the guard can no longer see a host-only import.
plant_self_test() {
    local rc log_ok
    PLANT_TARGET="${REPO_ROOT}/${PLANT_REL}"
    PLANT_BACKUP="$(mktemp)"
    cp -p "${PLANT_TARGET}" "${PLANT_BACKUP}"
    CHECK_LOG="$(mktemp)"

    printf '\n// wasm32 self-test plant: must break the build.\nuse std::os::unix::fs::PermissionsExt;\n' \
        >> "${PLANT_TARGET}"

    set_wasm_rustflags
    set +e
    cargo check -j 8 --locked -p aprender-present-core --target "${TARGET}" > "${CHECK_LOG}" 2>&1
    rc=$?
    set -e

    # Restore before reporting, so the tree is clean whatever the verdict below.
    cp -p "${PLANT_BACKUP}" "${PLANT_TARGET}"
    rm -f "${PLANT_BACKUP}"
    PLANT_BACKUP=""
    PLANT_TARGET=""

    if [ "${rc}" -eq 0 ]; then
        echo "SELF-TEST FAIL: a planted std::os::unix import COMPILED for ${TARGET}." >&2
        echo "                This guard can no longer see host-only imports." >&2
        return 1
    fi
    if [ "$(classify_cargo_failure "${CHECK_LOG}")" = 'ENV' ]; then
        report_cargo_env_failure "${CHECK_LOG}" "the planted-import self-test" >&2
        return 1
    fi
    log_ok=0
    grep -qE 'E0432|std::os::unix|`unix` in `os`' "${CHECK_LOG}" && log_ok=1
    if [ "${log_ok}" -ne 1 ]; then
        echo "SELF-TEST FAIL: the build failed, but not on the planted std::os::unix import." >&2
        grep -E '^error' -A 4 "${CHECK_LOG}" >&2 || cat "${CHECK_LOG}" >&2
        return 1
    fi
    echo "SELF-TEST PASS: a planted std::os::unix import breaks the ${TARGET} build (E0432)."
    return 0
}

# Release build of one package for wasm32. Usage: build_pkg <pkg> [extra cargo args...]
build_pkg() {
    local pkg="$1" rc
    shift
    CHECK_LOG="$(mktemp)"

    set_wasm_rustflags
    # Never read the status through a pipe: redirect, then read rc.
    set +e
    # -j 8 is an argv word, not an env var: the cargo-memcap wrapper on the build hosts admits
    # only `cargo build -j 8` (see the LD1 form in its refusal), and refuses the bare command.
    cargo build -j 8 --locked --release -p "${pkg}" --target "${TARGET}" "$@" > "${CHECK_LOG}" 2>&1
    rc=$?
    set -e

    if [ "${rc}" -ne 0 ] && [ "$(classify_cargo_failure "${CHECK_LOG}")" = 'ENV' ]; then
        report_cargo_env_failure "${CHECK_LOG}" "whether ${pkg} builds for ${TARGET}" >&2
        return 1
    fi
    if [ "${rc}" -ne 0 ]; then
        echo "FAIL: ${pkg} $* does not build for ${TARGET} (exit ${rc})." >&2
        echo "----- errors -----" >&2
        grep -E '^error' -A 6 "${CHECK_LOG}" >&2 || cat "${CHECK_LOG}" >&2
        echo "------------------" >&2
        return 1
    fi
    echo "PASS: ${pkg} $* builds for ${TARGET} (release)."
    return 0
}

build_all() {
    build_pkg aprender-viz --features geo || return 1
    build_pkg aprender-present-core || return 1
    build_pkg aprender-present-layout || return 1
    build_pkg aprender-present-widgets || return 1
    build_pkg aprender-present-lib || return 1
    build_pkg aprender-db --no-default-features --features wasm || return 1
    return 0
}

# The directory cargo actually writes to. The gitignored .cargo/config.toml and CARGO_TARGET_DIR
# can both move it, so it is asked of cargo rather than guessed.
target_directory() {
    local meta
    meta="$(cargo metadata --format-version 1 --no-deps 2> /dev/null)" || return 1
    grep -o '"target_directory":"[^"]*"' <<< "${meta}" | head -n 1 | cut -d'"' -f4
}

# One line per .wasm artifact: "<file> <gzip -9n bytes>". Fail closed on a missing artifact.
render_sizes() {
    local tdir artifact path size
    tdir="$(target_directory)" || return 1
    if [ -z "${tdir}" ]; then
        echo "FAIL: cargo metadata reported no target_directory." >&2
        return 1
    fi
    echo "# scripts/wasm32-viz-size.txt - gzip -9n byte count of each release .wasm artifact."
    echo "# Regenerate after an intended size change: bash scripts/check_wasm32_viz_builds.sh --write"
    for artifact in aprender_present_lib.wasm trueno_viz.wasm; do
        path="${tdir}/${TARGET}/release/${artifact}"
        if [ ! -f "${path}" ]; then
            echo "FAIL: expected artifact missing: ${path}" >&2
            return 1
        fi
        size="$(gzip -9n -c "${path}" | wc -c)"
        echo "${artifact} ${size// /}"
    done
    return 0
}

usage() {
    echo "usage: $0 [--self-test | --write | -h]"
    echo "  (no args)    literal self-test, build every wasm32 package, compare .wasm sizes to ${SIZE_FILE_REL}"
    echo "  --write      as above, rewrite ${SIZE_FILE_REL}"
    echo "  --self-test  classifier table, #2310 literal, and a planted std::os::unix import that must fail"
}

case "${1:-}" in
    --self-test)
        cargo_classify_selftest || exit 1
        ensure_target || exit 1
        literal_self_test || exit 1
        plant_self_test || exit 1
        ;;
    --write | "")
        cargo_classify_selftest --quiet || exit 1
        ensure_target || exit 1
        literal_self_test || exit 1
        build_all || exit 1
        fresh="$(mktemp)"
        if ! render_sizes > "${fresh}"; then
            rm -f "${fresh}"
            exit 1
        fi
        if [ "${1:-}" = "--write" ]; then
            cp "${fresh}" "${REPO_ROOT}/${SIZE_FILE_REL}"
            rm -f "${fresh}"
            echo "WROTE: ${SIZE_FILE_REL}"
            cat "${REPO_ROOT}/${SIZE_FILE_REL}"
            exit 0
        fi
        if [ ! -f "${REPO_ROOT}/${SIZE_FILE_REL}" ]; then
            echo "FAIL: ${SIZE_FILE_REL} is missing. Create it with: bash scripts/check_wasm32_viz_builds.sh --write" >&2
            rm -f "${fresh}"
            exit 1
        fi
        if ! cmp -s "${fresh}" "${REPO_ROOT}/${SIZE_FILE_REL}"; then
            echo "FAIL: ${SIZE_FILE_REL} is STALE. Measured:" >&2
            cat "${fresh}" >&2
            echo "If the size change is intended: bash scripts/check_wasm32_viz_builds.sh --write" >&2
            rm -f "${fresh}"
            exit 1
        fi
        rm -f "${fresh}"
        echo "PASS: wasm32 viz/present/db builds, and every .wasm size matches ${SIZE_FILE_REL}."
        ;;
    -h | --help)
        usage
        ;;
    *)
        echo "unknown argument: ${1}" >&2
        usage >&2
        exit 2
        ;;
esac
