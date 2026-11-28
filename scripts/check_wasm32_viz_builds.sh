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
#   --self-test   classifier table, the #2310 literal, the size comparator, the build path
#                 (build_pkg/build_all driven through a stub `cargo` on PATH, so a broken
#                 build_pkg or build_all turns it RED), and a PLANTED `use std::os::unix` in a
#                 COPY of the tracked tree that must break the build for that reason, at the
#                 planted line. The checkout is never edited.

set -euo pipefail

TARGET="wasm32-unknown-unknown"
SIZE_FILE_REL="scripts/wasm32-viz-size.txt"
PLANT_REL="crates/aprender-present-core/src/lib.rs"
# The #2310 literal and the compiler's wording for it (see check_wasm32_core_builds.sh).
DEFECT_LITERAL="6364136223846793005"
DEFECT_DIAGNOSTIC="literal out of range"

SELF_TEST_DIR=""
CHECK_LOG=""
PLANT_DIR=""
STUB_DIR=""
PLANT_LINE=""
RUN_RC=""

# Idempotent: every temp path is cleared once removed, so EXIT after INT/TERM is a no-op.
cleanup() {
    local d
    for d in "${SELF_TEST_DIR}" "${PLANT_DIR}" "${STUB_DIR}"; do
        if [ -n "${d}" ] && [ -d "${d}" ]; then
            rm -rf "${d:?}"
        fi
    done
    if [ -n "${CHECK_LOG}" ] && [ -f "${CHECK_LOG}" ]; then
        rm -f "${CHECK_LOG:?}"
    fi
    SELF_TEST_DIR=""
    PLANT_DIR=""
    STUB_DIR=""
    CHECK_LOG=""
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

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
    if ! rustup target add "${TARGET}" >&2; then
        echo "FAIL: could not install the ${TARGET} std component." >&2
        return 1
    fi
    return 0
}

# One fixed set of flags: inherited host flags (target-cpu, linker args) do not apply to wasm32 and
# would move the recorded sizes. The remaps keep checkout and cargo-home path lengths out of them.
set_wasm_rustflags() {
    unset CARGO_ENCODED_RUSTFLAGS
    export RUSTFLAGS="--cfg getrandom_backend=\"custom\" --remap-path-prefix=${REPO_ROOT}=/src --remap-path-prefix=${CARGO_HOME:-${HOME}/.cargo}=/cargo"
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
# The plant goes into a COPY of the tracked files: the checkout is never edited, so a killed
# run cannot leave the defect behind.
plant_self_test() {
    local rc
    PLANT_DIR="$(mktemp -d)"
    git ls-files -z | tar --null -T - -cf - | tar -xf - -C "${PLANT_DIR}"
    if [ ! -f "${PLANT_DIR}/${PLANT_REL}" ]; then
        echo "SELF-TEST FAIL: ${PLANT_REL} is not in the copied tree." >&2
        return 1
    fi
    CHECK_LOG="$(mktemp)"

    printf '\n// wasm32 self-test plant: must break the build.\nuse std::os::unix::fs::PermissionsExt;\n' \
        >> "${PLANT_DIR}/${PLANT_REL}"
    # The appended `use` is the last line, so the line count is its line number.
    PLANT_LINE="$(wc -l < "${PLANT_DIR}/${PLANT_REL}")"

    set_wasm_rustflags
    set +e
    (cd "${PLANT_DIR}" && CARGO_TARGET_DIR="${PLANT_DIR}/target" \
        cargo check -j 8 --locked -p aprender-present-core --target "${TARGET}") > "${CHECK_LOG}" 2>&1
    rc=$?
    set -e

    if [ "${rc}" -eq 0 ]; then
        echo "SELF-TEST FAIL: a planted std::os::unix import COMPILED for ${TARGET}." >&2
        echo "                This guard can no longer see host-only imports." >&2
        return 1
    fi
    if [ "$(classify_cargo_failure "${CHECK_LOG}")" = 'ENV' ]; then
        report_cargo_env_failure "${CHECK_LOG}" "the planted-import self-test" >&2
        return 1
    fi
    # Both must hold: an unresolved-import error code AND the compiler's location pointer at the
    # planted file and line. Either alone can be matched by an unrelated failure.
    if ! grep -qE '^error\[E04(32|33)\]' "${CHECK_LOG}" \
        || ! grep -qF -- "${PLANT_REL}:${PLANT_LINE}:" "${CHECK_LOG}"; then
        echo "SELF-TEST FAIL: the build failed, but not with E0432/E0433 at ${PLANT_REL}:${PLANT_LINE}." >&2
        grep -E '^error' -A 4 "${CHECK_LOG}" >&2 || cat "${CHECK_LOG}" >&2
        return 1
    fi
    echo "SELF-TEST PASS: a planted std::os::unix import breaks the ${TARGET} build."
    return 0
}

# Release build of one package for wasm32. Usage: build_pkg <pkg> [extra cargo args...]
build_pkg() {
    local pkg="$1" rc
    shift
    # Replace, never accumulate: each call's log is removed before the next one is made.
    if [ -n "${CHECK_LOG}" ]; then
        rm -f "${CHECK_LOG:?}"
    fi
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
    # A .wasm left by an earlier build must not stand in for one this build no longer emits.
    local tdir
    tdir="$(target_directory)"
    rm -f "${tdir:?}/${TARGET:?}/release/aprender_present_lib.wasm" "${tdir:?}/${TARGET:?}/release/trueno_viz.wasm"
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
    if ! meta="$(cargo metadata --format-version 1 --no-deps)"; then
        echo "FAIL: cargo metadata failed; the target directory is unknown." >&2
        return 1
    fi
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
        if ! size="$(gzip -9n -c "${path}" | wc -c)"; then
            echo "FAIL: gzip failed on ${path}" >&2
            return 1
        fi
        echo "${artifact} ${size// /}"
    done
    return 0
}

# Compare measured sizes with the committed file. Usage: compare_sizes <committed> <fresh>
compare_sizes() {
    if [ ! -f "$1" ]; then
        echo "FAIL: ${SIZE_FILE_REL} is missing. Create it with: bash scripts/check_wasm32_viz_builds.sh --write" >&2
        echo "Measured:" >&2
        cat "$2" >&2
        return 1
    fi
    if ! cmp -s "$2" "$1"; then
        echo "FAIL: ${SIZE_FILE_REL} is STALE. Measured:" >&2
        cat "$2" >&2
        echo "If the size change is intended: bash scripts/check_wasm32_viz_builds.sh --write" >&2
        return 1
    fi
    return 0
}

# Missing and stale must fail, equal must pass.
compare_self_test() {
    SELF_TEST_DIR="${SELF_TEST_DIR:-$(mktemp -d)}"
    echo "a.wasm 10" > "${SELF_TEST_DIR}/fresh"
    echo "a.wasm 11" > "${SELF_TEST_DIR}/stale"
    cp "${SELF_TEST_DIR}/fresh" "${SELF_TEST_DIR}/equal"
    if compare_sizes "${SELF_TEST_DIR}/absent" "${SELF_TEST_DIR}/fresh" 2> /dev/null \
        || compare_sizes "${SELF_TEST_DIR}/stale" "${SELF_TEST_DIR}/fresh" 2> /dev/null \
        || ! compare_sizes "${SELF_TEST_DIR}/equal" "${SELF_TEST_DIR}/fresh"; then
        echo "SELF-TEST FAIL: the size comparator does not separate missing/stale/equal." >&2
        return 1
    fi
    echo "SELF-TEST PASS: missing and stale size files fail, an equal one passes."
    return 0
}

# Drive the REAL build_pkg and build_all through a stub `cargo` placed first on PATH. The stub
# never compiles: it answers `cargo metadata` with a target directory and records each
# `cargo build` argv. So this proves the call sites, flags, order and failure propagation of the
# functions the guard actually runs. A build_pkg or build_all that skips, reorders or swallows a
# failure turns it RED. Sets RUN_RC; output goes to ${STUB_DIR}/out.
build_path_self_test() {
    local stub_bin calls tdir fragment n
    STUB_DIR="$(mktemp -d)"
    stub_bin="${STUB_DIR}/bin"
    calls="${STUB_DIR}/calls"
    tdir="${STUB_DIR}/target"
    mkdir -p "${stub_bin}" "${tdir}"
    cat > "${stub_bin}/cargo" << 'STUB'
#!/usr/bin/env bash
# Self-test stub for cargo: compiles nothing, records `cargo build` argv, fails one package on request.
case "${1:-}" in
    metadata)
        printf '{"target_directory":"%s"}\n' "${STUB_TARGET_DIR}"
        exit 0
        ;;
    build)
        printf '%s\n' "$*" >> "${STUB_CALLS}"
        case " $* " in
            *" -p ${STUB_FAIL_PKG:-__none__} "*)
                echo "error[E0432]: unresolved import (stub failure for the build-path self-test)"
                echo "error: could not compile \`${STUB_FAIL_PKG}\` (lib) due to 1 previous error"
                exit 101
                ;;
        esac
        exit 0
        ;;
    *)
        echo "stub cargo: unexpected argv: $*" >&2
        exit 99
        ;;
esac
STUB
    chmod +x "${stub_bin}/cargo"

    # Usage: run_build_all <package to fail, or empty>. Runs build_all in a subshell with the
    # stub first on PATH, so PATH and the STUB_* variables never leak into this shell.
    run_build_all() {
        : > "${calls}"
        set +e
        (
            export PATH="${stub_bin}:${PATH}" STUB_CALLS="${calls}" STUB_TARGET_DIR="${tdir}" \
                STUB_FAIL_PKG="$1"
            build_all
        ) > "${STUB_DIR}/out" 2>&1
        RUN_RC=$?
        set -e
    }

    run_build_all ""
    if [ "${RUN_RC}" -ne 0 ]; then
        echo "SELF-TEST FAIL: build_all reported RED with a green stub cargo (exit ${RUN_RC})." >&2
        cat "${STUB_DIR}/out" >&2
        return 1
    fi
    n="$(wc -l < "${calls}")"
    if [ "${n}" -ne 6 ]; then
        echo "SELF-TEST FAIL: build_all ran ${n} cargo builds, expected 6 (one per wasm32 package)." >&2
        cat "${calls}" >&2
        return 1
    fi
    # build_pkg puts the per-package extra args AFTER --target, so these are ordered patterns.
    for fragment in \
        "-p aprender-viz .*--features geo" \
        "-p aprender-present-core " \
        "-p aprender-present-layout " \
        "-p aprender-present-widgets " \
        "-p aprender-present-lib " \
        "-p aprender-db .*--no-default-features --features wasm"; do
        if ! grep -qE -- "${fragment}" "${calls}"; then
            echo "SELF-TEST FAIL: build_all never ran cargo build matching '${fragment}'." >&2
            return 1
        fi
    done
    n="$(grep -cF -- "--target ${TARGET}" "${calls}" || true)"
    if [ "${n}" -ne 6 ]; then
        echo "SELF-TEST FAIL: ${n} of 6 builds carried --target ${TARGET}." >&2
        return 1
    fi

    # A failing package must turn the whole run RED and stop before the packages after it.
    run_build_all "aprender-present-layout"
    if [ "${RUN_RC}" -eq 0 ]; then
        echo "SELF-TEST FAIL: build_all reported GREEN although cargo failed for aprender-present-layout." >&2
        return 1
    fi
    n="$(wc -l < "${calls}")"
    if [ "${n}" -ne 3 ] || ! grep -qF -- "FAIL: aprender-present-layout " "${STUB_DIR}/out"; then
        echo "SELF-TEST FAIL: the failing package was not named, or build_all did not stop there (${n} builds)." >&2
        cat "${STUB_DIR}/out" >&2
        return 1
    fi
    echo "SELF-TEST PASS: build_all drives the six wasm32 builds through build_pkg, and goes RED at the first failing package."
    return 0
}

usage() {
    echo "usage: $0 [--self-test | --write | -h]"
    echo "  (no args)    literal self-test, build every wasm32 package, compare .wasm sizes to ${SIZE_FILE_REL}"
    echo "  --write      as above, rewrite ${SIZE_FILE_REL}"
    echo "  --self-test  classifier table, #2310 literal, build path via a stub cargo, and a planted std::os::unix import that must fail"
}

case "${1:-}" in
    --self-test)
        cargo_classify_selftest
        ensure_target
        literal_self_test
        compare_self_test
        build_path_self_test
        plant_self_test
        ;;
    --write | "")
        cargo_classify_selftest --quiet
        ensure_target
        literal_self_test
        build_all
        SELF_TEST_DIR="${SELF_TEST_DIR:-$(mktemp -d)}"
        fresh="${SELF_TEST_DIR}/sizes"
        render_sizes > "${fresh}"
        if [ "${1:-}" = "--write" ]; then
            cp "${fresh}" "${REPO_ROOT}/${SIZE_FILE_REL}"
            echo "WROTE: ${SIZE_FILE_REL}"
            cat "${REPO_ROOT}/${SIZE_FILE_REL}"
            exit 0
        fi
        compare_sizes "${REPO_ROOT}/${SIZE_FILE_REL}" "${fresh}"
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
