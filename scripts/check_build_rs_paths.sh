#!/usr/bin/env bash
# check_build_rs_paths.sh — Poka-Yoke: flag build.rs files that PANIC when
# a path outside the crate root is absent. Those break `cargo install`.
#
# Root cause this guards against (v0.31.1 release blocker):
#   crates/aprender-mcp/build.rs resolved `CARGO_MANIFEST_DIR/../../contracts/…`
#   and called `.unwrap_or_else(|e| panic!(...))` when the file was missing.
#   That file lives in the monorepo tree but NOT in the published tarball
#   (outside the package root). `cargo install aprender@0.31.1` panicked at
#   build time for every external user. v0.31.1 was yanked.
#
# Acceptable patterns (NOT flagged):
#   1. build.rs has `ALLOW_ESCAPE: <reason>` comment
#   2. build.rs has `.exists()` + `return`/`Ok(())` graceful fallback
#      (crates/aprender-core/build.rs does this for provable-contracts/)
#   3. Path join of `".."` is inside a `#[cfg(test)]` block
#
# Failure pattern (FLAGGED):
#   - build.rs joins `".."` AND calls `panic!`/`unwrap_or_else(|…| panic!)`
#     AND has no `.exists()` guard before the read.
#
# Complementary gate: check_package_verify.sh runs `cargo package` per crate
# which actually unpacks the tarball and compiles — the dynamic equivalent.
#
# Usage: bash scripts/check_build_rs_paths.sh
# Exit 0 clean, exit 1 on any finding.

set -uo pipefail

errors=0
checked=0
suspects=0

echo "Build.rs path safety gate (static)..."

while IFS= read -r build_rs; do
    checked=$((checked + 1))

    # Skip if no `..` path escape at all.
    if ! grep -qE '"\.\."' "$build_rs"; then
        continue
    fi
    suspects=$((suspects + 1))

    # Acceptable: explicit allow-escape annotation.
    if grep -q 'ALLOW_ESCAPE' "$build_rs"; then
        continue
    fi
    # Acceptable: graceful fallback — checks `.exists()` before reading.
    # Pattern covers both `if path.exists()` and `if !path.exists() { return; }`.
    if grep -qE '\.exists\(\)' "$build_rs"; then
        continue
    fi

    # Hard-coded absolute paths are always wrong.
    if grep -qE '"/(home|tmp|usr|var|opt|mnt|root)/' "$build_rs"; then
        echo "  FAIL: $build_rs — hard-coded absolute path (portability bug)"
        errors=$((errors + 1))
        continue
    fi

    # Suspect: has `..` join AND panics on read failure, no .exists() guard.
    if grep -qE 'panic!|unwrap_or_else\(\|.*\|\s*panic' "$build_rs"; then
        echo "  FAIL: $build_rs"
        echo "    joins \"..\" onto CARGO_MANIFEST_DIR AND panics on missing file;"
        echo "    this class broke \`cargo install\` in v0.31.1 (aprender-mcp)."
        echo "    Fix: either"
        echo "      (a) copy the file inside the crate + update build.rs path, OR"
        echo "      (b) add a \`.exists()\` guard with graceful fallback, OR"
        echo "      (c) annotate with '// ALLOW_ESCAPE: <reason>' if only used in dev."
        errors=$((errors + 1))
    fi
done < <(git ls-files '**/build.rs' 'build.rs')

echo ""
if [ "$errors" -eq 0 ]; then
    echo "OK: $checked build.rs files checked ($suspects with \`..\` joins all have graceful fallbacks)."
    exit 0
else
    echo "FAIL: $errors/$suspects suspect build.rs files WILL break \`cargo install\`."
    echo ""
    echo "v0.31.1 was yanked for exactly this class of bug. Do not ship without fixing."
    exit 1
fi
