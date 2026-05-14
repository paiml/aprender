#!/bin/bash
# GH-696: Cross-compile `apr` for Jetson (aarch64, Ubuntu 22.04, GLIBC 2.35).
#
# Builds apr-cli targeting aarch64-unknown-linux-gnu with the GLIBC ABI
# pinned to 2.35 via cargo-zigbuild's `<triple>.<glibc_version>` syntax.
# zig 0.13.0 supplies the cross-compile sysroot with older-GLIBC stubs.
#
# Requirements (build-time, all on the build host):
#   - rustup target add aarch64-unknown-linux-gnu
#   - cargo install cargo-zigbuild
#   - zig 0.13.0+ on PATH
#
# Output: target/aarch64-unknown-linux-gnu/release/apr
#
# Verified 2026-05-14 — runs cleanly on Jetson (Ubuntu 22.04.5 LTS, GLIBC 2.35,
# Linux 5.15.185-tegra) producing `apr 0.33.0`.
#
# Per `evidence/gh-696-jetson-aarch64-2026-05-14/findings.json`, the resulting
# binary uses GLIBC symbols up to 2.29 only (6 minor versions of headroom on
# the Jetson's 2.35).
set -euo pipefail

readonly TARGET="aarch64-unknown-linux-gnu.2.35"
readonly OUT_DIR="${CARGO_TARGET_DIR:-target}/aarch64-unknown-linux-gnu/release"

echo "[GH-696] Cross-compile apr → $TARGET"
echo "[GH-696] Excludes feature gates that pull host-platform deps."
echo

# --no-default-features --features inference excludes:
#   hf-hub (network)            — keep portable
#   safetensors-compare         — fine to omit on edge
#   training, training-gpu      — out of scope for Jetson inference
#   visualization (renacer)     — host-only
#   zram                        — kernel-feature-gated
# Add features here only after confirming they cross-compile cleanly.
cargo zigbuild \
    -p apr-cli \
    --release \
    --bin apr \
    --target "$TARGET" \
    --no-default-features \
    --features inference

echo
echo "[GH-696] Build complete: $OUT_DIR/apr"
echo "[GH-696] Quick check:"
file "$OUT_DIR/apr" | head -1
echo "GLIBC versions linked:"
objdump -p "$OUT_DIR/apr" 2>/dev/null | grep -oE "GLIBC_[0-9.]+" | sort -uV | tail -5

echo
echo "[GH-696] Ship to Jetson:"
echo "    scp $OUT_DIR/apr jetson:/usr/local/bin/apr"
echo "    ssh jetson 'apr --version'"
