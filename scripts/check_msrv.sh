#!/usr/bin/env bash
# check_msrv.sh — verify the workspace actually builds on its DECLARED MSRV.
#
# PMAT-MSRV-GATE-001 (Fable rank-14). The declared `rust-version` had silently
# drifted: Cargo.toml claimed 1.89 while `pmcp` and `wasmtime` transitively require
# rustc 1.91, and rust-toolchain.toml pins 1.93 — so NOTHING ever verified the claim.
# This script is the gate: it reads the declared MSRV and compiles the whole
# workspace on exactly that toolchain, failing loudly on any drift.
#
# RED-turning mutation: set rust-version back to "1.89" and re-run — this fails with
#   `error: rustc 1.89.0 is not supported by pmcp@2.9.0 (requires 1.91)`.
#
# Usage:  bash scripts/check_msrv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MSRV="$(grep -m1 '^rust-version' Cargo.toml | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?')"
[ -n "$MSRV" ] || { echo "check_msrv: could not read rust-version from Cargo.toml" >&2; exit 2; }
echo "Declared MSRV (Cargo.toml rust-version): $MSRV"

if ! rustup toolchain list 2>/dev/null | grep -q "^${MSRV}"; then
  echo "Installing rust ${MSRV} toolchain ..."
  rustup toolchain install "${MSRV}" --profile minimal --no-self-update
fi

echo "→ cargo +${MSRV} check --workspace"
if cargo "+${MSRV}" check --workspace; then
  echo "✓ workspace builds on its declared MSRV ${MSRV}"
else
  echo "✗ MSRV DRIFT: workspace does NOT build on declared rust-version ${MSRV}." >&2
  echo "  Fix: bump rust-version in Cargo.toml to the true minimum, or pin the" >&2
  echo "  offending dependency to a version that supports ${MSRV}." >&2
  exit 1
fi
