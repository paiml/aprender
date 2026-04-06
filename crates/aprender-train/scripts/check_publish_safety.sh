#!/usr/bin/env bash
# check_publish_safety.sh — Prevent broken crates.io publishes (PMAT-517)
#
# Checks:
#   1. All contract_pre_*/contract_post_* macros used in source are DEFINED
#   2. No [patch.crates-io] in Cargo.toml (must be in .cargo/config.toml only)
#   3. No path-only dependencies without version fallback
#
# Usage: bash scripts/check_publish_safety.sh
# Exit 0 if all OK, exit 1 if any checks fail.

set -uo pipefail

errors=0
checked=0

echo "Entrenar publish safety gate (PMAT-517)..."

# Check 1: All contract macros used in source are defined
echo -n "  Contract macro completeness... "
checked=$((checked + 1))

# Write used and defined macros to temp files for comparison
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# Find all contract macro INVOCATIONS in source (contract_pre_foo! or contract_post_bar!)
grep -roh 'contract_pre_[a-z_]*!\|contract_post_[a-z_]*!' src/ --include='*.rs' 2>/dev/null \
    | sed 's/!$//' | sort -u > "$tmpdir/used.txt"

# Find all contract macro DEFINITIONS (macro_rules! contract_pre_foo)
grep -roh 'macro_rules! *contract_pre_[a-z_]*\|macro_rules! *contract_post_[a-z_]*' src/ --include='*.rs' 2>/dev/null \
    | sed 's/macro_rules! *//' | sort -u > "$tmpdir/defined.txt"

# Find macros used but not defined
missing=$(comm -23 "$tmpdir/used.txt" "$tmpdir/defined.txt")

if [ -n "$missing" ]; then
    echo "FAIL"
    echo "$missing" | while IFS= read -r macro; do
        echo "  MISSING: $macro (used but not defined)"
    done
    echo "Fix: Add fallback stub to src/lib.rs"
    errors=$((errors + 1))
else
    count=$(wc -l < "$tmpdir/used.txt")
    echo "OK ($count macros, all defined)"
fi

# Check 2: No [patch.crates-io] in Cargo.toml
echo -n "  No patches in Cargo.toml... "
checked=$((checked + 1))
if grep -q 'patch.crates-io' Cargo.toml 2>/dev/null; then
    echo "FAIL"
    echo "FAIL: [patch.crates-io] found in Cargo.toml"
    echo "Fix: Move to .cargo/config.toml (excluded from cargo publish)"
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 3: No path-only sibling deps without version
echo -n "  No version-less path deps... "
checked=$((checked + 1))
bad_deps=$(grep -n 'path *= *"\.\.' Cargo.toml 2>/dev/null | grep -v 'version' || true)
if [ -n "$bad_deps" ]; then
    echo "WARN"
    echo "WARN: Path dependencies without version fallback:"
    echo "$bad_deps"
else
    echo "OK"
fi

# Summary
echo ""
if [ "$errors" -gt 0 ]; then
    echo "FAIL: $errors publish safety checks failed out of $checked"
    exit 1
else
    echo "OK: All $checked publish safety checks passed"
fi
