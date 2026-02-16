#!/usr/bin/env bash
# check_package_includes.sh — Verify cargo package includes all include!() files
#
# Second line of defense: even if git tracks a file, Cargo.toml `exclude`
# patterns can strip it from the published crate. This checks the actual
# package manifest.
#
# Usage: ./scripts/check_package_includes.sh
# Exit 0 if all OK, exit 1 if any include!() files would be excluded from the crate.

set -uo pipefail

errors=0
checked=0

# Get the package file list (requires --allow-dirty for uncommitted changes)
package_list=$(cargo package -p aprender --list --allow-dirty 2>/dev/null)
if [ -z "$package_list" ]; then
    echo "ERROR: cargo package --list failed"
    exit 1
fi

# Find all include!() directives in src/ (the aprender crate)
while IFS=: read -r file line content; do
    included=$(echo "$content" | grep -oP 'include!\(\s*"([^"]+)"\s*\)' | sed 's/include!("//;s/")//' || true)
    [ -z "$included" ] && continue

    dir=$(dirname "$file")
    resolved="$dir/$included"
    checked=$((checked + 1))

    # Check if the resolved path appears in cargo package --list
    if ! echo "$package_list" | grep -qF "$resolved"; then
        echo "EXCLUDED: $resolved (referenced by $file:$line)"
        echo "  This file would be MISSING from the published crate!"
        errors=$((errors + 1))
    fi
done < <(grep -rn 'include!(' src/ --include='*.rs' | grep -v '/target/')

if [ "$errors" -gt 0 ]; then
    echo ""
    echo "FAIL: $errors include!() files would be excluded from crates.io package out of $checked checked"
    echo "Fix: check Cargo.toml [package] exclude patterns"
    exit 1
else
    echo "OK: All $checked include!() files are included in cargo package"
    exit 0
fi
