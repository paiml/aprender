#!/usr/bin/env bash
# check_include_files.sh — Verify all include!() referenced files are tracked by git
#
# Prevents the CB-510 bug where .gitignore hid src/models/ part files from git,
# causing crates.io publishes to silently exclude required source files.
#
# Usage: ./scripts/check_include_files.sh
# Exit 0 if all OK, exit 1 if any include!() files are untracked/missing.

set -uo pipefail

errors=0
checked=0

# Find all include!() directives in Rust source files
while IFS=: read -r file line content; do
    # Extract the included filename from include!("filename.rs")
    included=$(echo "$content" | grep -oP 'include!\(\s*"([^"]+)"\s*\)' | sed 's/include!("//;s/")//' || true)
    [ -z "$included" ] && continue

    # Resolve relative to the directory containing the source file
    dir=$(dirname "$file")
    resolved="$dir/$included"

    checked=$((checked + 1))

    # Check file exists
    if [ ! -f "$resolved" ]; then
        echo "MISSING: $resolved (referenced by $file:$line)"
        errors=$((errors + 1))
        continue
    fi

    # Check file is tracked by git (not gitignored)
    if ! git ls-files --error-unmatch "$resolved" >/dev/null 2>&1; then
        # Double-check: is it ignored?
        if git check-ignore -q "$resolved" 2>/dev/null; then
            echo "GITIGNORED: $resolved (referenced by $file:$line)"
        else
            echo "UNTRACKED: $resolved (referenced by $file:$line)"
        fi
        errors=$((errors + 1))
    fi
done < <(grep -rn 'include!(' src/ crates/ --include='*.rs' | grep -v '/target/' | grep -v '#\[')

if [ "$errors" -gt 0 ]; then
    echo ""
    echo "FAIL: $errors include!() files are missing, gitignored, or untracked out of $checked checked"
    echo "Fix: git add the files and check .gitignore / Cargo.toml exclude patterns"
    exit 1
else
    echo "OK: All $checked include!() files are tracked by git"
    exit 0
fi
