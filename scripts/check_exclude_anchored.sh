#!/usr/bin/env bash
#
# check_exclude_anchored.sh — CB-510 class guard, generalised.
#
# A `exclude = [...]` entry in Cargo.toml is a gitignore-style pattern. A bare
# directory name such as `"models/"` or `"tests/"` is NOT anchored to the package
# root: it matches at EVERY depth, so it silently swallows `src/models/` and
# `src/api/tests/` on the way to crates.io.
#
# This has now shipped twice:
#   * CB-510  — `"models/"` hid `src/models/` from git and from the package.
#   * v0.63.0 — `"tests/"` dropped 443 files from `aprender-serve` and 117 broken
#               `mod tests;` declarations from `aprender-train`. Both published
#               crates cannot compile their own test suites.
#
# The fix in both cases is a leading slash: `"/models/"`, `"/tests/"`.
#
# This guard fails when a non-anchored directory pattern in any `exclude` list
# matches a directory that actually exists under that crate's `src/`.
#
# Usage: bash scripts/check_exclude_anchored.sh [repo_root]

set -euo pipefail

repo_root="${1:-$(git rev-parse --show-toplevel)}"
cd "$repo_root"

violations=0
checked=0

# Emit the quoted entries of the `exclude = [ ... ]` array of a Cargo.toml, one
# per line, with the surrounding quotes stripped. Comment lines are skipped.
extract_exclude_patterns() {
    manifest="$1"
    sed -n '/^exclude[[:space:]]*=[[:space:]]*\[/,/\]/p' "$manifest" \
        | sed 's/#.*//' \
        | grep -o '"[^"]*"' \
        | tr -d '"'
}

while IFS= read -r manifest; do
    crate_dir="$(dirname "$manifest")"
    src_dir="$crate_dir/src"
    [ -d "$src_dir" ] || continue

    checked=$((checked + 1))

    while IFS= read -r pattern; do
        # Anchored (`/tests/`), glob (`*.rs`, `**/x`), and non-directory
        # patterns are all fine — only a bare `name/` is ambiguous.
        case "$pattern" in
            /*) continue ;;
            *\**) continue ;;
            */) : ;;
            *) continue ;;
        esac

        dir_name="${pattern%/}"
        # A pattern containing a slash in the middle (`contracts/foo/`) is
        # already specific enough to be unambiguous in practice.
        case "$dir_name" in
            */*) continue ;;
        esac

        matches="$(find "$src_dir" -type d -name "$dir_name" -print 2>/dev/null || true)"
        if [ -n "$matches" ]; then
            count="$(printf '%s\n' "$matches" | wc -l | tr -d ' ')"
            printf 'FAIL %s\n' "$manifest"
            printf '     exclude pattern "%s" is not root-anchored and matches %s director(y|ies) under %s:\n' \
                "$pattern" "$count" "$src_dir"
            printf '%s\n' "$matches" | sed 's/^/       /' | head -5
            if [ "$count" -gt 5 ]; then
                printf '       ... and %s more\n' "$((count - 5))"
            fi
            printf '     fix: write it as "/%s"\n\n' "$pattern"
            violations=$((violations + 1))
        fi
    done < <(extract_exclude_patterns "$manifest")
done < <(git ls-files '*Cargo.toml' | sort)

if [ "$violations" -gt 0 ]; then
    printf 'check_exclude_anchored: %s violation(s) across %s manifest(s) with a src/ tree\n' \
        "$violations" "$checked" >&2
    printf 'A non-anchored exclude pattern removes source from the published crate.\n' >&2
    exit 1
fi

printf 'check_exclude_anchored: OK - %s manifest(s) checked, no non-anchored exclude swallows a src/ directory\n' "$checked"
