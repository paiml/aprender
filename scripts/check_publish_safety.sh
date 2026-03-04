#!/usr/bin/env bash
# check_publish_safety.sh — Prevent P0 publish regressions from shipping to crates.io
#
# Bug classes caught:
#   1. Tracked symlinks (mode 120000) — broke cargo install for all external users
#   2. with_file_name() anti-pattern — misses hash-prefixed companion files
#   3. .pmat dev tool artifacts — shipped DB files and caches to crates.io
#   4. Large binary files in packages — bloated crate downloads
#   5. Hardcoded local paths — break on other machines
#
# Refs: PMAT-SQI (symlinks), GAP-UX-002 (companion lookup), CB-510 (gitignore)
# Contract: contracts/publish-safety-v1.yaml
#
# Usage: bash scripts/check_publish_safety.sh
# Exit 0 if all OK, exit 1 if any checks fail.

set -uo pipefail

errors=0
checked=0

echo "Publish safety gate..."

# Check 1: No tracked symlinks (P0: symlinks to build dirs broke all users)
echo -n "  Symlink check... "
symlinks=$(git ls-files -s | grep "^120000" || true)
checked=$((checked + 1))
if [ -n "$symlinks" ]; then
    echo "FAIL"
    echo "FAIL: Tracked symlinks found in git:"
    echo "$symlinks"
    echo "Fix: git rm --cached <symlink-path>"
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 2: No .cargo/config.toml in any workspace package
echo -n "  Cargo config leak check... "
checked=$((checked + 1))
config_leak=0
for pkg in aprender apr-cli; do
    if cargo package -p "$pkg" --list 2>/dev/null | grep -q '\.cargo/config'; then
        if [ "$config_leak" -eq 0 ]; then
            echo "FAIL"
        fi
        echo "FAIL: .cargo/config.toml found in $pkg package"
        echo "Fix: add .cargo/config.toml to Cargo.toml exclude and run git rm --cached .cargo/config.toml"
        config_leak=1
    fi
done
if [ "$config_leak" -gt 0 ]; then
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 3: No with_file_name("tokenizer.json"|"config.json") anti-pattern in apr-cli
echo -n "  Companion lookup check... "
checked=$((checked + 1))
bad_lookups=$(grep -rn 'with_file_name\s*(.*"tokenizer\.json"' crates/apr-cli/src/ 2>/dev/null || true)
bad_lookups2=$(grep -rn 'with_file_name\s*(.*"config\.json"' crates/apr-cli/src/ 2>/dev/null || true)
bad_all="${bad_lookups}${bad_lookups2}"
if [ -n "$bad_all" ]; then
    echo "FAIL"
    echo "FAIL: with_file_name() used for companion file lookup (use find_sibling_file instead):"
    echo "$bad_all"
    echo "Fix: replace path.with_file_name(\"tokenizer.json\") with find_sibling_file(path, \"tokenizer.json\")"
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 4: find_sibling_file is actually used (sanity check — regression if all removed)
echo -n "  find_sibling_file usage check... "
checked=$((checked + 1))
sibling_count=$(grep -rn 'find_sibling_file' crates/apr-cli/src/ 2>/dev/null | wc -l)
if [ "$sibling_count" -lt 1 ]; then
    echo "FAIL"
    echo "FAIL: No find_sibling_file() calls found in apr-cli (expected at least 1)"
    echo "This suggests companion file lookup was removed or broken"
    errors=$((errors + 1))
else
    echo "OK ($sibling_count usages)"
fi

# Check 5: No .pmat dev tool artifacts tracked by git (CB-510 class)
echo -n "  Dev tool artifacts check... "
checked=$((checked + 1))
pmat_tracked=$(git ls-files | grep '\.pmat/' || true)
if [ -n "$pmat_tracked" ]; then
    echo "FAIL"
    count=$(echo "$pmat_tracked" | wc -l)
    echo "FAIL: $count .pmat/ files tracked by git (these ship to crates.io):"
    echo "$pmat_tracked" | head -10
    echo "Fix: git rm --cached <paths> and ensure **/.pmat/ is in .gitignore"
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 6: No large binary files (>1MB) in publishable packages
echo -n "  Package size check... "
checked=$((checked + 1))
large_files=""
for pkg in aprender apr-cli; do
    # Check for binary/model files that shouldn't be in packages
    binaries=$(cargo package -p "$pkg" --list --allow-dirty 2>/dev/null \
        | grep -E '\.(apr|gguf|safetensors|bin|pt|onnx|wav|mp3|db|db-shm|db-wal)$' || true)
    if [ -n "$binaries" ]; then
        large_files="${large_files}${pkg}: ${binaries}\n"
    fi
done
if [ -n "$large_files" ]; then
    echo "FAIL"
    echo "FAIL: Binary/model files found in packages:"
    echo -e "$large_files"
    echo "Fix: add to Cargo.toml [package] exclude or .gitignore"
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 7: No hardcoded /home/ paths in non-test production source
echo -n "  Hardcoded paths check... "
checked=$((checked + 1))
# Only check non-test source files that ship to users
# Exclude: test files, cfg(test) blocks, #[ignore] tests, examples
bad_paths=$(grep -rn '/home/' crates/apr-cli/src/ src/lib.rs src/traits.rs src/primitives/ src/format/ src/text/ \
    --include='*.rs' 2>/dev/null \
    | grep -v '_test' | grep -v 'mod tests' | grep -v '#\[test' | grep -v '#\[ignore' \
    | grep -v 'falsification' | grep -v '// ' \
    || true)
if [ -n "$bad_paths" ]; then
    echo "WARN"
    echo "WARN: Hardcoded /home/ paths found in source (may break other users):"
    echo "$bad_paths" | head -5
else
    echo "OK"
fi

# Check 8: No CI/CD or dev infrastructure in publishable packages
echo -n "  Package hygiene check... "
checked=$((checked + 1))
hygiene_fail=0
for pkg in aprender apr-cli; do
    leaked=$(cargo package -p "$pkg" --list --allow-dirty 2>/dev/null \
        | grep -E '\.github/|\.githooks/|\.pmat-metrics|Dockerfile' || true)
    if [ -n "$leaked" ]; then
        if [ "$hygiene_fail" -eq 0 ]; then
            echo "FAIL"
        fi
        echo "FAIL: Dev infrastructure files found in $pkg package:"
        echo "$leaked" | head -5
        hygiene_fail=1
    fi
done
if [ "$hygiene_fail" -gt 0 ]; then
    errors=$((errors + 1))
else
    echo "OK"
fi

# Check 9: Package file count sanity (catch accidental bloat regression)
echo -n "  Package file count check... "
checked=$((checked + 1))
aprender_count=$(cargo package -p aprender --list --allow-dirty 2>/dev/null | wc -l)
# Threshold: current is ~1520. Alert if it grows by >200 files (new bloat).
if [ "$aprender_count" -gt 1800 ]; then
    echo "WARN"
    echo "WARN: aprender package has $aprender_count files (threshold: 1800)"
    echo "Check for new directories that should be excluded in Cargo.toml"
else
    echo "OK ($aprender_count files)"
fi

# Summary
echo ""
if [ "$errors" -gt 0 ]; then
    echo "FAIL: $errors publish safety checks failed out of $checked"
    echo "See: contracts/publish-safety-v1.yaml"
    exit 1
else
    echo "OK: All $checked publish safety checks passed"
fi
