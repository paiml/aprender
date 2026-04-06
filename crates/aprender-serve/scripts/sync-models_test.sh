#!/bin/bash
# bashrs unit tests for sync-models.sh
# Run: bashrs test scripts/sync-models_test.sh
set -euo pipefail

# Derive path from test file location
readonly SCRIPT_UNDER_TEST="$(cd "$(dirname "$0")" && pwd)/sync-models.sh"

# Test: Script exists and is executable
test_script_exists() {
    [[ -f "$SCRIPT_UNDER_TEST" && -x "$SCRIPT_UNDER_TEST" ]]
}

# Test: Script has bash shebang
test_has_shebang() {
    head -1 "$SCRIPT_UNDER_TEST" | grep -q '^#!/bin/bash'
}

# Test: Script uses strict mode
test_has_strict_mode() {
    grep -q '^set -euo pipefail' "$SCRIPT_UNDER_TEST"
}

# Test: SEC010 - Path traversal validation function exists
test_sec010_validate_path_exists() {
    grep -q 'validate_path()' "$SCRIPT_UNDER_TEST"
}

# Test: SEC010 - Checks for '..' in paths
test_sec010_checks_dotdot() {
    grep -q '"\.\."' "$SCRIPT_UNDER_TEST"
}

# Test: Validates ARTIFACTS_DIR containment
test_artifacts_dir_containment() {
    grep -qE 'ARTIFACTS_DIR.*PROJECT_ROOT' "$SCRIPT_UNDER_TEST"
}

# Test: Uses readonly for constants
test_uses_readonly() {
    local count
    count="$(grep -c '^readonly' "$SCRIPT_UNDER_TEST")"
    [[ "$count" -ge 3 ]]
}

# Test: Defines Qwen2 model
test_defines_qwen2() {
    grep -q 'qwen2-0.5b' "$SCRIPT_UNDER_TEST"
}

# Test: Defines TinyLlama model
test_defines_tinyllama() {
    grep -q 'tinyllama' "$SCRIPT_UNDER_TEST"
}

# Test: Uses ln -sf for symlinks
test_uses_symlinks() {
    grep -q 'ln -sf' "$SCRIPT_UNDER_TEST"
}

# Test: Checks source file readability
test_checks_source_readable() {
    grep -q '\-r "\$source_file"' "$SCRIPT_UNDER_TEST"
}

# Test: Idempotent symlink creation
test_idempotent_symlink() {
    local tmpdir
    tmpdir="$(mktemp -d)"

    # SEC011: Validate tmpdir before cleanup
    if [[ -z "$tmpdir" || "$tmpdir" == "/" || "$tmpdir" == *".."* ]]; then
        echo "ERROR: Invalid tmpdir"
        return 1
    fi
    trap 'rm -rf "${tmpdir:?}"' RETURN

    local source_file="$tmpdir/source.gguf"
    local link_file="$tmpdir/link.gguf"

    echo "mock" > "$source_file"
    # shellcheck disable=SEC010  # Paths validated via tmpdir check above
    ln -sf "$source_file" "$link_file"
    # shellcheck disable=SEC010
    ln -sf "$source_file" "$link_file"  # Should not fail (idempotent)
    [[ -L "$link_file" ]]
}

# Test: No bashrs lint errors
test_no_lint_errors() {
    ! bashrs lint "$SCRIPT_UNDER_TEST" 2>&1 | grep -q '^✗'
}

# Test: No bashrs lint warnings (only warnings, not info)
test_no_lint_warnings() {
    ! bashrs lint "$SCRIPT_UNDER_TEST" 2>&1 | grep -qF '[warning]'
}
