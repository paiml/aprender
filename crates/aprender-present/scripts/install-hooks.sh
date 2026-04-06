#!/bin/bash
#
# SPEC-024 Section 0: ENFORCEMENT ARCHITECTURE
#
# This script installs git hooks that enforce test-first development.
# Run once after cloning: ./scripts/install-hooks.sh
#
# TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  SPEC-024: Installing Test-First Enforcement Hooks                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'HOOK'
#!/bin/bash
#
# SPEC-024: Test-First Enforcement Pre-Commit Hook
#
# This hook BLOCKS commits that violate test-first development.

set -e

echo "SPEC-024: Running test-first enforcement..."

# Get staged files
STAGED=$(git diff --cached --name-only --diff-filter=ACM)

# Check for ptop implementation files without tests
PTOP_IMPL=$(echo "$STAGED" | grep 'crates/presentar-terminal/src/ptop/.*\.rs$' | grep -v 'mod.rs' || true)

for f in $PTOP_IMPL; do
    MODULE=$(basename "$f" .rs)

    # Check for inline tests
    if grep -q "#\[cfg(test)\]" "$f"; then
        continue
    fi

    # Check for external test file mentioning this module
    if grep -rq "test_.*${MODULE}\|${MODULE}.*test" crates/presentar-terminal/tests/ 2>/dev/null; then
        continue
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║  SPEC-024 ENFORCEMENT: COMMIT BLOCKED                                        ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                              ║"
    echo "║  File: $f"
    echo "║                                                                              ║"
    echo "║  This implementation file has no corresponding tests.                        ║"
    echo "║                                                                              ║"
    echo "║  REQUIRED STEPS:                                                             ║"
    echo "║  1. Write the test FIRST (it should fail to compile initially)               ║"
    echo "║  2. Add the interface (struct fields, function signatures)                   ║"
    echo "║  3. Implement until tests pass                                               ║"
    echo "║  4. Stage both test and implementation files                                 ║"
    echo "║                                                                              ║"
    echo "║  TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.                             ║"
    echo "║                                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    exit 1
done

# Run interface tests for ptop changes
if echo "$STAGED" | grep -q 'crates/presentar-terminal/src/ptop/'; then
    echo "Running interface tests for ptop changes..."
    if ! cargo test -p presentar-terminal --features ptop --test cpu_exploded_async --quiet 2>/dev/null; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════════════════════╗"
        echo "║  SPEC-024 ENFORCEMENT: INTERFACE TESTS FAILED                                ║"
        echo "╠══════════════════════════════════════════════════════════════════════════════╣"
        echo "║                                                                              ║"
        echo "║  The interface-defining tests in tests/cpu_exploded_async.rs failed.         ║"
        echo "║                                                                              ║"
        echo "║  Your changes may have broken the async data flow contract.                  ║"
        echo "║  Fix the tests or update the interface definition.                           ║"
        echo "║                                                                              ║"
        echo "╚══════════════════════════════════════════════════════════════════════════════╝"
        exit 1
    fi
fi

echo "SPEC-024: Enforcement passed."
HOOK

chmod +x "$HOOKS_DIR/pre-commit"
echo "Installed: pre-commit hook"

# Create pre-push hook (runs full test suite)
cat > "$HOOKS_DIR/pre-push" << 'HOOK'
#!/bin/bash
#
# SPEC-024: Test-First Enforcement Pre-Push Hook
#
# This hook runs the full test suite before pushing.

set -e

echo "SPEC-024: Running full test suite before push..."

# Build with enforcement (triggers build.rs checks)
if ! cargo build -p presentar-terminal --features ptop 2>/dev/null; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║  SPEC-024 ENFORCEMENT: BUILD FAILED                                          ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                              ║"
    echo "║  The build failed. This may be due to missing interface tests.               ║"
    echo "║  Check build.rs output for enforcement violations.                           ║"
    echo "║                                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    exit 1
fi

# Run interface tests
if ! cargo test -p presentar-terminal --features ptop --test cpu_exploded_async; then
    echo "Interface tests failed. Push blocked."
    exit 1
fi

echo "SPEC-024: All enforcement checks passed. Push allowed."
HOOK

chmod +x "$HOOKS_DIR/pre-push"
echo "Installed: pre-push hook"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  SPEC-024 Hooks Installed Successfully                                       ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║                                                                              ║"
echo "║  pre-commit: Blocks commits without tests                                    ║"
echo "║  pre-push:   Runs full test suite before push                                ║"
echo "║                                                                              ║"
echo "║  TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.                             ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
