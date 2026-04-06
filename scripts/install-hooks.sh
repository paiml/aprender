#!/bin/bash
# Install git hooks for Repartir development
#
# This script installs pre-commit hooks that enforce code quality standards:
# - Zero-tolerance clippy (no warnings)
# - Code formatting (cargo fmt)
# - Fast test suite (library tests)
# - mdBook validation (documentation builds)
# - bashrs shell linting (Makefile and scripts)
#
# Usage: ./scripts/install-hooks.sh

set -e

echo "🔧 Installing Repartir git hooks..."
echo ""

# Check if .git directory exists
if [ ! -d ".git" ]; then
    echo "❌ Error: .git directory not found"
    echo "   Please run this script from the repository root"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Install pre-commit hook
if [ -f ".githooks/pre-commit" ]; then
    cp .githooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo "✅ Installed pre-commit hook"
else
    echo "❌ Error: .githooks/pre-commit not found"
    exit 1
fi

echo ""
echo "🎉 Git hooks installed successfully!"
echo ""
echo "The following checks will run before each commit:"
echo "  1. cargo clippy --all-targets --all-features -- -D warnings"
echo "  2. cargo fmt --check"
echo "  3. cargo test --lib"
echo "  4. mdbook build book/"
echo "  5. bashrs lint (on staged shell files)"
echo ""
echo "Quality standards enforced:"
echo "  • Iron Lotus Framework (Jidoka, Kaizen, Genchi Genbutsu)"
echo "  • Zero tolerance for warnings or lint violations"
echo "  • All tests must pass"
echo "  • Documentation must build without errors"
echo ""
echo "To bypass hooks (EMERGENCY ONLY):"
echo "  git commit --no-verify"
echo ""
echo "Recommended: Install optional tools for full validation:"
echo "  cargo install mdbook    # For book validation"
echo "  cargo install bashrs    # For shell script linting"
echo ""
