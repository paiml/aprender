#!/bin/bash
# Install git hooks for Trueno development
#
# This script installs pre-commit hooks that enforce code quality standards:
# - SIMD attribute validation (prevents missing #[target_feature])
# - Prevents commits that would cause 5.9x-21x performance degradation
#
# Usage: ./scripts/install-hooks.sh

set -e

echo "🔧 Installing git hooks..."
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
    echo "✅ Installed pre-commit hook: SIMD attribute validation"
else
    echo "❌ Error: .githooks/pre-commit not found"
    exit 1
fi

echo ""
echo "🎉 Git hooks installed successfully!"
echo ""
echo "The following checks will run before each commit:"
echo "  • SIMD attribute validation (cargo xtask check-simd)"
echo ""
echo "To bypass hooks (EMERGENCY ONLY):"
echo "  git commit --no-verify"
echo ""
