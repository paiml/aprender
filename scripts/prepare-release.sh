#!/bin/bash
# Release preparation script for Aprender
# Usage: ./scripts/prepare-release.sh <version>
# Example: ./scripts/prepare-release.sh 0.1.1

set -e

VERSION="${1}"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.1"
    exit 1
fi

echo "🚀 Preparing release v${VERSION}"
echo "================================"

# 1. Verify clean working directory
echo "📋 Checking git status..."
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Working directory is not clean. Commit or stash changes first."
    git status --short
    exit 1
fi
echo "✅ Working directory is clean"

# 2. Ensure on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Not on main branch (currently on ${CURRENT_BRANCH})"
    exit 1
fi
echo "✅ On main branch"

# 3. Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main
echo "✅ Up to date with remote"

# 4. Run all quality checks
echo "🔍 Running quality gates..."
# shellcheck source=scripts/pmat_bin.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/pmat_bin.sh" || exit 1
if ! "$PMAT" quality-gate; then
    echo "❌ Quality gates failed"
    exit 1
fi
echo "✅ Quality gates passed"

# 5. Run full test suite
echo "🧪 Running full test suite..."
if ! make test; then
    echo "❌ Tests failed"
    exit 1
fi
echo "✅ All tests passed"

# 6. Check test coverage
echo "📊 Checking test coverage..."
COVERAGE=$(cargo llvm-cov report | grep TOTAL | awk '{print $10}' | sed 's/%//')
if (( $(echo "$COVERAGE < 95.0" | bc -l) )); then
    echo "❌ Test coverage ${COVERAGE}% is below 95%"
    exit 1
fi
echo "✅ Test coverage: ${COVERAGE}%"

# 7. Verify cargo package
echo "📦 Verifying cargo package..."
if ! cargo publish --dry-run; then
    echo "❌ Cargo package verification failed"
    exit 1
fi
echo "✅ Package verification passed"

# 8. Check documentation builds
echo "📚 Building documentation..."
if ! cargo doc --no-deps --all-features; then
    echo "❌ Documentation build failed"
    exit 1
fi
echo "✅ Documentation builds successfully"

# 9. Run examples
echo "🎯 Running examples..."
for example in boston_housing iris_clustering dataframe_basics; do
    echo "  Running ${example}..."
    if ! cargo run --example "$example" > /dev/null 2>&1; then
        echo "❌ Example ${example} failed"
        exit 1
    fi
done
echo "✅ All examples run successfully"

# 10. Update version in Cargo.toml
echo "📝 Updating version in Cargo.toml..."
sed -i "s/^version = \".*\"/version = \"${VERSION}\"/" Cargo.toml
echo "✅ Version updated to ${VERSION}"

# 11. Summary
echo ""
echo "✅ Release v${VERSION} is ready!"
echo ""
echo "Next steps:"
echo "  1. Review CHANGELOG.md and update [Unreleased] section"
echo "  2. Commit version bump: git add Cargo.toml CHANGELOG.md && git commit -m 'Release v${VERSION}'"
echo "  3. Create tag: git tag -a v${VERSION} -m 'Release v${VERSION}'"
echo "  4. Push: git push origin main && git push origin v${VERSION}"
echo ""
echo "GitHub Actions will automatically:"
echo "  - Build artifacts for Linux, macOS, Windows"
echo "  - Run tests on all platforms"
echo "  - Create GitHub Release with artifacts"
echo "  - Publish to crates.io (if CARGO_REGISTRY_TOKEN is configured)"
echo ""
