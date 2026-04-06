#!/bin/bash
# ABSOLUTE ZERO JAVASCRIPT enforcement script
# Fails if any JavaScript/TypeScript files are found in the codebase

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Checking for ABSOLUTE ZERO JAVASCRIPT compliance..."

# Check for forbidden file extensions
# Excludes: target/, node_modules/, .git/, book/ (mdbook output)
JS_FILES=$(find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.mjs" -o -name "*.cjs" \) \
    -not -path "./target/*" \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -not -path "./book/book/*" \
    2>/dev/null || true)

if [ -n "$JS_FILES" ]; then
    echo -e "${RED}ERROR: JavaScript/TypeScript files found!${NC}"
    echo "$JS_FILES"
    echo ""
    echo "ABSOLUTE ZERO JAVASCRIPT policy violated."
    echo "All code must be pure Rust with web-sys bindings."
    exit 1
fi

# Check for npm artifacts
if [ -f "package.json" ] || [ -f "package-lock.json" ] || [ -f "yarn.lock" ] || [ -f "pnpm-lock.yaml" ]; then
    echo -e "${RED}ERROR: npm/yarn/pnpm artifacts found!${NC}"
    echo "Found package manager files - these are forbidden."
    exit 1
fi

# Check for node_modules
if [ -d "node_modules" ]; then
    echo -e "${RED}ERROR: node_modules directory found!${NC}"
    echo "JavaScript dependencies are forbidden."
    exit 1
fi

echo -e "${GREEN}ABSOLUTE ZERO JAVASCRIPT: PASSED${NC}"
exit 0
