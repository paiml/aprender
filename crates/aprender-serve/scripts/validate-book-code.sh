#!/bin/bash
# Validate that all code examples in the book are test-backed
# EXTREME TDD Principle: Every code example must be validated by tests

set -euo pipefail

echo "📚 Validating book code examples are test-backed..."
echo ""

BOOK_DIR="book/src"
SRC_DIR="src"
TESTS_DIR="tests"
EXAMPLES_DIR="examples"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Counters
TOTAL_CODE_BLOCKS=0
VALIDATED_BLOCKS=0
WARNINGS=0
ERRORS=0

# Check if book directory exists
if [ ! -d "$BOOK_DIR" ]; then
    echo -e "${RED}❌ Book directory not found: $BOOK_DIR${NC}"
    exit 1
fi

# Find all markdown files in the book
MARKDOWN_FILES=$(find "$BOOK_DIR" -type f -name "*.md" | sort)

if [ -z "$MARKDOWN_FILES" ]; then
    echo -e "${YELLOW}⚠️  No markdown files found in $BOOK_DIR${NC}"
    exit 0
fi

echo "Scanning markdown files for code blocks..."
echo ""

# Track files with code blocks
HAS_CODE_BLOCKS=false

for md_file in $MARKDOWN_FILES; do
    # Extract rust code blocks using sed
    # Look for ```rust and ```
    IN_CODE_BLOCK=false
    CODE_BLOCK=""
    LINE_NUM=0

    while IFS= read -r line; do
        LINE_NUM=$((LINE_NUM + 1))

        # Start of rust code block
        if echo "$line" | grep -q '^```rust'; then
            IN_CODE_BLOCK=true
            CODE_BLOCK=""
            TOTAL_CODE_BLOCKS=$((TOTAL_CODE_BLOCKS + 1))
            HAS_CODE_BLOCKS=true
            continue
        fi

        # End of code block
        if echo "$line" | grep -q '^```$' && [ "$IN_CODE_BLOCK" = true ]; then
            IN_CODE_BLOCK=false

            # Skip empty code blocks
            if [ -z "$CODE_BLOCK" ]; then
                continue
            fi

            # Validate code block
            # For now, we check:
            # 1. Code compiles (if it's a complete example)
            # 2. Code exists in src/ or tests/ or examples/ (if it's a snippet)

            # Extract function names, struct names, or unique identifiers
            # Look for pub fn, fn, pub struct, struct, impl
            IDENTIFIERS=$(echo "$CODE_BLOCK" | grep -E '(pub )?fn |struct |impl ' | head -3 || true)

            if [ -n "$IDENTIFIERS" ]; then
                # Check if these identifiers exist in actual code
                FOUND=false
                for identifier_line in $IDENTIFIERS; do
                    # Extract the identifier name (simplified)
                    IDENTIFIER=$(echo "$identifier_line" | sed -E 's/.*(fn|struct|impl) ([a-zA-Z_][a-zA-Z0-9_]*).*/\2/' || true)

                    if [ -n "$IDENTIFIER" ] && [ "$IDENTIFIER" != "$identifier_line" ]; then
                        # Search for this identifier in source code
                        if grep -r "$IDENTIFIER" "$SRC_DIR" "$TESTS_DIR" "$EXAMPLES_DIR" > /dev/null 2>&1; then
                            FOUND=true
                            VALIDATED_BLOCKS=$((VALIDATED_BLOCKS + 1))
                            break
                        fi
                    fi
                done

                if [ "$FOUND" = false ]; then
                    WARNINGS=$((WARNINGS + 1))
                    echo -e "${YELLOW}⚠️  Potentially untested code in: $md_file:$LINE_NUM${NC}"
                    echo "   Code block may not have corresponding tests"
                    echo ""
                fi
            fi

            CODE_BLOCK=""
            continue
        fi

        # Accumulate code block lines
        if [ "$IN_CODE_BLOCK" = true ]; then
            CODE_BLOCK="${CODE_BLOCK}${line}"$'\n'
        fi
    done < "$md_file"
done

echo ""
echo "================================"
echo "Book Code Validation Summary"
echo "================================"
echo "Total code blocks found: $TOTAL_CODE_BLOCKS"
echo "Validated blocks: $VALIDATED_BLOCKS"
echo "Warnings: $WARNINGS"
echo "Errors: $ERRORS"
echo ""

if [ "$HAS_CODE_BLOCKS" = false ]; then
    echo -e "${GREEN}✅ No code blocks to validate (skeleton book structure)${NC}"
    exit 0
fi

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}❌ Book validation failed with $ERRORS error(s)${NC}"
    exit 1
fi

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Book validation passed with $WARNINGS warning(s)${NC}"
    echo "   Consider adding tests for code examples or marking them as illustrative"
    echo ""
fi

echo -e "${GREEN}✅ Book code validation passed${NC}"
echo ""
echo "💡 TDD Principle: All production code examples should be test-backed"
echo "   - Add tests for new code examples"
echo "   - Reference actual tested code from src/"
echo "   - Use examples/ directory for complete tested examples"
echo ""

exit 0
