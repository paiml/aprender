#!/bin/bash
# Add placeholder content to skeleton book chapters
# Follows trueno's gating approach: structure exists, content added when feature is implemented

set -euo pipefail

BOOK_SRC="book/src"

# Skip these files (already have content)
SKIP_FILES=(
    "SUMMARY.md"
    "introduction.md"
    "appendix/contributing.md"
)

# Function to check if file should be skipped
should_skip() {
    local file="$1"
    for skip in "${SKIP_FILES[@]}"; do
        if [[ "$file" == *"$skip" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to extract chapter title from filename
get_chapter_title() {
    local file="$1"
    local basename=$(basename "$file" .md)
    # Convert kebab-case to Title Case
    echo "$basename" | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) tolower(substr($i,2))}1'
}

# Function to determine phase from path
get_phase() {
    local file="$1"

    # Phase 1 (COMPLETE): Core implementation
    if [[ "$file" == *"/formats/"* ]] || \
       [[ "$file" == *"/quantization/q4"* ]] || \
       [[ "$file" == *"/quantization/q8"* ]] || \
       [[ "$file" == *"/quantization/what-is"* ]] || \
       [[ "$file" == *"/transformer/"* ]] || \
       [[ "$file" == *"/tokenization/"* ]] || \
       [[ "$file" == *"/generation/"* ]] && [[ "$file" != *"/streaming"* ]] || \
       [[ "$file" == *"/api/"* ]] || \
       [[ "$file" == *"/cli/"* ]] || \
       [[ "$file" == *"/phases/phase1"* ]] || \
       [[ "$file" == *"/tdd/"* ]] || \
       [[ "$file" == *"/quality/"* ]] || \
       [[ "$file" == *"/performance/"* ]]; then
        echo "Phase 1 (COMPLETE)"

    # Phase 2: Optimization
    elif [[ "$file" == *"/quantization/advanced"* ]] || \
         [[ "$file" == *"/quantization/k-quants"* ]] || \
         [[ "$file" == *"/streaming"* ]] || \
         [[ "$file" == *"/phases/phase2"* ]]; then
        echo "Phase 2 (Optimization)"

    # Phase 3: Advanced Models
    elif [[ "$file" == *"/phases/phase3"* ]]; then
        echo "Phase 3 (Advanced Models)"

    # Phase 4: Production
    elif [[ "$file" == *"/phases/phase4"* ]]; then
        echo "Phase 4 (Production)"

    # General/Always relevant
    else
        echo "Documentation"
    fi
}

# Count files processed
PROCESSED=0
SKIPPED=0

echo "Adding placeholder content to skeleton chapters..."
echo ""

# Find all markdown files
while IFS= read -r file; do
    # Get relative path from book/src
    rel_path="${file#$BOOK_SRC/}"

    # Skip certain files
    if should_skip "$rel_path"; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Check if file is empty or very small (< 100 bytes)
    file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    if [ "$file_size" -gt 100 ]; then
        # File has substantial content, skip it
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Get chapter title and phase
    title=$(get_chapter_title "$file")
    phase=$(get_phase "$rel_path")

    # Create placeholder content using printf to avoid heredoc parsing issues with bashrs
    {
        printf '# %s\n\n' "$title"
        printf '%s **Status**: %s\n' ">" "$phase"
        printf '%s\n' ">"
        printf '%s This chapter will be written when the corresponding feature is implemented and test-backed.\n\n' ">"
        printf '[Content to be added]\n\n'
        printf 'This chapter will cover:\n'
        printf -- '- Overview and key concepts\n'
        printf -- '- Implementation details using EXTREME TDD\n'
        printf -- '- Code examples (all test-backed, zero hallucination)\n'
        printf -- '- Best practices and patterns\n'
        printf -- '- Performance considerations\n'
        printf -- '- Real-world examples from the codebase\n\n'
        printf '## Placeholder Notice\n\n'
        printf "This section is currently under development following Realizar's **gating approach**:\n\n"
        printf '1. **Feature must be implemented** - Code exists in `src/`\n'
        printf '2. **Tests must pass** - Comprehensive test coverage (unit, property, mutation)\n'
        printf '3. **Examples must run** - All code examples are validated by tests\n'
        printf '4. **Zero hallucination** - Only document what actually exists\n\n'
        printf 'Please check back later or refer to:\n'
        printf -- '- Source code in `src/` directory\n'
        printf -- '- Test files in `tests/` directory\n'
        printf -- '- Working examples in `examples/` directory\n'
        printf -- '- Inline rustdoc documentation\n\n'
        printf '**Contributing**: See [Contributing to This Book](../appendix/contributing.md) for guidelines on adding content.\n'
    } > "$file"

    PROCESSED=$((PROCESSED + 1))
    echo "✓ $rel_path"
done < <(find "$BOOK_SRC" -type f -name "*.md" | sort)

echo ""
echo "================================"
echo "Placeholder Addition Summary"
echo "================================"
echo "Files processed: $PROCESSED"
echo "Files skipped: $SKIPPED"
echo ""
echo "✅ All skeleton chapters now have placeholder content"
echo "   following trueno's gating approach"
echo ""
