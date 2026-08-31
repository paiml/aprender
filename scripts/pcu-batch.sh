#!/usr/bin/env bash
set -euo pipefail

# PCU Batch: Generate contract YAML + add frontmatter for all book pages
# Usage: scripts/pcu-batch.sh [--dry-run]
#
# For each .md in book/src/ (except SUMMARY.md):
#   1. Derive page ID from filepath
#   2. Detect category from directory
#   3. Detect example from {{#include}} directives
#   4. Extract H2 section headings
#   5. Generate contracts/apr-page-{id}-v1.yaml
#   6. Prepend PCU frontmatter to .md file

DRY_RUN="${1:-}"
CREATED=0
SKIPPED=0

for md in $(find book/src -name "*.md" -not -name "SUMMARY.md" | sort); do
    # Derive ID from path: book/src/examples/foo-bar.md -> examples-foo-bar
    REL="${md#book/src/}"
    ID=$(echo "$REL" | sed 's|/|-|g; s|\.md$||')
    CONTRACT="contracts/apr-page-${ID}-v1.yaml"

    # Skip if contract already exists
    if [ -f "$CONTRACT" ]; then
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    # Detect category from directory
    DIR=$(dirname "$REL")
    case "$DIR" in
        examples)          CATEGORY="case-study" ;;
        ml-fundamentals*)  CATEGORY="theory" ;;
        chapters)          CATEGORY="chapter" ;;
        getting-started)   CATEGORY="guide" ;;
        cli-reference)     CATEGORY="tool" ;;
        architecture)      CATEGORY="guide" ;;
        quality-gates)     CATEGORY="guide" ;;
        methodology)       CATEGORY="theory" ;;
        advanced-testing)  CATEGORY="theory" ;;
        best-practices)    CATEGORY="guide" ;;
        tools)             CATEGORY="tool" ;;
        .)                 CATEGORY="guide" ;;
        *)                 CATEGORY="guide" ;;
    esac

    # Detect part from directory
    case "$DIR" in
        examples|ml-fundamentals*) PART="reference" ;;
        chapters)                  PART="I" ;;  # will be overridden per chapter
        getting-started)           PART="reference" ;;
        *)                         PART="reference" ;;
    esac

    # Detect example from {{#include}} directive
    EXAMPLE=""
    INCLUDE_LINE=$(grep '{{#include' "$md" 2>/dev/null | head -1 || true)
    if [ -n "$INCLUDE_LINE" ]; then
        EXAMPLE=$(echo "$INCLUDE_LINE" | grep -oP 'examples/\K[^.]+' || true)
    fi

    # Extract title from first H1
    TITLE=$(grep '^# ' "$md" | head -1 | sed 's/^# //' | tr -d '\r')
    [ -z "$TITLE" ] && TITLE="$ID"

    # Extract H2 section headings
    SECTIONS=""
    while IFS= read -r heading; do
        heading=$(echo "$heading" | sed 's/^## //' | tr -d '\r')
        [ -n "$heading" ] && SECTIONS="${SECTIONS}
  - heading: \"${heading}\"
    has_code: true
    has_assertion: false
    citation: null"
    done < <(grep '^## ' "$md" 2>/dev/null || true)

    # Determine if api_calls required
    API_CALLS=""
    if [ "$CATEGORY" = "case-study" ] || [ "$CATEGORY" = "chapter" ]; then
        if [ -n "$EXAMPLE" ]; then
            EXAMPLE_FILE="crates/aprender-core/examples/${EXAMPLE}.rs"
            if [ -f "$EXAMPLE_FILE" ]; then
                # Extract aprender modules used
                MODULES=$(grep '^use aprender::' "$EXAMPLE_FILE" 2>/dev/null | sed 's/use aprender::\([^:;{]*\).*/\1/' | sort -u | head -3)
                if [ -n "$MODULES" ]; then
                    API_CALLS="
api_calls:"
                    for mod in $MODULES; do
                        API_CALLS="${API_CALLS}
  - module: \"aprender::${mod}\"
    functions: []
    min_calls: 1"
                    done
                fi
            fi
        fi
    fi

    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "DRY: $CONTRACT ($CATEGORY, example=$EXAMPLE)"
        CREATED=$((CREATED+1))
        continue
    fi

    # Generate contract YAML
    cat > "$CONTRACT" << YAML
contract: apr-page-${ID}
version: 1
status: enforced
date: 2026-04-08

page:
  id: "${ID}"
  title: "${TITLE}"
  part: "${PART}"
  category: "${CATEGORY}"
  path: "${md}"
  example: "${EXAMPLE}"
  arxiv: []
${API_CALLS}
sections:${SECTIONS:-"
  []"}

falsification:
  - condition: "Page .md file does not exist at declared path"
    severity: P0
    action: delete_from_summary
  - condition: "Example does not compile"
    severity: P0
    action: delete_from_summary
  - condition: "Example exits non-zero"
    severity: P0
    action: delete_from_summary
  - condition: "Section in .md not listed in contract"
    severity: P0
    action: delete_section
  - condition: "Legacy name in page text"
    severity: P0
    action: delete_from_summary
YAML

    # Prepend PCU frontmatter to .md (if not already present)
    if ! grep -q "PCU:" <<< "$(head -1 "$md")" ; then
        TMPF=$(mktemp)
        echo "<!-- PCU: ${ID} | contract: ${CONTRACT} -->" > "$TMPF"
        echo "<!-- Example: cargo run -p aprender-core --example ${EXAMPLE:-none} -->" >> "$TMPF"
        echo "<!-- Status: enforced -->" >> "$TMPF"
        echo "" >> "$TMPF"
        cat "$md" >> "$TMPF"
        mv "$TMPF" "$md"
    fi

    CREATED=$((CREATED+1))
done

echo ""
echo "Created: $CREATED contracts"
echo "Skipped: $SKIPPED (already exist)"
echo "Total contracts: $(ls contracts/apr-page-*-v1.yaml 2>/dev/null | wc -l)"
