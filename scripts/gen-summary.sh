#!/usr/bin/env bash
set -euo pipefail

# Generate SUMMARY.md from enforced Page Contract Units (PCUs)
# Usage: scripts/gen-summary.sh > book/src/SUMMARY.md
#
# Only pages with status=enforced and an existing .md file appear.
# Pages with status=draft are excluded — they don't exist in the book.

echo "# Aprender — Pure Rust ML Framework"
echo ""
echo "[Introduction](./introduction.md)"
echo ""

CURRENT_PART=""

for contract in contracts/apr-page-*-v1.yaml; do
    [ -f "$contract" ] || continue

    STATUS=$(grep "^  status:" "$contract" 2>/dev/null | head -1 | awk '{print $2}' || echo "")
    [ "$STATUS" = "enforced" ] || continue

    TITLE=$(grep "^  title:" "$contract" 2>/dev/null | head -1 | sed 's/.*title: *"//' | tr -d '"')
    PATH_VAL=$(grep "^  path:" "$contract" 2>/dev/null | head -1 | sed 's/.*path: *"//' | tr -d '"' | xargs)
    PART=$(grep "^  part:" "$contract" 2>/dev/null | head -1 | awk '{print $2}' | tr -d '"')
    CATEGORY=$(grep "^  category:" "$contract" 2>/dev/null | head -1 | awk '{print $2}' | tr -d '"')

    [ -z "$TITLE" ] && continue
    [ -z "$PATH_VAL" ] && continue
    [ -f "$PATH_VAL" ] || continue

    # Strip book/src/ prefix for relative path
    REL_PATH="${PATH_VAL#book/src/}"

    # Emit part headers
    if [ "$PART" != "$CURRENT_PART" ] && [ -n "$PART" ]; then
        CURRENT_PART="$PART"
        case "$PART" in
            I)         echo "# Part I: Foundations" ;;
            II)        echo "# Part II: Algorithms" ;;
            III)       echo "# Part III: Deep Learning & Inference" ;;
            IV)        echo "# Part IV: Production" ;;
            V)         echo "# Part V: Advanced Topics" ;;
            reference) echo "# Reference" ;;
        esac
        echo ""
    fi

    echo "- [${TITLE}](./${REL_PATH})"
done
