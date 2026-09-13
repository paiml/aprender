#!/usr/bin/env bash
set -euo pipefail

# Book Gate: validates a single Page Contract Unit (PCU)
# Usage: scripts/book-gate.sh <page-id> [--strict]
#
# 12 gates covering the full proof chain:
#   G1-G7:  existence + compilation + runtime + namespace
#   G8-G12: section cross-check, prose minimum, api_calls, arXiv, oracle
#
# --strict: fail on prose/section leaks (for chapters only)
# Without --strict: warn but don't fail (for CI on case-studies/theory)

ID="${1:?Usage: book-gate.sh <page-id> [--strict]}"
STRICT="${2:-}"
FAIL=0
WARN=0

# Try chapter contract first, then page contract
if [ -f "contracts/apr-book-${ID}-v1.yaml" ]; then
    CONTRACT="contracts/apr-book-${ID}-v1.yaml"
elif [ -f "contracts/apr-page-${ID}-v1.yaml" ]; then
    CONTRACT="contracts/apr-page-${ID}-v1.yaml"
else
    echo "  FAIL: no contract found for ${ID}"
    exit 1
fi

echo "=== Book Gate: ${ID} ($(basename $CONTRACT)) ==="

# Gate 1: Contract exists (already confirmed above)
echo "  G1  PASS: contract exists"

# Gate 2: Contract is enforced
STATUS=$(grep "^status:" "$CONTRACT" | head -1 | awk '{print $2}')
if [ "$STATUS" != "enforced" ]; then
    echo "  G2  SKIP: status=$STATUS (not enforced)"
    exit 0
fi
echo "  G2  PASS: status=enforced"

# Gate 3: Page .md exists at declared path
PAGE_PATH=$(grep "path:" "$CONTRACT" | head -1 | sed 's/.*path: *"//' | tr -d '"' | xargs)
if [ -z "$PAGE_PATH" ] || [ ! -f "$PAGE_PATH" ]; then
    echo "  G3  FAIL: page not found at $PAGE_PATH"
    FAIL=1
else
    echo "  G3  PASS: page exists"
fi

# Gate 4: Page has PCU frontmatter
if [ -f "$PAGE_PATH" ]; then
    if grep -q "PCU:" <<< "$(head -1 "$PAGE_PATH")" ; then
        echo "  G4  PASS: PCU frontmatter"
    else
        echo "  G4  FAIL: no PCU frontmatter"
        FAIL=1
    fi
fi

# Gate 5: Example compiles
EXAMPLE=$(grep "example:" "$CONTRACT" | head -1 | sed 's/.*example: *"//' | tr -d '"' | xargs)
if [ -n "$EXAMPLE" ] && [ "$EXAMPLE" != "none" ]; then
    if cargo build -p aprender-core --example "$EXAMPLE" 2>/dev/null; then
        echo "  G5  PASS: example compiles"
    else
        echo "  G5  FAIL: example $EXAMPLE does not compile"
        FAIL=1
    fi
else
    echo "  G5  SKIP: no example"
fi

# Gate 6: Example runs exit 0
if [ -n "$EXAMPLE" ] && [ "$EXAMPLE" != "none" ]; then
    if cargo run -p aprender-core --example "$EXAMPLE" >/dev/null 2>&1; then
        echo "  G6  PASS: example runs"
    else
        echo "  G6  FAIL: example exits non-zero"
        FAIL=1
    fi
else
    echo "  G6  SKIP: no example"
fi

# Gate 7: No legacy names in page
if [ -f "$PAGE_PATH" ]; then
    if grep -qE '\b(trueno|realizar|entrenar|batuta|presentar|renacer)\b' "$PAGE_PATH" 2>/dev/null; then
        # Allow legacy names only in strikethrough (~~name~~) or code comments explaining migration
        REAL_LEGACY=$(grep -cE '\buse (trueno|realizar|entrenar|batuta|presentar|renacer)::' "$PAGE_PATH" 2>/dev/null || echo 0)
        if [ "$REAL_LEGACY" -gt 0 ]; then
            echo "  G7  FAIL: $REAL_LEGACY legacy use-imports"
            FAIL=1
        else
            echo "  G7  PASS: no legacy imports (mentions in context OK)"
        fi
    else
        echo "  G7  PASS: no legacy names"
    fi
fi

# Gate 8: No placeholder text
if [ -f "$PAGE_PATH" ]; then
    PLACEHOLDERS=$(grep -ciE '\bTODO\b|\bTBD\b|\bWIP\b|\bcoming soon\b|\bunder construction\b' "$PAGE_PATH" 2>/dev/null || echo 0)
    if [ "$PLACEHOLDERS" -gt 0 ]; then
        echo "  G8  FAIL: $PLACEHOLDERS placeholder(s)"
        FAIL=1
    else
        echo "  G8  PASS: no placeholders"
    fi
fi

# Gate 9: Section cross-check (L2 fix)
# Contract sections must appear as H2 headings in page
if [ -f "$PAGE_PATH" ]; then
    CONTRACT_SECTIONS=$(grep "heading:" "$CONTRACT" 2>/dev/null | sed 's/.*heading: *"//' | tr -d '"' | head -20)
    if [ -n "$CONTRACT_SECTIONS" ]; then
        PAGE_H2S=$(grep "^## " "$PAGE_PATH" 2>/dev/null | sed 's/^## //')
        MISSING_SECTIONS=0
        while IFS= read -r section; do
            [ -z "$section" ] && continue
            if ! grep -qiF "$section" <<< "$PAGE_H2S" ; then
                MISSING_SECTIONS=$((MISSING_SECTIONS+1))
            fi
        done <<< "$CONTRACT_SECTIONS"
        if [ "$MISSING_SECTIONS" -gt 0 ]; then
            if [ "$STRICT" = "--strict" ]; then
                echo "  G9  FAIL: $MISSING_SECTIONS contract sections missing from page"
                FAIL=1
            else
                echo "  G9  WARN: $MISSING_SECTIONS contract sections missing from page"
                WARN=$((WARN+1))
            fi
        else
            echo "  G9  PASS: all contract sections present"
        fi
    else
        echo "  G9  SKIP: no sections in contract"
    fi
fi

# Gate 10: Prose minimum (L6 fix)
# Chapter pages must have at least 5 lines of prose (not just frontmatter + include)
if [ -f "$PAGE_PATH" ]; then
    CATEGORY=$(grep "category:" "$CONTRACT" 2>/dev/null | head -1 | awk '{print $2}' | tr -d '"')
    # Count non-boilerplate lines (skip frontmatter, blank, headings, code fences, includes)
    PROSE_LINES=$(tail -n +5 "$PAGE_PATH" | grep -cvE '^$|^#|^>|^```|^\{\{|^Run:|^---|^<!--' 2>/dev/null || echo 0)
    if [ "$CATEGORY" = "chapter" ] && [ "$PROSE_LINES" -lt 5 ]; then
        if [ "$STRICT" = "--strict" ]; then
            echo "  G10 FAIL: only $PROSE_LINES prose lines (chapter needs >=5)"
            FAIL=1
        else
            echo "  G10 WARN: only $PROSE_LINES prose lines (chapter needs >=5)"
            WARN=$((WARN+1))
        fi
    else
        echo "  G10 PASS: $PROSE_LINES prose lines"
    fi
fi

# Gate 11: api_calls verification (L1 fix — automated)
# If contract declares api_calls, example must have matching use statements
if [ -n "$EXAMPLE" ] && [ "$EXAMPLE" != "none" ]; then
    EXAMPLE_FILE="crates/aprender-core/examples/${EXAMPLE}.rs"
    HAS_API_CALLS=$(grep -c "api_calls:" "$CONTRACT" 2>/dev/null || echo 0)
    if [ "$HAS_API_CALLS" -gt 0 ] && [ -f "$EXAMPLE_FILE" ]; then
        IMPORTS=$(grep -c "^use aprender::" "$EXAMPLE_FILE" 2>/dev/null || echo 0)
        if [ "$IMPORTS" -eq 0 ]; then
            echo "  G11 FAIL: contract has api_calls but example has 0 aprender imports"
            FAIL=1
        else
            echo "  G11 PASS: $IMPORTS aprender imports (api_calls satisfied)"
        fi
    else
        echo "  G11 SKIP: no api_calls in contract"
    fi
else
    echo "  G11 SKIP: no example"
fi

# Gate 12: arXiv IDs are well-formed (L3 partial fix)
ARXIV_LINE=$(grep "arxiv:" "$CONTRACT" 2>/dev/null | head -1)
if grep -qP '\d{4}\.\d{4,5}' <<< "$ARXIV_LINE" ; then
    # Check format: YYMM.NNNNN
    BAD_IDS=$(echo "$ARXIV_LINE" | grep -oP '"[^"]*"' | tr -d '"' | while read -r id; do
        grep -qP '^\d{4}\.\d{4,5}$|^math/\d{7}$' <<< "$id" || echo "$id"
    done)
    if [ -n "$BAD_IDS" ]; then
        echo "  G12 WARN: malformed arXiv ID(s): $BAD_IDS"
        WARN=$((WARN+1))
    else
        echo "  G12 PASS: arXiv IDs well-formed"
    fi
else
    echo "  G12 SKIP: no arXiv IDs"
fi

echo ""
if [ "$FAIL" -eq 0 ] && [ "$WARN" -eq 0 ]; then
    echo "RESULT: ALL 12 GATES PASS"
elif [ "$FAIL" -eq 0 ]; then
    echo "RESULT: PASS with $WARN warning(s)"
else
    echo "RESULT: FAILED ($FAIL hard failures, $WARN warnings)"
    exit 1
fi
