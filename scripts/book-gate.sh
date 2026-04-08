#!/usr/bin/env bash
set -euo pipefail

# Book Gate: validates a single Page Contract Unit (PCU)
# Usage: scripts/book-gate.sh <page-id>
# Example: scripts/book-gate.sh ch01-why-rust

ID="${1:?Usage: book-gate.sh <page-id>}"
CONTRACT="contracts/apr-page-${ID}-v1.yaml"
FAIL=0

echo "=== Book Gate: ${ID} ==="

# Gate 1: Contract exists
if [ ! -f "$CONTRACT" ]; then
    echo "  FAIL: contract $CONTRACT not found"
    exit 1
fi
echo "  PASS: contract exists"

# Gate 2: Contract is enforced
STATUS=$(grep "status:" "$CONTRACT" | head -1 | awk '{print $2}')
if [ "$STATUS" != "enforced" ]; then
    echo "  SKIP: status=$STATUS (not enforced)"
    exit 0
fi
echo "  PASS: status=enforced"

# Gate 3: Page .md exists at declared path
PAGE_PATH=$(grep "path:" "$CONTRACT" | head -1 | sed 's/.*path: *"//' | tr -d '"' | xargs)
if [ ! -f "$PAGE_PATH" ]; then
    echo "  FAIL: page $PAGE_PATH not found"
    FAIL=1
else
    echo "  PASS: page exists at $PAGE_PATH"
fi

# Gate 4: Page has PCU frontmatter
if [ -f "$PAGE_PATH" ]; then
    if ! head -1 "$PAGE_PATH" | grep -q "PCU:"; then
        echo "  FAIL: no PCU frontmatter in $PAGE_PATH"
        FAIL=1
    else
        echo "  PASS: PCU frontmatter present"
    fi
fi

# Gate 5: Example compiles and runs
EXAMPLE=$(grep "example:" "$CONTRACT" | head -1 | sed 's/.*example: *"//' | tr -d '"' | xargs)
if [ -n "$EXAMPLE" ]; then
    if cargo build -p aprender-core --example "$EXAMPLE" 2>/dev/null; then
        echo "  PASS: example compiles"
    else
        echo "  FAIL: example $EXAMPLE does not compile"
        FAIL=1
    fi
    if cargo run -p aprender-core --example "$EXAMPLE" >/dev/null 2>&1; then
        echo "  PASS: example runs"
    else
        echo "  FAIL: example $EXAMPLE exits non-zero"
        FAIL=1
    fi
else
    echo "  SKIP: no example declared"
fi

# Gate 6: No legacy names
if [ -f "$PAGE_PATH" ]; then
    LEGACY=$(grep -cE '\buse (trueno|realizar|entrenar|batuta|presentar|renacer)::' "$PAGE_PATH" 2>/dev/null || true)
    if [ "$LEGACY" -gt 0 ]; then
        echo "  FAIL: $LEGACY legacy name imports"
        FAIL=1
    else
        echo "  PASS: no legacy names"
    fi
fi

# Gate 7: No placeholder text
if [ -f "$PAGE_PATH" ]; then
    PLACEHOLDERS=$(grep -ciE '\bTODO\b|\bTBD\b|\bWIP\b|\bcoming soon\b' "$PAGE_PATH" 2>/dev/null || true)
    if [ "$PLACEHOLDERS" -gt 0 ]; then
        echo "  FAIL: $PLACEHOLDERS placeholder(s) found"
        FAIL=1
    else
        echo "  PASS: no placeholders"
    fi
fi

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "RESULT: ALL GATES PASS"
else
    echo "RESULT: FAILED — page should be deleted or fixed"
    exit 1
fi
