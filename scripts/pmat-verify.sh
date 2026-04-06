#!/bin/bash
# PMAT Verification Tool - Tarantula Fault Localization & Annotation Links
# Per SPEC-024: Links specs to coverage, tracks churn, identifies fault hotspots

set -e

SPEC_FILE="docs/specifications/pixel-by-pixel-demo-ptop-ttop.md"
CRATE="presentar-terminal"

echo "=========================================="
echo "PMAT VERIFICATION REPORT"
echo "=========================================="
echo ""

# 1. SPEC↔COVERAGE LINKS
echo "1. SPEC↔COVERAGE LINKS"
echo "----------------------"

# Run coverage and capture summary
COVERAGE_OUTPUT=$(cargo llvm-cov --package $CRATE --features ptop 2>&1 | grep "^TOTAL" || echo "TOTAL 0 0 0%")
TOTAL_COV=$(echo "$COVERAGE_OUTPUT" | awk '{print $4}' | tr -d '%')

echo "Overall Coverage: ${TOTAL_COV}%"

if (( $(echo "$TOTAL_COV < 95" | bc -l) )); then
    echo "  ❌ BELOW TARGET (95% required)"
else
    echo "  ✅ MEETS TARGET"
fi

# Check spec sections vs implementation
echo ""
echo "Key Module Coverage:"
for module in app.rs ui.rs analyzers/mod.rs config.rs; do
    if [ -f "crates/$CRATE/src/ptop/$module" ]; then
        COV=$(cargo llvm-cov --package $CRATE --features ptop 2>&1 | grep "ptop/$module" | awk '{print $4}' | head -1)
        echo "  ptop/$module: ${COV:-N/A}"
    fi
done

echo ""

# 2. TICKET↔TDG (Technical Debt Grade)
echo "2. TICKET↔TDG LINKS"
echo "-------------------"

# Calculate cyclomatic complexity proxy via function count
FUNC_COUNT=$(grep -r "pub fn\|fn " crates/$CRATE/src/ptop/*.rs 2>/dev/null | wc -l)
FILE_COUNT=$(find crates/$CRATE/src/ptop -name "*.rs" | wc -l)
AVG_FUNCS=$((FUNC_COUNT / FILE_COUNT))

echo "ptop module stats:"
echo "  Files: $FILE_COUNT"
echo "  Functions: $FUNC_COUNT"
echo "  Avg funcs/file: $AVG_FUNCS"

if [ "$AVG_FUNCS" -gt 30 ]; then
    echo "  ⚠️  HIGH TDG - Consider refactoring"
else
    echo "  ✅ TDG within bounds"
fi

echo ""

# 3. TICKET↔CHURN TRACKING
echo "3. TICKET↔CHURN TRACKING"
echo "------------------------"

echo "File changes in last 30 days (top 10):"
git log --since="30 days ago" --name-only --pretty=format: -- "crates/$CRATE/src/ptop/**/*.rs" 2>/dev/null | \
    sort | uniq -c | sort -rn | head -10 | while read count file; do
    if [ -n "$file" ]; then
        echo "  $count changes: $file"
    fi
done

echo ""

# 4. TARANTULA FAULT LOCALIZATION
echo "4. TARANTULA FAULT LOCALIZATION"
echo "--------------------------------"

echo "Commits with 'fix' in message (potential bug hotspots):"
git log --oneline --since="30 days ago" --grep="fix" -- "crates/$CRATE/src/ptop/" 2>/dev/null | head -10

echo ""
echo "Files with multiple fixes (Tarantula alerts):"
git log --since="30 days ago" --grep="fix" --name-only --pretty=format: -- "crates/$CRATE/src/ptop/**/*.rs" 2>/dev/null | \
    sort | uniq -c | sort -rn | head -5 | while read count file; do
    if [ -n "$file" ] && [ "$count" -gt 2 ]; then
        echo "  ⚠️  TARANTULA ALERT: $file fixed $count times"
    elif [ -n "$file" ]; then
        echo "  $count fixes: $file"
    fi
done

echo ""

# 5. SUMMARY
echo "=========================================="
echo "SUMMARY"
echo "=========================================="

# Count PMAT tickets
CLOSED_TICKETS=$(grep -c "CLOSED\|✅" "$SPEC_FILE" 2>/dev/null || echo "0")
OPEN_TICKETS=$(grep -c "OPEN\|❌" "$SPEC_FILE" 2>/dev/null || echo "0")

echo "PMAT Tickets: ~$CLOSED_TICKETS closed, ~$OPEN_TICKETS open"
echo "Coverage: ${TOTAL_COV}% (target: 95%)"
echo "TDG: $AVG_FUNCS funcs/file avg"

if (( $(echo "$TOTAL_COV >= 95" | bc -l) )) && [ "$AVG_FUNCS" -le 30 ]; then
    echo ""
    echo "✅ PMAT VERIFICATION PASSED"
    exit 0
else
    echo ""
    echo "❌ PMAT VERIFICATION FAILED"
    echo "  - Coverage gap: $(echo "95 - $TOTAL_COV" | bc)%"
    exit 1
fi
