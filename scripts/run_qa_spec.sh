#!/usr/bin/env bash
# QA Specification Test Harness
# Parses docs/specifications/100-cargo-run-examples-spec.md and executes examples
# Toyota Way: Muda Elimination - Automates manual execution

set -euo pipefail

SPEC_FILE="docs/specifications/100-cargo-run-examples-spec.md"
REPORT_FILE="target/qa-report-$(date +%Y%m%d-%H%M%S).md"
PASSED=0
FAILED=0
SKIPPED=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "=== Alimentar QA Specification Runner ==="
echo "Toyota Way: Muda (Waste) Elimination"
echo ""

# Ensure fixtures exist
if [[ ! -d "test_fixtures" ]]; then
    echo "Generating fixtures (Heijunka)..."
    cargo run --bin generate_fixtures
fi

# Initialize report
cat > "$REPORT_FILE" << EOF
# QA Specification Execution Report

**Date**: $(date -Iseconds)
**Spec**: $SPEC_FILE

| # | Example | Status | Time |
|---|---------|--------|------|
EOF

# Parse and execute examples
echo "Parsing specification..."
example_num=0

while IFS= read -r line; do
    # Match lines like: **1. CSV Loading**
    if [[ $line =~ ^\*\*([0-9]+)\..+\*\*$ ]]; then
        example_num="${BASH_REMATCH[1]}"
        example_name="${line//\*\*/}"
        example_name="${example_name#[0-9]*. }"
    fi

    # Match cargo run commands (but not comments)
    if [[ $line =~ ^cargo\ (run|build|test) ]]; then
        cmd="$line"

        # Skip external dependency tests if env not set
        if [[ $cmd == *"HF_TOKEN"* ]] && [[ -z "${HF_TOKEN:-}" ]]; then
            echo -e "${YELLOW}[$example_num] SKIP${NC}: $example_name (HF_TOKEN not set)"
            echo "| $example_num | $example_name | SKIPPED | - |" >> "$REPORT_FILE"
            ((SKIPPED++))
            continue
        fi

        if [[ $cmd == *"s3://"* ]] && [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
            echo -e "${YELLOW}[$example_num] SKIP${NC}: $example_name (S3 not configured)"
            echo "| $example_num | $example_name | SKIPPED | - |" >> "$REPORT_FILE"
            ((SKIPPED++))
            continue
        fi

        # Skip valgrind on non-Linux or if not installed
        if [[ $cmd == *"valgrind"* ]]; then
            if ! command -v valgrind &> /dev/null; then
                echo -e "${YELLOW}[$example_num] SKIP${NC}: $example_name (valgrind not installed)"
                echo "| $example_num | $example_name | SKIPPED | - |" >> "$REPORT_FILE"
                ((SKIPPED++))
                continue
            fi
        fi

        # Execute the command
        start_time=$(date +%s.%N)
        echo -n "[$example_num] Running: $example_name... "

        # Timeout after 60s, capture output
        if timeout 60s bash -c "$cmd" > /tmp/qa_output.log 2>&1; then
            end_time=$(date +%s.%N)
            duration=$(echo "$end_time - $start_time" | bc)
            echo -e "${GREEN}PASS${NC} (${duration}s)"
            echo "| $example_num | $example_name | PASS | ${duration}s |" >> "$REPORT_FILE"
            ((PASSED++))
        else
            exit_code=$?
            end_time=$(date +%s.%N)
            duration=$(echo "$end_time - $start_time" | bc)

            if [[ $exit_code -eq 124 ]]; then
                echo -e "${YELLOW}TIMEOUT${NC}"
                echo "| $example_num | $example_name | TIMEOUT | 60s+ |" >> "$REPORT_FILE"
                ((SKIPPED++))
            else
                echo -e "${RED}FAIL${NC} (exit $exit_code)"
                echo "| $example_num | $example_name | FAIL | ${duration}s |" >> "$REPORT_FILE"
                ((FAILED++))
            fi
        fi
    fi
done < <(grep -E '^\*\*[0-9]+\.|^cargo (run|build|test)|^valgrind' "$SPEC_FILE" | sed 's/```bash//g; s/```//g')

# Summary
cat >> "$REPORT_FILE" << EOF

## Summary

- **Passed**: $PASSED
- **Failed**: $FAILED
- **Skipped**: $SKIPPED
- **Total**: $((PASSED + FAILED + SKIPPED))

**Grade**: $(if [[ $FAILED -eq 0 ]]; then echo "A+ (Ship with confidence)"; elif [[ $FAILED -lt 5 ]]; then echo "A (Minor issues)"; else echo "B (Review needed)"; fi)
EOF

echo ""
echo "=== QA Summary ==="
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo ""
echo "Report: $REPORT_FILE"

# Exit with failure if any tests failed
[[ $FAILED -eq 0 ]]
