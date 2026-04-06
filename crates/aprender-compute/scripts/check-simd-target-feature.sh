#!/bin/bash
# Pre-commit hook to detect SIMD intrinsics without #[target_feature] attribute
#
# This script prevents commits that add SIMD intrinsics without the required
# #[target_feature] attribute, which causes the compiler to fail to emit
# SIMD instructions (resulting in significant performance degradation).
#
# Bug Pattern Detected: Missing #[target_feature] on functions using SIMD intrinsics
# Instances Found: 12 (6 sqrt/recip, 6 logarithms) across 2 audits
# Impact: 5.9x regression (recip) to missing 21x speedup (log10)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Checking for SIMD intrinsics without #[target_feature]...${NC}"

# Counter for violations
VIOLATIONS=0
VIOLATION_FILES=()

# Backend files to check
BACKEND_FILES=(
    "src/backends/sse2.rs"
    "src/backends/avx2.rs"
    "src/backends/avx512.rs"
    "src/backends/neon.rs"
    "src/backends/wasm.rs"
)

# SIMD intrinsic patterns to detect
# Format: "pattern:required_attribute:backend_name"
declare -A INTRINSIC_PATTERNS=(
    ["_mm_"]="sse2:SSE2"
    ["_mm256_"]="avx2:AVX2"
    ["_mm512_"]="avx512f:AVX512"
    ["vld1q_f32|vst1q_f32|vaddq_f32|vmulq_f32|vsubq_f32|vdivq_f32"]="neon:NEON"
)

check_file() {
    local file=$1
    local backend=$2

    # Skip if file doesn't exist
    [[ ! -f "$file" ]] && return 0

    # Extract unsafe functions with their preceding line
    # This allows us to check if #[target_feature] is present
    local current_line=0
    local prev_line=""
    local in_unsafe_fn=false
    local unsafe_fn_line=0
    local unsafe_fn_name=""
    local has_target_feature=false

    while IFS= read -r line; do
        ((current_line++))

        # Check if previous line has #[target_feature]
        if [[ "$prev_line" =~ \#\[target_feature ]]; then
            has_target_feature=true
        fi

        # Check for unsafe fn declaration
        if [[ "$line" =~ ^[[:space:]]*unsafe[[:space:]]+fn[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*) ]]; then
            unsafe_fn_name="${BASH_REMATCH[1]}"
            unsafe_fn_line=$current_line
            in_unsafe_fn=true

            # Check what intrinsics are used in this function
            # Read ahead to scan function body
            local fn_body=""
            local brace_count=0
            local found_opening_brace=false

            # Read function body (simplified - reads until closing brace)
            while IFS= read -r body_line; do
                fn_body+="$body_line"$'\n'

                # Count braces to find function end
                if [[ "$body_line" =~ \{ ]]; then
                    found_opening_brace=true
                    ((brace_count++))
                fi
                if [[ "$body_line" =~ \} ]]; then
                    ((brace_count--))
                fi

                # Break if we've closed all braces
                if [[ $found_opening_brace == true && $brace_count -eq 0 ]]; then
                    break
                fi
            done

            # Check each intrinsic pattern
            for pattern in "${!INTRINSIC_PATTERNS[@]}"; do
                IFS=':' read -r feature backend_name <<< "${INTRINSIC_PATTERNS[$pattern]}"

                # Skip if this backend doesn't match the file
                if [[ "$backend" != "$backend_name" ]]; then
                    continue
                fi

                # Check if function body contains SIMD intrinsics
                if echo "$fn_body" | grep -qE "$pattern"; then
                    # Function uses SIMD intrinsics
                    if [[ "$has_target_feature" == false ]]; then
                        # Missing #[target_feature] attribute!
                        echo -e "${RED}❌ VIOLATION FOUND${NC}"
                        echo -e "   File: ${YELLOW}$file:$unsafe_fn_line${NC}"
                        echo -e "   Function: ${YELLOW}$unsafe_fn_name${NC}"
                        echo -e "   Problem: Uses ${YELLOW}$backend_name${NC} intrinsics without ${YELLOW}#[target_feature(enable = \"$feature\")]${NC}"
                        echo ""
                        ((VIOLATIONS++))
                        VIOLATION_FILES+=("$file:$unsafe_fn_line:$unsafe_fn_name")
                        break
                    fi
                fi
            done

            # Reset for next function
            has_target_feature=false
        fi

        prev_line="$line"
    done < "$file"
}

# Check each backend file
for file in "${BACKEND_FILES[@]}"; do
    # Extract backend name from filename
    backend=$(basename "$file" .rs | tr '[:lower:]' '[:upper:]')

    # Map backend names to check patterns
    case "$backend" in
        SSE2) check_file "$file" "SSE2" ;;
        AVX2) check_file "$file" "AVX2" ;;
        AVX512) check_file "$file" "AVX512" ;;
        NEON) check_file "$file" "NEON" ;;
        WASM) check_file "$file" "WASM" ;;
    esac
done

# Report results
echo ""
if [ $VIOLATIONS -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: No SIMD intrinsics without #[target_feature] detected${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ COMMIT BLOCKED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Found $VIOLATIONS function(s) using SIMD intrinsics without #[target_feature]${NC}"
    echo ""
    echo -e "${BLUE}Why this matters:${NC}"
    echo "  Without #[target_feature], the Rust compiler CANNOT emit SIMD instructions,"
    echo "  even though the code compiles successfully. This causes severe performance"
    echo "  degradation (5.9x slower observed) or missing speedups (21x potential lost)."
    echo ""
    echo -e "${BLUE}How to fix:${NC}"
    echo "  Add the appropriate #[target_feature] attribute above each unsafe function"
    echo "  that uses SIMD intrinsics:"
    echo ""
    echo "  ${GREEN}// BEFORE (BROKEN)${NC}"
    echo "  ${RED}unsafe fn my_function(a: &[f32]) {${NC}"
    echo "  ${RED}    let vec = _mm256_loadu_ps(...);${NC}"
    echo "  ${RED}}${NC}"
    echo ""
    echo "  ${GREEN}// AFTER (FIXED)${NC}"
    echo "  ${GREEN}#[target_feature(enable = \"avx2\")]  // ← ADD THIS${NC}"
    echo "  ${GREEN}unsafe fn my_function(a: &[f32]) {${NC}"
    echo "  ${GREEN}    let vec = _mm256_loadu_ps(...);${NC}"
    echo "  ${GREEN}}${NC}"
    echo ""
    echo -e "${BLUE}Required attributes by backend:${NC}"
    echo "  SSE2:   #[target_feature(enable = \"sse2\")]"
    echo "  AVX2:   #[target_feature(enable = \"avx2\")]"
    echo "  AVX512: #[target_feature(enable = \"avx512f\")]"
    echo "  NEON:   #[target_feature(enable = \"neon\")]"
    echo ""
    echo -e "${BLUE}Violations found:${NC}"
    for violation in "${VIOLATION_FILES[@]}"; do
        echo "  • $violation"
    done
    echo ""
    echo -e "${YELLOW}Fix these issues and try committing again.${NC}"
    echo ""
    exit 1
fi
