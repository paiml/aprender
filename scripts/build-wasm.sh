#!/usr/bin/env bash
# Build Batuta for WebAssembly target
#
# This script compiles Batuta to WASM and generates JavaScript bindings
# using wasm-bindgen. The output can be used in web browsers and Node.js.
#
# Usage:
#   ./scripts/build-wasm.sh [release]
#
# Options:
#   release - Build with optimizations (default: debug)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build mode (debug or release)
BUILD_MODE="${1:-debug}"
OUTPUT_DIR="wasm-dist"

echo -e "${BLUE}🦀 Building Batuta for WebAssembly${NC}"
echo "================================="
echo

# Check if wasm32 target is installed
if ! rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
    echo -e "${YELLOW}⚠️  wasm32-unknown-unknown target not installed${NC}"
    echo "Installing..."
    rustup target add wasm32-unknown-unknown
    echo -e "${GREEN}✓ Target installed${NC}"
    echo
fi

# Check if wasm-bindgen-cli is installed
if ! command -v wasm-bindgen &> /dev/null; then
    echo -e "${YELLOW}⚠️  wasm-bindgen-cli not found${NC}"
    echo "Install with: cargo install wasm-bindgen-cli"
    exit 1
fi

# Check if wasm-opt is installed (for optimization)
if ! command -v wasm-opt &> /dev/null; then
    echo -e "${YELLOW}⚠️  wasm-opt not found (optional, but recommended for release builds)${NC}"
    echo "Install with: cargo install wasm-opt"
    echo
fi

# Build for WASM
echo -e "${BLUE}📦 Compiling to WASM...${NC}"
if [ "$BUILD_MODE" = "release" ]; then
    cargo build --target wasm32-unknown-unknown \
        --no-default-features \
        --features wasm \
        --release
    WASM_FILE="target/wasm32-unknown-unknown/release/batuta.wasm"
else
    cargo build --target wasm32-unknown-unknown \
        --no-default-features \
        --features wasm
    WASM_FILE="target/wasm32-unknown-unknown/debug/batuta.wasm"
fi

echo -e "${GREEN}✓ WASM compilation complete${NC}"
echo

# Run wasm-bindgen to generate JavaScript bindings
echo -e "${BLUE}🔗 Generating JavaScript bindings...${NC}"

mkdir -p "$OUTPUT_DIR"

wasm-bindgen "$WASM_FILE" \
    --out-dir "$OUTPUT_DIR" \
    --target web \
    --no-typescript

echo -e "${GREEN}✓ JavaScript bindings generated${NC}"
echo

# Optimize WASM if in release mode and wasm-opt is available
if [ "$BUILD_MODE" = "release" ] && command -v wasm-opt &> /dev/null; then
    echo -e "${BLUE}⚡ Optimizing WASM...${NC}"
    wasm-opt -Oz "$OUTPUT_DIR/batuta_bg.wasm" \
        -o "$OUTPUT_DIR/batuta_bg_opt.wasm"
    mv "$OUTPUT_DIR/batuta_bg_opt.wasm" "$OUTPUT_DIR/batuta_bg.wasm"
    echo -e "${GREEN}✓ WASM optimization complete${NC}"
    echo
fi

# Display size information
echo -e "${BLUE}📊 Build Statistics${NC}"
echo "-------------------"
WASM_SIZE=$(du -h "$OUTPUT_DIR/batuta_bg.wasm" | cut -f1)
echo "WASM file size: $WASM_SIZE"
echo

# Copy example HTML if it exists
if [ -f "examples/wasm/index.html" ]; then
    cp examples/wasm/index.html "$OUTPUT_DIR/"
    echo -e "${GREEN}✓ Copied example HTML to $OUTPUT_DIR/${NC}"
fi

# Success message
echo -e "${GREEN}✅ Build complete!${NC}"
echo
echo "Output directory: $OUTPUT_DIR/"
echo "Files generated:"
echo "  - batuta_bg.wasm       (WebAssembly binary)"
echo "  - batuta.js            (JavaScript bindings)"
echo
echo "To test in a browser:"
echo "  1. Start a local web server: python3 -m http.server 8080"
echo "  2. Open http://localhost:8080/$OUTPUT_DIR/index.html"
echo
