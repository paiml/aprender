#!/bin/bash
# Build WASM noise generator demo
# Usage: ./scripts/build-wasm-noise.sh

set -e

TARGET="wasm32-unknown-unknown"
FEATURES="audio-noise-wasm"
OUT_DIR="examples/wasm-noise"
CRATE_NAME="aprender"

echo "Building WASM noise generator..."

# Check for wasm-bindgen
if ! command -v wasm-bindgen &> /dev/null; then
    echo "Error: wasm-bindgen not found. Install with:"
    echo "  cargo install wasm-bindgen-cli"
    exit 1
fi

# Build the WASM binary
echo "Compiling for $TARGET..."
cargo build --target $TARGET --features $FEATURES --release

# Generate JS bindings
echo "Generating JS bindings..."
wasm-bindgen target/$TARGET/release/${CRATE_NAME}.wasm \
    --out-dir $OUT_DIR \
    --target web \
    --out-name aprender_noise

# Optimize WASM (optional, requires wasm-opt)
if command -v wasm-opt &> /dev/null; then
    echo "Optimizing WASM binary..."
    wasm-opt -Oz $OUT_DIR/aprender_noise_bg.wasm -o $OUT_DIR/aprender_noise_bg.wasm
else
    echo "Note: wasm-opt not found. Skipping optimization."
    echo "Install binaryen for smaller WASM: apt install binaryen"
fi

# Show output
echo ""
echo "Build complete!"
echo "Output files:"
ls -lh $OUT_DIR/aprender_noise*
echo ""
echo "To run the demo:"
echo "  cd $OUT_DIR && python3 -m http.server 8080"
echo "  Then open: http://localhost:8080"
