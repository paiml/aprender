# Batuta WASM Example

This example demonstrates how to use Batuta compiled to WebAssembly in a web browser.

## Features

- **Language Detection**: Analyze code snippets and detect languages
- **NumPy → Trueno**: Convert NumPy operations to Rust Trueno equivalents
- **sklearn → Aprender**: Convert sklearn algorithms to Rust Aprender
- **PyTorch → Realizar**: Convert PyTorch operations to Rust Realizar
- **Backend Selection**: Get optimal compute backend recommendations

## Building

### Prerequisites

```bash
# Install WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-bindgen-cli
cargo install wasm-bindgen-cli

# Optional: Install wasm-opt for optimization
cargo install wasm-opt
```

### Build Commands

```bash
# Debug build
make wasm
# or
./scripts/build-wasm.sh debug

# Release build (optimized)
make wasm-release
# or
./scripts/build-wasm.sh release
```

The build output will be in the `wasm-dist/` directory.

## Running the Demo

1. **Build the WASM module** (see above)

2. **Start a local web server**:
   ```bash
   # From the project root
   python3 -m http.server 8080
   ```

3. **Open your browser**:
   ```
   http://localhost:8080/wasm-dist/index.html
   ```

## API Reference

### JavaScript API

```javascript
import init, {
    analyze_code,
    convert_numpy,
    convert_sklearn,
    convert_pytorch,
    backend_recommend,
    version
} from './batuta.js';

// Initialize WASM module
await init();

// Analyze code
const analysis = analyze_code("import numpy as np\nx = np.array([1, 2, 3])");
console.log(analysis.language); // "Python"
console.log(analysis.has_numpy); // true

// Convert NumPy to Trueno
const numpy_result = convert_numpy("np.add(a, b)", 10000);
console.log(numpy_result.rust_code);
console.log(numpy_result.backend_recommendation); // "SIMD" or "GPU"

// Convert sklearn to Aprender
const sklearn_result = convert_sklearn("LinearRegression()", 1000);
console.log(sklearn_result.rust_code);

// Convert PyTorch to Realizar
const pytorch_result = convert_pytorch("model.generate()", 7000000000);
console.log(pytorch_result.rust_code);

// Get backend recommendation
const backend = backend_recommend("matmul", 1024);
console.log(backend); // "SIMD" or "GPU"

// Get version
const ver = version();
console.log(ver); // "0.1.0"
```

## Architecture

### Feature Flags

The WASM build uses a different set of features than the native build:

- **Native build**: `--features native` (includes CLI, file system, tracing)
- **WASM build**: `--features wasm` (JavaScript interop, browser APIs only)

### API Design

The WASM API is designed to work with in-memory code snippets rather than files:

- ✅ **Supported**: Code analysis, conversion mapping, backend recommendations
- ❌ **Not supported**: File system operations, subprocess execution, syscall tracing

### Size Optimization

Release builds include several optimizations:

1. **wasm-bindgen**: Generates minimal JavaScript glue code
2. **wasm-opt -Oz**: Aggressive size optimization
3. **LTO + strip**: Link-time optimization and symbol stripping

Expected sizes:
- Debug: ~2-3 MB
- Release: ~500-800 KB

## Browser Compatibility

Batuta WASM requires:
- **WebAssembly support** (all modern browsers)
- **ES6 modules** (Chrome 61+, Firefox 60+, Safari 11+, Edge 16+)

## Limitations

Due to WASM security model:

- **No file system access**: Work with in-memory code only
- **No subprocess execution**: Can't run external tools
- **No native code**: All operations must be pure Rust
- **No syscall tracing**: Renacer features unavailable

## Integration Examples

### React

```javascript
import React, { useEffect, useState } from 'react';
import init, { analyze_code } from './batuta.js';

function CodeAnalyzer() {
    const [ready, setReady] = useState(false);

    useEffect(() => {
        init().then(() => setReady(true));
    }, []);

    const analyze = (code) => {
        if (!ready) return;
        const result = analyze_code(code);
        console.log(result);
    };

    return <div>{/* UI */}</div>;
}
```

### Node.js

```javascript
// Use Node.js target instead of web target
// Build with: wasm-bindgen --target nodejs

const { analyze_code } = require('./batuta.js');

const code = "import numpy as np\nx = np.array([1, 2, 3])";
const result = analyze_code(code);
console.log(result);
```

## Performance

WASM performance is typically:
- **70-90%** of native Rust performance
- **2-10×** faster than equivalent JavaScript
- **Best for**: CPU-intensive analysis and conversion logic
- **Consider native**: For file system-heavy workflows

## Troubleshooting

### WASM module fails to load

- Check browser console for errors
- Ensure web server serves `.wasm` files with correct MIME type
- Verify `batuta.js` and `batuta_bg.wasm` are in the same directory

### API functions not available

- Ensure `init()` completes before calling API functions
- Check for JavaScript errors in browser console
- Verify WASM feature flag was used during build

### Performance issues

- Use release builds (`make wasm-release`)
- Enable gzip compression on your web server
- Consider caching strategy for WASM module

## Contributing

Found a bug or want to add features? See [CONTRIBUTING.md](../../CONTRIBUTING.md).

## License

MIT - See [LICENSE](../../LICENSE) for details.
