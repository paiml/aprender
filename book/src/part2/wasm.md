# WebAssembly (WASM) Build Target

> **"Batuta in the browser: Analyze, convert, and optimize code without leaving your documentation or web IDE."**

## Overview

Batuta can be compiled to **WebAssembly (WASM)** to run directly in web browsers, enabling client-side code analysis, conversion demonstrations, and interactive documentation. This brings Batuta's core capabilities to:

- **Interactive documentation** with live code conversion examples
- **Web-based IDEs** integrating Batuta's analysis engine
- **Educational platforms** demonstrating transpilation techniques
- **Browser extensions** for code quality analysis
- **Offline-first web applications** without server-side dependencies

## Why WASM?

Running Batuta in the browser provides several advantages:

### **1. Zero Server Costs**
All analysis and conversion happens client-side. No need for backend infrastructure to demonstrate transpilation capabilities.

### **2. Instant Feedback**
No network latency - code analysis and conversion results appear immediately as users type.

### **3. Privacy**
User code never leaves their browser. Perfect for proprietary code analysis or security-sensitive environments.

### **4. Educational Value**
Interactive examples in documentation allow users to experiment with Batuta's features before installing.

### **5. Integration Flexibility**
Embed Batuta into React, Vue, or vanilla JavaScript applications as a lightweight library.

## Building for WASM

### Prerequisites

Install the WASM toolchain:

```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-bindgen CLI (matches Cargo.toml version)
cargo install wasm-bindgen-cli --version 0.2.89

# Install wasm-opt for size optimization (optional)
cargo install wasm-opt
```

### Quick Build

Use the provided build script:

```bash
# Debug build (faster compilation, larger size)
./scripts/build-wasm.sh debug

# Release build (optimized, ~500-800 KB)
./scripts/build-wasm.sh release
```

The script will:
1. Compile Rust to WASM (`wasm32-unknown-unknown` target)
2. Generate JavaScript bindings (`wasm-bindgen`)
3. Optimize WASM binary (`wasm-opt -Oz`)
4. Copy browser demo files to `wasm-dist/`

### Manual Build

For custom builds:

```bash
# Build WASM module
cargo build --target wasm32-unknown-unknown \
    --no-default-features \
    --features wasm \
    --release

# Generate JavaScript bindings
wasm-bindgen target/wasm32-unknown-unknown/release/batuta.wasm \
    --out-dir wasm-dist \
    --target web \
    --no-typescript

# Optimize (optional, reduces size by 30-50%)
wasm-opt -Oz wasm-dist/batuta_bg.wasm \
    -o wasm-dist/batuta_bg_opt.wasm
```

### Build Output

After building, `wasm-dist/` contains:

```
wasm-dist/
├── batuta.js              # JavaScript glue code
├── batuta_bg.wasm         # WASM module (~1.5 MB debug)
├── batuta_bg_opt.wasm     # Optimized WASM (~500 KB release)
├── index.html             # Interactive demo
└── README.md              # Integration guide
```

## JavaScript API

Batuta exposes a JavaScript-friendly API via `wasm-bindgen`. All functions are asynchronous and return Promises.

### Initialization

```javascript
import init, * as batuta from './batuta.js';

// Initialize WASM module (call once)
await init();

// Module is ready to use
console.log('Batuta version:', batuta.version());
```

### Code Analysis

Detect language and ML library usage:

```javascript
const code = `
import numpy as np
import sklearn.linear_model as lm

X = np.array([[1, 2], [3, 4]])
model = lm.LinearRegression()
`;

const analysis = batuta.analyze_code(code);

console.log(analysis);
// Output:
// {
//   language: "Python",
//   has_numpy: true,
//   has_sklearn: true,
//   has_pytorch: false,
//   lines_of_code: 5
// }
```

### NumPy Conversion

Convert NumPy operations to Trueno:

```javascript
const numpy_code = "np.add(a, b)";
const data_size = 10000;

const result = batuta.convert_numpy(numpy_code, data_size);

console.log(result);
// Output:
// {
//   rust_code: "trueno::add(&a, &b)",
//   imports: ["use trueno;"],
//   backend_recommendation: "SIMD",
//   explanation: "Array addition using SIMD vectorization"
// }
```

For GPU-scale operations:

```javascript
const large_matmul = "np.dot(a, b)";
const gpu_size = 1000000;

const result = batuta.convert_numpy(large_matmul, gpu_size);

// backend_recommendation: "GPU"
// Uses trueno's CUDA/Metal backend for large matrices
```

### sklearn Conversion

Convert scikit-learn to Aprender:

```javascript
const sklearn_code = "LinearRegression()";

const result = batuta.convert_sklearn(sklearn_code, 5000);

console.log(result);
// Output:
// {
//   rust_code: "aprender::LinearRegression::new()",
//   imports: ["use aprender::LinearRegression;"],
//   backend_recommendation: "CPU",
//   explanation: "First-principles linear regression implementation"
// }
```

Supported algorithms:
- **Linear Models**: `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`
- **Clustering**: `KMeans`, `DBSCAN`
- **Ensemble**: `RandomForest` (limited support)
- **Preprocessing**: `StandardScaler`, `MinMaxScaler`

### PyTorch Conversion

Convert PyTorch inference to Realizar:

```javascript
const pytorch_code = "model.generate(prompt, max_length=100)";

const result = batuta.convert_pytorch(pytorch_code, 2000);

console.log(result);
// Output:
// {
//   rust_code: "realizar::generate_text(&model, prompt, 100)",
//   imports: ["use realizar;"],
//   backend_recommendation: "GPU",
//   explanation: "Optimized LLM inference with KV cache"
// }
```

### Backend Recommendation

Get MoE backend selection for specific operations:

```javascript
// Small dataset → CPU
const backend1 = batuta.backend_recommend("matrix_multiply", 1000);
console.log(backend1); // "CPU"

// Medium dataset → SIMD
const backend2 = batuta.backend_recommend("matrix_multiply", 50000);
console.log(backend2); // "SIMD"

// Large dataset → GPU
const backend3 = batuta.backend_recommend("matrix_multiply", 1000000);
console.log(backend3); // "GPU"
```

Supported operation types:
- `"matrix_multiply"` - Dense matrix multiplication
- `"element_wise"` - Element-wise operations (add, sub, mul)
- `"reduction"` - Sum, mean, max, min
- `"dot_product"` - Vector dot products
- `"convolution"` - 2D convolutions (CNN)
- `"linear_regression"` - ML training
- `"kmeans"` - Clustering
- `"text_generation"` - LLM inference

## Browser Integration

### Vanilla JavaScript

```html
<!DOCTYPE html>
<html>
<head>
    <title>Batuta WASM Demo</title>
</head>
<body>
    <textarea id="code" rows="10" cols="80">
import numpy as np
x = np.array([1, 2, 3])
    </textarea>
    <button onclick="analyzeCode()">Analyze</button>
    <pre id="output"></pre>

    <script type="module">
        import init, * as batuta from './batuta.js';

        await init();

        window.analyzeCode = async () => {
            const code = document.getElementById('code').value;
            const result = batuta.analyze_code(code);
            document.getElementById('output').textContent =
                JSON.stringify(result, null, 2);
        };
    </script>
</body>
</html>
```

### React Integration

```jsx
import { useEffect, useState } from 'react';
import init, * as batuta from './batuta.js';

function BatutaConverter() {
    const [initialized, setInitialized] = useState(false);
    const [code, setCode] = useState('');
    const [result, setResult] = useState(null);

    useEffect(() => {
        init().then(() => setInitialized(true));
    }, []);

    const handleConvert = () => {
        if (!initialized) return;

        const analysis = batuta.analyze_code(code);
        if (analysis.has_numpy) {
            const conversion = batuta.convert_numpy(code, 10000);
            setResult(conversion);
        }
    };

    return (
        <div>
            <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="Paste NumPy code here..."
            />
            <button onClick={handleConvert} disabled={!initialized}>
                Convert to Rust
            </button>
            {result && (
                <pre>{result.rust_code}</pre>
            )}
        </div>
    );
}
```

### Vue Integration

```vue
<template>
    <div>
        <textarea v-model="code"></textarea>
        <button @click="analyze" :disabled="!ready">
            Analyze
        </button>
        <pre v-if="analysis">{{ analysis }}</pre>
    </div>
</template>

<script>
import init, * as batuta from './batuta.js';

export default {
    data() {
        return {
            ready: false,
            code: '',
            analysis: null
        };
    },
    async mounted() {
        await init();
        this.ready = true;
    },
    methods: {
        analyze() {
            this.analysis = batuta.analyze_code(this.code);
        }
    }
};
</script>
```

## Feature Flags

Batuta uses conditional compilation to support both native and WASM builds:

```toml
# Cargo.toml
[features]
default = ["native"]

native = [
    "clap",           # CLI parsing
    "walkdir",        # Filesystem traversal
    "tracing",        # Logging
    "serde_yaml",     # Config files
    # ... native-only dependencies
]

wasm = [
    "wasm-bindgen",       # JS bindings
    "wasm-bindgen-futures",
    "js-sys",             # JavaScript types
    "web-sys",            # Web APIs
]
```

This allows:
- **Native builds**: Full CLI with file I/O, logging, process spawning
- **WASM builds**: Browser-safe API with in-memory operations

## Limitations

The WASM build has intentional limitations compared to the native CLI:

### **No Filesystem Access**
- ❌ Cannot read/write files directly
- ✅ Works with in-memory code strings
- **Workaround**: Use File API in browser to read user-selected files

### **No Process Spawning**
- ❌ Cannot call external transpilers (Decy, Depyler, Bashrs)
- ✅ Can analyze code and recommend conversions
- **Workaround**: Use WASM for analysis, native CLI for actual transpilation

### **No Logging Infrastructure**
- ❌ No `tracing` or `env_logger` support
- ✅ Uses JavaScript `console.log()` via `web-sys`
- **Workaround**: Stub macros for logging (`info!`, `debug!`, etc.)

### **Synchronous-Only API**
- ❌ No async file I/O or network requests
- ✅ All API calls are instant (no disk I/O)
- **Workaround**: Use Web Workers for long-running analysis

### **Size Constraints**
- Release WASM binary: ~500-800 KB (after `wasm-opt -Oz`)
- Debug binary: ~1.5-2 MB
- **Optimization**: Use `wasm-opt`, enable LTO, strip debug symbols

## Capabilities

Despite limitations, WASM builds support:

✅ **Language Detection**: Identify Python, C, C++, Shell, Rust, JavaScript
✅ **ML Library Detection**: Recognize NumPy, sklearn, PyTorch usage
✅ **Code Conversion**: Generate Rust equivalents for ML operations
✅ **Backend Selection**: MoE-based compute backend recommendations
✅ **Quality Analysis**: Complexity estimation (without full PMAT)
✅ **Interactive Demos**: Real-time code analysis in documentation

## Size Optimization

Reduce WASM binary size:

### **1. Use `wasm-opt`**
```bash
wasm-opt -Oz input.wasm -o output.wasm
```
Savings: **30-50%** reduction in file size.

### **2. Enable LTO**
```toml
# Cargo.toml
[profile.release]
lto = true
codegen-units = 1
opt-level = "z"  # Optimize for size
```

### **3. Strip Debug Symbols**
```toml
[profile.release]
strip = true
debug = false
```

### **4. Remove Unused Features**
Only include necessary WASM features:

```toml
[dependencies.web-sys]
features = [
    "console",  # Only if logging needed
    # Omit unused features like "Window", "Document", etc.
]
```

### **5. Use `wee_alloc`**
Smaller allocator for WASM:

```toml
[dependencies]
wee_alloc = "0.4"
```

```rust
#[cfg(feature = "wasm")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
```

Savings: **10-20 KB** reduction.

## Deployment

### **Static Hosting**
Serve WASM files from any static host:

```bash
# GitHub Pages
cp -r wasm-dist/* docs/demo/

# Netlify
netlify deploy --dir=wasm-dist

# Vercel
vercel wasm-dist/
```

### **CDN Distribution**
Use a CDN for faster global access:

```html
<script type="module">
    import init from 'https://cdn.example.com/batuta/batuta.js';
    await init('https://cdn.example.com/batuta/batuta_bg.wasm');
</script>
```

### **npm Package**
Publish as an npm package:

```json
{
  "name": "@paiml/batuta-wasm",
  "version": "0.1.0",
  "files": ["batuta.js", "batuta_bg.wasm"],
  "main": "batuta.js",
  "type": "module"
}
```

Users can install via:
```bash
npm install @paiml/batuta-wasm
```

## Practical Use Cases

### **1. Interactive Documentation**
Embed live code examples in Batuta's docs:

```markdown
Try converting NumPy code to Trueno:

<textarea id="numpy-input">np.dot(a, b)</textarea>
<button onclick="convertNumpy()">Convert</button>
<pre id="rust-output"></pre>
```

### **2. Web-Based Code Review**
Build a browser extension that analyzes Python code for migration potential:

```javascript
// Chrome extension content script
const code = getSelectedCodeFromGitHub();
const analysis = batuta.analyze_code(code);

if (analysis.has_numpy) {
    showMigrationSuggestion("This code can be 10x faster with Trueno!");
}
```

### **3. Educational Platforms**
Interactive Rust learning platform:

- Students paste Python code
- Batuta generates Rust equivalent
- Side-by-side comparison with explanations
- Instant feedback without server costs

### **4. Code Quality Dashboards**
Real-time complexity analysis:

```javascript
const files = await loadProjectFiles();
const analyses = files.map(f => batuta.analyze_code(f.content));

const avgComplexity = analyses.reduce((sum, a) =>
    sum + a.lines_of_code, 0) / analyses.length;

renderDashboard({ avgComplexity, mlLibraries: ... });
```

### **5. Offline-First Migration Tool**
Progressive Web App (PWA) for code migration:

- Works without internet connection
- Stores project state in IndexedDB
- Generates Rust code locally
- Syncs to cloud when online

## Testing WASM Builds

Run WASM-specific tests:

```bash
# Run tests targeting WASM
cargo test --target wasm32-unknown-unknown \
    --no-default-features \
    --features wasm \
    --lib

# Run in headless browser (requires wasm-pack)
wasm-pack test --headless --firefox
```

Add WASM-specific tests:

```rust
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_analyze_python() {
        let code = "import numpy as np";
        let result = analyze_code(code).unwrap();
        assert_eq!(result.language, "Python");
        assert!(result.has_numpy);
    }
}
```

## Next Steps

- **[Tool Selection](./tool-selection.md)**: How Batuta selects transpilers
- **[MoE Backend Selection](./moe.md)**: Mixture-of-Experts algorithm details
- **[Phase 3: Optimization](./phase3-optimization.md)**: Backend-specific optimizations

---

**Navigate:** [Table of Contents](../SUMMARY.md)
