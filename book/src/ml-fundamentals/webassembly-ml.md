# WebAssembly for Machine Learning

WebAssembly (WASM) enables running ML models in browsers and edge devices with near-native performance.

## Why WASM for ML?

| Deployment | Traditional | WASM |
|------------|-------------|------|
| Browser | JavaScript (slow) | Near-native |
| Edge | Native binary per platform | Single binary |
| Security | Full system access | Sandboxed |
| Distribution | App store review | Instant deploy |

## WASM Architecture

```
┌─────────────────────────────────────────┐
│              Host Environment           │
│  ┌─────────────────────────────────┐   │
│  │         WASM Runtime            │   │
│  │  ┌───────────────────────────┐  │   │
│  │  │      WASM Module          │  │   │
│  │  │  ┌─────┐  ┌─────────┐    │  │   │
│  │  │  │Stack│  │ Linear  │    │  │   │
│  │  │  │     │  │ Memory  │    │  │   │
│  │  │  └─────┘  └─────────┘    │  │   │
│  │  └───────────────────────────┘  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Compiling Rust to WASM

### Setup

```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack for JS bindings
cargo install wasm-pack
```

### Build

```bash
# Pure WASM
cargo build --target wasm32-unknown-unknown --release

# With JS bindings
wasm-pack build --target web
```

### Cargo.toml

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
getrandom = { version = "0.2", features = ["js"] }

[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { version = "0.2", features = ["js"] }
```

## Memory Considerations

### Linear Memory

WASM has one contiguous memory buffer:

```rust
// Pass large arrays efficiently
#[wasm_bindgen]
pub fn predict(data: &[f32]) -> Vec<f32> {
    // data points directly into WASM memory
    model.forward(data)
}
```

### Memory Limits

| Browser | Default | Max |
|---------|---------|-----|
| Chrome | 2GB | 4GB |
| Firefox | 2GB | 4GB |
| Safari | 2GB | 4GB |

Plan for models < 2GB in-browser.

## SIMD in WASM

WASM SIMD provides 128-bit vectors:

```rust
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

// 4x f32 operations
let a = f32x4(1.0, 2.0, 3.0, 4.0);
let b = f32x4(5.0, 6.0, 7.0, 8.0);
let c = f32x4_add(a, b);
```

**Speedup:** 2-4x for vectorizable operations.

### Browser Support

| Feature | Chrome | Firefox | Safari |
|---------|--------|---------|--------|
| WASM | ✅ | ✅ | ✅ |
| SIMD | ✅ (91+) | ✅ (89+) | ✅ (16.4+) |
| Threads | ✅ | ✅ | ✅ (15+) |

## Threading

WASM threads require SharedArrayBuffer:

```javascript
// Check support
if (crossOriginIsolated) {
    // Can use threads
}
```

**Security headers required:**
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Model Loading

### From URL

```javascript
const modelUrl = 'model.wasm';
const response = await fetch(modelUrl);
const wasmModule = await WebAssembly.instantiateStreaming(response);
```

### From Bytes

```javascript
const bytes = new Uint8Array(modelData);
const module = await WebAssembly.instantiate(bytes);
```

### Lazy Loading

```javascript
// Load model on demand
let model = null;
async function getModel() {
    if (!model) {
        model = await loadModel();
    }
    return model;
}
```

## Performance Optimization

### Minimize JS/WASM Boundary

```rust
// ❌ Many small calls
for i in 0..1000 {
    js_call(data[i]);
}

// ✅ Batch operations
process_batch(&data[0..1000]);
```

### Use Typed Arrays

```javascript
// ❌ Regular array (copy required)
const input = [1.0, 2.0, 3.0];

// ✅ Float32Array (zero-copy)
const input = new Float32Array([1.0, 2.0, 3.0]);
```

### Pre-allocate Memory

```rust
#[wasm_bindgen]
pub struct Model {
    // Pre-allocated buffers
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
}
```

## WebGPU Integration

Future: WASM + WebGPU for GPU inference:

```javascript
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Use GPU for matrix operations
const buffer = device.createBuffer({...});
```

## Deployment Patterns

### Static Hosting

```
/index.html
/app.js
/model.wasm
/model_bg.wasm  (if using wasm-pack)
```

### CDN Distribution

```html
<script type="module">
import init, { Model } from 'https://cdn.example.com/model/model.js';
await init();
const model = new Model();
</script>
```

### Service Worker Cache

```javascript
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open('model-v1').then((cache) => {
            return cache.addAll(['/model.wasm']);
        })
    );
});
```

## Limitations

| Feature | Status |
|---------|--------|
| File system | ❌ (use IndexedDB) |
| Network | Via fetch API |
| GPU | WebGPU (emerging) |
| Threading | Requires special headers |
| Memory | 4GB max |

## References

- WebAssembly Specification: <https://webassembly.org>
- wasm-bindgen: <https://rustwasm.github.io/wasm-bindgen/>
- WebAssembly SIMD: <https://v8.dev/features/simd>
