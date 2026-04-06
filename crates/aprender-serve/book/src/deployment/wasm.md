# WebAssembly Deployment

Deploy realizar to edge environments via WebAssembly.

## WasmConfig

```rust
pub struct WasmConfig {
    /// Target runtime
    pub runtime: WasmRuntime,
    /// Enable SIMD (experimental)
    pub enable_simd: bool,
    /// Maximum memory pages (64KB each)
    pub max_memory_pages: u32,
    /// Optimization level
    pub opt_level: WasmOptLevel,
}

pub enum WasmRuntime {
    Browser,
    CloudflareWorkers,
    Deno,
    Node,
}

pub enum WasmOptLevel {
    Debug,
    Release,
    Size,  // Optimize for binary size
}
```

## Capabilities

WASM deployments have limited capabilities:

| Feature | Support | Notes |
|---------|---------|-------|
| SIMD | ⚠️ | Limited, requires `--enable-simd` |
| Threads | ❌ | SharedArrayBuffer not available |
| Filesystem | ❌ | Use fetch/KV store |
| GPU | ❌ | WebGPU future possibility |
| Async I/O | ✅ | Via JavaScript promises |
| Max Memory | 128MB | Cloudflare Workers limit |

## Cloudflare Workers

Generate a Cloudflare Worker:

```rust
let config = WasmConfig {
    runtime: WasmRuntime::CloudflareWorkers,
    enable_simd: false,
    max_memory_pages: 2048,  // 128MB
    opt_level: WasmOptLevel::Size,
};

let worker_js = config.generate_worker();
```

Output (`worker.js`):

```javascript
import init, { predict } from './realizar.js';

export default {
    async fetch(request, env, ctx) {
        await init();

        if (request.method !== 'POST') {
            return new Response('Method not allowed', { status: 405 });
        }

        const body = await request.json();
        const result = predict(body.features);

        return new Response(JSON.stringify(result), {
            headers: { 'Content-Type': 'application/json' }
        });
    }
};
```

## Build for WASM

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for Cloudflare Workers
wasm-pack build --target web --release

# Optimize binary size
wasm-opt -Os pkg/realizar_bg.wasm -o pkg/realizar_bg.wasm
```

## wrangler.toml

```toml
name = "realizar-worker"
main = "worker.js"
compatibility_date = "2024-01-01"

[build]
command = "wasm-pack build --target web --release"

[[rules]]
type = "CompiledWasm"
globs = ["**/*.wasm"]

[vars]
MODEL_VERSION = "1.0.0"
```

## Performance Considerations

WASM inference is slower than native due to:

1. **No SIMD**: ~3-5x slower matrix operations
2. **Single-threaded**: No parallel batch processing
3. **Memory limits**: 128MB max on Workers
4. **JIT warmup**: First invocations are slow

### When to Use WASM

| Use Case | Recommendation |
|----------|----------------|
| Edge inference (<10ms latency) | ✅ Cloudflare Workers |
| High throughput | ❌ Use Lambda or Docker |
| Large models (>50MB) | ❌ Use Lambda or Docker |
| Browser ML | ✅ With small models |

## Testing WASM Builds

```bash
# Run WASM tests
wasm-pack test --headless --firefox

# Local development server
npx wrangler dev
```
