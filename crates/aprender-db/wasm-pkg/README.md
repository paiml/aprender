# Trueno-DB Browser Demo

WebAssembly package for in-browser analytics with GPU/SIMD acceleration.

## Features

- **Tiered Compute**: WebGPU → SIMD128 → Scalar fallback
- **SQL Queries**: Execute analytics queries in the browser
- **Zero Dependencies**: No JavaScript charting libraries needed

## Quick Start

### Build

```bash
# From trueno-db root:
make wasm-build-simd
```

### Serve Locally

```bash
make wasm-serve
# Opens http://localhost:8080
```

## Browser Usage

```javascript
import init, { Database, DatabaseConfig, detect_capabilities } from './pkg/trueno_db_wasm.js';

// Initialize WASM module
await init();

// Detect compute capabilities
const caps = await detect_capabilities();
console.log(`Using: ${caps.tier}`);

// Create database
const config = new DatabaseConfig()
    .backend('auto')
    .cache_size_mb(256);

const db = new Database(config);

// Load data (HTTP range requests supported)
await db.load_table('events', '/data/events.parquet');

// Execute query
const result = await db.query('SELECT * FROM events LIMIT 10');
```

## Architecture

```
Browser JS
    ↓
wasm-bindgen
    ↓
trueno-db (Rust)
    ↓
trueno SIMD backend
    ↓
Results (Arrow IPC)
```

## Compute Tiers

| Tier | Description | Speedup |
|------|-------------|---------|
| WebGPU | GPU compute shaders | 10-100x |
| SIMD128 | 128-bit vector ops | 4x |
| Scalar | Fallback | 1x |

## Browser Support

- **Chrome/Edge**: WebGPU + SIMD128
- **Firefox**: SIMD128 (WebGPU experimental)
- **Safari**: SIMD128 (WebGPU coming)

## Memory Considerations

Browsers have ~2GB memory limits. The implementation uses:
- HTTP range requests for streaming Parquet
- Late materialization (defer row reconstruction)
- Out-of-core execution for large datasets

## Development

```bash
# Check WASM compiles
make wasm-check

# Build with SIMD + WebGPU
make wasm-build-simd

# Clean build artifacts
make wasm-clean
```

## Phase 4 Status

- [x] WASM bindings
- [x] Browser demo
- [x] Compute tier detection
- [ ] HTTP range request Parquet reader
- [ ] Late materialization
- [ ] Performance benchmarks

## References

- [DuckDB-WASM](https://duckdb.org/docs/api/wasm/overview): HTTP range request pattern
- [Abadi et al. 2008](https://doi.org/10.1145/1376616.1376712): Late materialization
- [trueno](https://github.com/paiml/trueno): SIMD backend (v0.7.1)
